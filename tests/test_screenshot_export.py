import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import pytest
import torch
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/export_screenshot_data.py'


def exporter_module():
    spec = importlib.util.spec_from_file_location('export_screenshot', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_preserves_screenshot_inputs_loss_and_independence(tmp_path):
    from framediff.ir import node
    from framediff.tree_edits import FIELDS
    from framediff.tree_policy import TreeConfig, TreePolicy, batch_inputs
    torch.set_num_threads(2)
    source = tmp_path/'old';assets = source/'master';assets.mkdir(parents=True)
    Image.new('RGB', (100,100), 'orange').save(assets/'target.png')
    Image.new('RGB', (100,100), 'green').save(assets/'current.png')
    (assets/'page.html').write_text('<html><body>Hello</body></html>')
    row = dict(id='page/clean', group='page', split='train', source_sha='unchanged',
               viewport=[100,100], current={'version':1,'nodes':[node('root',None,'page'),node('a','root','card')]},
               current_boxes={'root':[0,0,100,100],'a':[0,0,50,50]},
               target_image=str(assets/'target.png'),current_image=str(assets/'current.png'),
               current_html=str(assets/'page.html'),target_html=str(assets/'page.html'),
               declaration_state=[[['',''] for _ in FIELDS] for _ in range(2)],
               replacement_edit=[1,'width','50%',''])
    folder = source/'7000/css';folder.mkdir(parents=True)
    for split in ('train','val','test'):
        for kind in ('policy','pages'):
            (folder/f'{kind}-{split}.jsonl').write_text(json.dumps({**row,'split':split})+'\n')
    cfg = TreeConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=8,mode='screenshot')
    net = TreePolicy(cfg,False).eval()
    original = batch_inputs(net,[row],'cpu')
    original_loss = net.loss(*original,[row['replacement_edit']])[0]
    original_loss.backward()
    gradients = {name:p.grad.clone() for name,p in net.named_parameters() if p.grad is not None}
    out=tmp_path/'new'
    command=[sys.executable,str(SCRIPT),'--data',str(source),'--out',str(out),'--sizes','7000']
    subprocess.run(command,check=True,capture_output=True)
    assert (out/'READY').exists()
    exported=json.loads((out/'7000/css/policy-train.jsonl').read_text())
    assert os.path.samefile(exported['target_image'], row['target_image'])
    assert os.path.samefile(exported['target_html'], row['target_html'])
    assert not os.path.samefile(out/'7000/css/policy-train.jsonl', folder/'policy-train.jsonl')
    stamp=Path(exported['target_image']).stat().st_mtime_ns
    subprocess.run(command,check=True,capture_output=True)
    assert Path(exported['target_image']).stat().st_mtime_ns == stamp
    shutil.rmtree(source)  # Only this test's generated fixture, not user data.
    subprocess.run([sys.executable,str(SCRIPT),'--out',str(out),'--verify'],check=True,capture_output=True)
    after=batch_inputs(net,[exported],'cpu')
    assert torch.equal(original[1],after[1]) and torch.equal(original[2],after[2])
    assert original[3] == after[3]
    net.zero_grad()
    loss=net.loss(*after,[exported['replacement_edit']])[0]
    assert torch.equal(original_loss,loss)
    loss.backward()
    for name,p in net.named_parameters():
        if name in gradients:
            assert torch.equal(gradients[name],p.grad)
    assert Path(exported['target_html']).read_text() == '<html><body>Hello</body></html>'
    assert not any(p.is_symlink() for p in out.rglob('*'))


@pytest.mark.browser
def test_lazy_source_materializes_after_original_removed(tmp_path):
    from framediff.lazy_webui import CONTRACT, TrainViewCache
    module=exporter_module()
    old=tmp_path/'old';old.mkdir()
    html=old/'page.html';html.write_text('<html><body><p>Hello</p></body></html>')
    Image.new('RGB',(5,5),'red').save(old/'rick.jpg')
    row={'id':'p/clean','lazy_contract':CONTRACT,
         'lazy_source':{'id':'p','group':'p','split':'train','html':str(html),
                        'source_sha':hashlib.sha256(html.read_bytes()).hexdigest(),'viewport':[320,200]},
         'lazy_settings':{'seed':42,'max_nodes':32,'abstract_size':32,'min_elements':1,
                          'max_source_mae':1.,'resume':True,'clean_only':True,'lazy_train':False}}
    result=module.Exporter(tmp_path/'new').rewrite(row)
    assert result['lazy_source']['source_sha'] == row['lazy_source']['source_sha']
    assert Path(result['lazy_source']['html']).with_name('rick.jpg').is_file()
    assert os.path.samefile(result['lazy_source']['html'], html)
    shutil.rmtree(old)  # Exercise unlinking the last original path.
    cache=TrainViewCache()
    try:
        materialized=cache.get(result)
        assert Path(materialized['image']).is_file()
    finally:
        cache.close()


def test_missing_asset_and_absolute_html_resource_rejected(tmp_path):
    module=exporter_module();exporter=module.Exporter(tmp_path/'new')
    with pytest.raises(FileNotFoundError):
        exporter.rewrite({'target_image':str(tmp_path/'missing.png')})
    image=tmp_path/'x.png';Image.new('RGB',(1,1)).save(image)
    html=tmp_path/'x.html';html.write_text(f'<img src="{image}">')
    with pytest.raises(ValueError,match='Absolute local URL'):
        exporter.asset(html)


def test_hardlink_cross_device_never_falls_back_to_copy(tmp_path, monkeypatch):
    import errno
    module=exporter_module()
    source=tmp_path/'source.png';source.write_bytes(b'original')
    def fail(*args, **kwargs):
        raise OSError(errno.EXDEV, 'Cross-device link')
    monkeypatch.setattr(module.os, 'link', fail)
    with pytest.raises(ValueError, match='same filesystem'):
        module.Exporter(tmp_path/'out').asset(source)
    assert source.read_bytes() == b'original'
    assert not list((tmp_path/'out').rglob('source.png'))


def test_json_independent_and_old_copies_converted(tmp_path):
    module=exporter_module()
    old=tmp_path/'old';old.mkdir()
    image=old/'x.png';image.write_bytes(b'image')
    meta=old/'labels.json';meta.write_text(json.dumps({'image':str(image)}))
    original=meta.read_bytes()
    out=tmp_path/'out'
    target=out/'assets'/str(image).lstrip('/')
    target.parent.mkdir(parents=True);shutil.copyfile(image,target)
    assert not os.path.samefile(image,target)
    result=Path(module.Exporter(out).asset(meta))
    assert os.path.samefile(image,target)
    assert not os.path.samefile(meta,result)
    assert meta.read_bytes() == original
    assert json.loads(result.read_text())['image'] == str(target)
    shutil.rmtree(old)
    assert target.read_bytes() == b'image'
    assert result.is_file()

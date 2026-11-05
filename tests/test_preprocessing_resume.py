from types import SimpleNamespace
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from framediff.ir import read_json,write_json
from framediff.visual_data import _guard_render_run
from framediff.tree_normalize import _guard_normalization_run


def test_clean_render_growth_preserves_existing_input_contract(tmp_path):
    first=dict(id='a',html='/a',source_sha='a',split='train')
    config=dict(settings={'clean_only':True,'seed':42},sources=[first])
    _guard_render_run(tmp_path,config,True)
    grown={**config,'sources':[first,dict(first,id='b',source_sha='b')]}
    _guard_render_run(tmp_path,grown,True)
    assert read_json(tmp_path/'config.json')==grown
    for bad in ({**grown,'sources':[dict(first,source_sha='changed'),grown['sources'][1]]},
                {**grown,'sources':[first]},
                {**grown,'settings':{'clean_only':True,'seed':9}}):
        with pytest.raises(ValueError,match='configuration/input changed'):
            _guard_render_run(tmp_path,bad,True)


def test_legacy_normalized_growth_validates_kept_pages(tmp_path):
    html=tmp_path/'input.html';html.write_text('<html>original</html>')
    sha=hashlib.sha256(html.read_bytes()).hexdigest()
    row=dict(id='a/clean',current_html=str(html),group='a',split='train',
             source_sha=sha,viewport=[320,240])
    record=tmp_path/'normalized-pages/train/a/record.json'
    write_json(record,dict(source_sha=sha,row=row,page={'id':'a'}))
    old=dict(clean_targets_only=True,safe_sizes_sha='same',sources={'policy':'old'},seed=42)
    write_json(tmp_path/'config.json',old)
    grown={**old,'sources':{'policy':'new'},'source_records':{'a':'identity-a','b':'identity-b'}}
    _guard_normalization_run(tmp_path,grown,True,{'a':row,'b':dict(row,id='b/clean')})
    assert read_json(tmp_path/'config.json')['source_records']==grown['source_records']
    changed={**grown,'source_records':{'a':'changed','b':'identity-b'}}
    with pytest.raises(ValueError,match='configuration/input changed'):
        _guard_normalization_run(tmp_path,changed,True,{'a':row})
    write_json(tmp_path/'config.json',old)
    html.write_text('changed')
    with pytest.raises(ValueError,match='configuration/input changed'):
        _guard_normalization_run(tmp_path,grown,True,{'a':row})


def test_parallel_image_download_uses_one_atomic_cache(tmp_path,monkeypatch):
    import io
    from PIL import Image
    from framediff import image_asset_pilot as assets
    image=io.BytesIO();Image.new('RGB',(10,5)).save(image,format='PNG')
    payload=image.getvalue();calls=[]
    def urlopen(*args,**kwargs):
        calls.append(1);time.sleep(.05);return io.BytesIO(payload)
    monkeypatch.setattr(assets,'urlopen',urlopen)
    with ThreadPoolExecutor(4) as pool:
        outputs=list(pool.map(lambda _:assets.fetch_image('https://test/image.png',tmp_path,1),range(8)))
    assert len(calls)==1 and all(r==outputs[0] and r['ok'] for r in outputs)
    assert len(list(tmp_path.glob('*.json')))==1
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.browser
def test_parallel_color_resume_and_legacy_implementation(tmp_path):
    from framediff.webui_placeholders import prepare
    from framediff.ir import read_jsonl,write_jsonl
    rows=[]
    for i in range(3):
        html=tmp_path/f'{i}.html';html.write_text(f'<html><body><h1>Page {i}</h1></body></html>')
        rows.append(dict(id=f'page-{i}',page_id=f'page-{i}',html=str(html),viewport=[320,240],
                         group=f'g-{i}',split='train'))
    manifest=tmp_path/'input.jsonl';write_jsonl(manifest,rows)
    out=tmp_path/'out'
    args=SimpleNamespace(manifest=str(manifest),out=str(out),workers=2)
    prepare(args)
    before={p:p.stat().st_mtime_ns for p in out.rglob('*') if p.suffix in ('.png','.html','.jpg')}
    config=read_json(out/'config.json')
    config['implementation']='4c3f2bda926e498b1ad2b9262a212db7617daecd1bdfd7703a58e9d5c09f1cf9'
    write_json(out/'config.json',config)
    prepare(args)
    assert all(p.stat().st_mtime_ns==t for p,t in before.items())
    assert [r['id'] for r in read_jsonl(out/'manifest.jsonl')]==[r['id'] for r in rows]


@pytest.mark.browser
def test_recovered_render_grows_normalization_without_rebuilding_successes(tmp_path,monkeypatch):
    from framediff import visual_data,tree_normalize
    from framediff.ir import write_jsonl
    rows=[]
    for i,split in enumerate(('train','val','test')):
        html=tmp_path/f'{split}.html'
        html.write_text(f'<html><body><div style="width:{100+i}px;height:50px;background:red">{split}</div></body></html>')
        rows.append(dict(id=split,group=split,split=split,html=str(html),viewport=[400,200]))
    manifest=tmp_path/'input.jsonl';write_jsonl(manifest,rows)
    render=SimpleNamespace(manifest=str(manifest),out=str(tmp_path/'rendered'),resume=True,
        workers=1,trajectories=1,max_noise=1,seed=42,max_nodes=32,abstract_size=384,
        min_elements=1,max_source_mae=1.,clean_only=True)
    normalize=SimpleNamespace(rendered=render.out,out=str(tmp_path/'normalized'),workers=1,
        resume=True,seed=42,max_css_owners=512,observation_size=384,max_pixel_mae=.01,
        max_box_error=1.,limit_pages=0,latest_visual_labels=True,clean_targets_only=True,final_repartition=True)
    original=visual_data._render_page
    def missing_val(task):
        if task[-1]['id']=='val':return None,{'id':'val','error':'transient fixture'},False
        return original(task)
    monkeypatch.setattr(visual_data,'_render_page',missing_val)
    visual_data.build(render);tree_normalize.prepare(normalize)
    before={p:p.stat().st_mtime_ns for p in (tmp_path/'normalized').rglob('*') if p.suffix in ('.html','.png')}
    assert before
    monkeypatch.setattr(visual_data,'_render_page',original)
    visual_data.build(render);tree_normalize.prepare(normalize)
    assert all(p.stat().st_mtime_ns==stamp for p,stamp in before.items())
    assert read_json(tmp_path/'normalized/prepare-report.json')['val']['kept']==1
    assert read_json(tmp_path/'rendered/progress.json')['cache_hits']==2

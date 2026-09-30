from types import SimpleNamespace
from PIL import Image
import pytest
from framediff.benchmarks import combine_data,discover_design2code,prepare_design2code,run_design2code_vlm
from framediff.data import synthetic
from framediff.ir import read_json,read_jsonl,write_jsonl
from framediff.metrics import hungarian_box_metrics


def test_discover_design2code_pairs_only(tmp_path):
    (tmp_path/'2.png').write_bytes(b'x');(tmp_path/'2.html').write_text('<html/>')
    (tmp_path/'1.png').write_bytes(b'x')
    nested=tmp_path/'hard';nested.mkdir()
    (nested/'3.png').write_bytes(b'x');(nested/'3.html').write_text('<html/>')
    pairs=discover_design2code(tmp_path)
    assert [p[0] for p in pairs]==['2','hard__3']


def test_hungarian_geometry_ignores_generated_ids():
    target={'ref-a':[0,0,100,100],'ref-b':[200,0,100,100]}
    pred={'made-up-2':[200,0,100,100],'made-up-1':[0,0,100,100]}
    metrics=hungarian_box_metrics(pred,target,[400,200])
    assert metrics['box_iou']==1
    assert metrics['matched_pairs']==2


def test_combine_data_checks_split(tmp_path):
    write_jsonl(tmp_path/'a.jsonl',[{'id':'1','source':'a','split':'train'}])
    write_jsonl(tmp_path/'b.jsonl',[{'id':'1','source':'b','split':'train'}])
    args=SimpleNamespace(input=[tmp_path/'a.jsonl',tmp_path/'b.jsonl'],out=tmp_path/'all.jsonl',split='train')
    combine_data(args)
    assert len(list(read_jsonl(args.out)))==2


def test_design2code_batch_builds_independent_record(tmp_path,monkeypatch):
    image=tmp_path/'1.png';Image.new('RGB',(100,80)).save(image)
    manifest={'id':'1','group':'1','dataset':'design2code','screenshot':str(image),'html':'1.html',
              'reference_observations':[{'viewport':[100,80],'target':{'dom-0':[0,0,100,80]}}],
              'reference_root_ids':['dom-0']}
    write_jsonl(tmp_path/'manifest.jsonl',[manifest])
    tree=synthetic(1)
    def fake_call(args,task,image,current,runtime):
        value=tree if task=='generate-ir' else {'viewport':[100,80],'target':{n['id']:[0,0,10,10] for n in tree['nodes']}}
        return value,{'seconds':0},'prompt','raw'
    monkeypatch.setattr('framediff.benchmarks._vlm_call',fake_call)
    args=SimpleNamespace(manifest=tmp_path/'manifest.jsonl',out=tmp_path/'out',limit=0,backend='openai-compatible',
                         resume=False,fail_fast=True)
    run_design2code_vlm(args)
    record=list(read_jsonl(tmp_path/'out/test.jsonl'))[0]
    assert record['target_kind']=='predicted_frames'
    assert record['evaluation_matching']=='hungarian_geometry'
    assert record['reference_observations']==manifest['reference_observations']


@pytest.mark.browser
def test_prepare_design2code_smoke(tmp_path):
    Image.new('RGB',(320,200)).save(tmp_path/'1.png')
    buttons=''.join(f'<button style="width:40px;height:20px">{i}</button>' for i in range(30))
    (tmp_path/'1.html').write_text('<style>*{box-sizing:border-box}body{margin:0}</style><main style="width:300px;height:180px">'+buttons+'</main>')
    args=SimpleNamespace(root=tmp_path,out=tmp_path/'prepared',limit=0,max_nodes=10,max_fit_error=.3,
                         dataset='design2code',seed=42,fail_fast=True)
    prepare_design2code(args)
    report=read_json(tmp_path/'prepared/prepare-report.json')
    assert report['prepared']==1
    assert report['records'][0]['total_visible_nodes']>report['records'][0]['visible_nodes']==10
    assert len(list(read_jsonl(tmp_path/'prepared/manifest.jsonl')))==1

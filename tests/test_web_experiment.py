import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from framediff.html_bridge import HtmlBrowser, html_answer
from framediff.ir import DEFAULTS, read_json, read_jsonl
from framediff.web_experiment import evaluate_pages, prepare, repair_pages
from framediff.web_experiment import validated_generation, validate_target
from framediff import vlm


def test_validation_retries_report_missing_ids_and_bad_json():
    responses = ['{"target":', '{"viewport":[100,100],"target":{"extra":[0,0,2,2]}}',
                 '{"viewport":[100,100],"target":{"page":[0,0,100,100]}}']
    feedbacks = []
    def generate(attempt, feedback):
        feedbacks.append(feedback)
        return responses[attempt], 1
    result,_ = validated_generation(generate,lambda s:validate_target(s,[100,100],{'page'}),2)
    assert result['target']['page']==[0,0,100,100]
    assert 'missing IDs' in feedbacks[2] and 'extra' in feedbacks[2]
    assert 'previous response failed' in feedbacks[1]


def test_truncated_html_retry_and_runtime_error():
    seen = []
    def generate(attempt, feedback):
        seen.append(feedback)
        return ('<html>' if attempt==0 else '<html><body>ok</body></html>'),1
    result,_ = validated_generation(generate,html_answer,2)
    assert 'ok' in result and len(seen)==2
    def oom(attempt, feedback):
        raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError,match='CUDA'):
        validated_generation(oom,html_answer,2)


def test_extract_frames_prompt_uses_concrete_ids(tmp_path):
    current=tmp_path/'ir.json'
    current.write_text(json.dumps({'version':1,'nodes':[
        {'id':'**viewport**','parent':None,'role':'page','name':'page','props':dict(DEFAULTS)},
        {'id':'fd-0','parent':'**viewport**','role':'text','name':'title','props':dict(DEFAULTS)}]}))
    args=SimpleNamespace(task='extract-frames',current=str(current),frames=None)
    prompt=vlm.prompt_for(args)
    assert '"**viewport**": ["x", "y", "width", "height"]' in prompt
    assert '"fd-0": ["x", "y", "width", "height"]' in prompt
    assert 'Do not use the literal key "id"' in prompt


HTML = '''<html><head><style>body{margin:0;min-height:200px}main{position:absolute;left:10px;top:20px;
width:200px;height:140px;background:#abc}button{position:absolute;left:20px;top:30px;
width:80px;height:30px;color:red}</style></head><body><main><button>Hello</button></main></body></html>'''


def test_html_answer_rejects_truncation():
    assert html_answer('```html\n'+HTML+'\n```') == HTML
    with pytest.raises(ValueError): html_answer('<html><body>partial')


@pytest.mark.browser
def test_html_bridge_nested_transfer_and_noop():
    with HtmlBrowser() as b:
        dom = b.snapshot(HTML,[320,240])
        boxes = {n['id']:n['box'] for n in dom['nodes']}
        html, meta = b.patch(dom['html'],[320,240],boxes)
        assert meta['transfer_max_error_px'] < .01
        assert 'translate:' not in html
        desired = {k:list(v) for k,v in boxes.items()}
        desired['fd-1'] = [40,40,240,160]
        desired['fd-2'] = [90,90,100,40]
        patched,meta = b.patch(dom['html'],[320,240],desired)
        assert meta['transfer_max_error_px'] < .1
        final = b.snapshot(patched,[320,240])
        assert final['nodes'][2]['box'] == pytest.approx(desired['fd-2'],abs=.1)
        assert 'Hello' in patched and 'color:red' in patched


@pytest.mark.browser
def test_absolute_children_with_zero_height_body():
    from framediff.adapters import fit_observation
    with HtmlBrowser() as b:
        dom=b.snapshot(HTML.replace('min-height:200px','height:0'),[320,240])
        assert len(dom['nodes'])==2
        tree,_,_=fit_observation(dom)
        assert len(tree['nodes'])==3


@pytest.mark.browser
def test_real_html_pipeline_shared_initial_no_truth_leakage(tmp_path,monkeypatch):
    root = tmp_path/'dataset'; root.mkdir()
    Image.new('RGB',(320,240),'white').save(root/'a.png')
    (root/'a.html').write_text(HTML.replace('Hello','SECRET_REFERENCE_TEXT'))
    seen=[]
    def generate(args,prompt,images,runtime):
        assert 'SECRET_REFERENCE_TEXT' not in prompt
        seen.append(args.task)
        if args.task == 'generate-html':
            assert len(images)==1
            return HTML,{}
        if args.task == 'revise-html':
            assert len(images)==2
            assert Path(args.current).read_text()==HTML
            return HTML.replace('left:10px','left:30px'),{}
        from framediff.ir import execute
        tree=read_json(args.current)
        return json.dumps({'viewport':[320,240],'target':execute(tree,[320,240])}),{}
    monkeypatch.setattr('framediff.vlm.generate',generate)
    args=SimpleNamespace(root=str(root),manifest=None,out=str(tmp_path/'prepared'),rounds=1,
        max_nodes=16,max_pixels=1048576,max_new_tokens=8192,limit=0,seed=42,resume=False,
        backend='openai-compatible',model='fake',revision='main',endpoint='http://unused',
        api_key_env='TEST_KEY',four_bit=False,initial_mode='direct')
    rows=prepare(args)
    assert seen==['generate-html','revise-html','extract-frames']
    assert not rows[0]['errors']
    assert Path(rows[0]['methods']['initial']['html']).read_text()==HTML
    assert 'left:30px' in Path(rows[0]['methods']['self-revision-1']['html']).read_text()
    args.resume=True
    assert prepare(args)==rows
    assert len(seen)==3
    ck=tmp_path/'test.pt'; ck.write_bytes(b'mock')
    monkeypatch.setattr('framediff.model.load_model',lambda *a:(None,{'objective':'edits','training_groups':[]}))
    # Zero-edit repair must not change the HTML geometry just because IR fitting is lossy.
    monkeypatch.setattr('framediff.search.repair',lambda tree,*a,**kw:(tree,{'executions':1}))
    repair_args=SimpleNamespace(data=str(tmp_path/'prepared/prepared.jsonl'),out=str(tmp_path/'repaired'),
        checkpoint=str(ck),device='cpu',cpu_threads=1,methods='coordinate,model',steps=1,beam=1,topk=1,budget=1,
        seed=42,resume=False)
    repaired=repair_pages(repair_args)
    assert repaired[0]['methods']['model']['transfer_max_error_px']<.01
    eval_args=SimpleNamespace(data=str(tmp_path/'repaired/results.jsonl'),out=str(tmp_path/'eval'),
        max_nodes=128,official_repo=None,resume=False)
    metrics=evaluate_pages(eval_args)
    assert len(metrics)==4
    assert all(r['evaluation_error'] is None for r in metrics)
    by={r['method']:r for r in metrics}
    assert by['model']['pixel_mae']==pytest.approx(by['initial']['pixel_mae'],abs=1e-6)
    assert by['model']['vlm_calls']==2
    assert by['self-revision-1']['vlm_calls']==2
    summary=read_json(tmp_path/'eval/summary.json')
    assert all(r['n']==1 and r['dom_box_iou_n']==1 for r in summary)
    assert all(r['n']==1 for r in read_json(tmp_path/'eval/summary-common-success.json'))
    # Exercise the actual neural proposal/search path as well, using a tiny random checkpoint.
    monkeypatch.undo()
    import torch
    from dataclasses import asdict
    from framediff.model import EditDenoiser,ModelConfig
    cfg=ModelConfig(hidden=16,layers=1,heads=2,max_nodes=16)
    network=EditDenoiser(cfg)
    torch.save({'config':asdict(cfg),'model':network.state_dict(),'objective':'edits','training_groups':[]},ck)
    repair_args.out=str(tmp_path/'neural-repaired')
    neural=repair_pages(repair_args)
    assert not neural[0]['methods']['model']['failed']
    assert neural[0]['methods']['model']['proxy_executions']>=1
    repair_args.out=str(tmp_path/'neural-feedback')
    repair_args.methods='coordinate-feedback,model-feedback'
    feedback=repair_pages(repair_args)
    for method in ('coordinate-feedback','model-feedback'):
        assert not feedback[0]['methods'][method]['failed']
        assert feedback[0]['methods'][method]['proxy_executions']==0
        assert feedback[0]['methods'][method]['browser_executions']>=1
        assert feedback[0]['methods'][method]['feedback_mode']=='frames'


@pytest.mark.browser
def test_failed_html_generation_keeps_page_for_all_methods(tmp_path,monkeypatch):
    Image.new('RGB',(100,100)).save(tmp_path/'a.png')
    (tmp_path/'a.html').write_text(HTML)
    monkeypatch.setattr('framediff.vlm.generate',lambda *a:('truncated HTML',{}))
    args=SimpleNamespace(root=str(tmp_path),manifest=None,out=str(tmp_path/'out'),rounds=2,
        max_nodes=16,max_pixels=1024,max_new_tokens=10,limit=0,seed=42,resume=False,
        backend='openai-compatible',model='fake',revision='main',endpoint='http://unused',
        api_key_env='TEST_KEY',four_bit=False,initial_mode='direct')
    rows=prepare(args)
    assert len(rows)==1 and rows[0]['methods']['initial']['failed']
    assert rows[0]['methods']['self-revision-2']['failed']
    assert 'frames' in rows[0]['errors']
    assert len(list(read_jsonl(tmp_path/'out/prepared.jsonl')))==1
    assert rows[0]['methods']['initial']['vlm_calls']==3
    args.resume=True
    assert prepare(args)==rows
    args.retry_failed=True
    def recovered(c,*unused):
        if c.task.endswith('html'): return HTML,{}
        from framediff.ir import execute
        return json.dumps({'viewport':[100,100],'target':execute(read_json(c.current),[100,100])}),{}
    monkeypatch.setattr('framediff.vlm.generate',recovered)
    recovered_rows=prepare(args)
    assert not recovered_rows[0]['errors']
    assert list((tmp_path/'out/pages').glob('*/previous-attempts/*/record.json'))


@pytest.mark.browser
@pytest.mark.skipif(not os.environ.get('FRAMEDIFF_OFFICIAL_REPO'),reason='Optional upstream metrics + CLIP weights')
def test_official_real_scoring_identity_and_shift(tmp_path):
    from framediff.official_metrics import OfficialMetrics
    scorer=OfficialMetrics(os.environ['FRAMEDIFF_OFFICIAL_REPO'])
    with HtmlBrowser() as browser:
        identity=scorer.score(browser,HTML,HTML,[320,240],tmp_path/'identity')
        shifted=scorer.score(browser,HTML.replace('left:10px','left:70px'),HTML,[320,240],tmp_path/'shifted')
    for key in ('official_block','official_text','official_position','official_color','official_clip'):
        assert identity[key]==pytest.approx(1,abs=1e-4)
    assert shifted['official_position']<identity['official_position']

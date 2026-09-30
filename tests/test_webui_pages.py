import gzip
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from framediff.ir import execute, read_json
from framediff.model import EditDenoiser, ModelConfig
from framediff.web_experiment import prepare, repair_pages, evaluate_pages
from framediff.webui_pages import discover_webui


def raw_page(root,prefix='default_320-200',metadata=True):
    root.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(320,200),'white').save(root/f'{prefix}-screenshot.webp')
    # Full-page images must never be accidentally used with viewport boxes.
    Image.new('RGB',(320,800),'white').save(root/f'{prefix}-screenshot-full.webp')
    (root/f'{prefix}-url.txt').write_text('https://example.org/page')
    if metadata:
        ax={'nodes':[
            {'nodeId':'root','backendDOMNodeId':1,'role':{'value':'RootWebArea'}},
            {'nodeId':'ref','backendDOMNodeId':2,'parentId':'root','role':{'value':'button'},'name':{'value':'SECRET_REFERENCE'}},
            {'nodeId':'offscreen','backendDOMNodeId':3,'parentId':'root'}]}
        bb={'1':{'x':0,'y':0,'width':320,'height':800},
            '2':{'x':20,'y':10,'width':100,'height':40},
            '3':{'x':0,'y':500,'width':100,'height':50}}
        for suffix,value in [('axtree',ax),('bb',bb)]:
            with gzip.open(root/f'{prefix}-{suffix}.json.gz','wt') as f: json.dump(value,f)


def test_webui_raw_discovery_view_and_missing_metadata(tmp_path):
    raw_page(tmp_path/'one')
    raw_page(tmp_path/'two',metadata=False)
    records=discover_webui(tmp_path,'default_320-200')
    assert len(records)==2
    assert records[0]['html'] is None
    assert records[0]['group']=='example.org'
    assert records[0]['reference_boxes']=={'ref':[20,10,100,40]}
    assert records[1]['reference_box_error']
    assert len(discover_webui(tmp_path,'all',limit=1))==1
    with pytest.raises(ValueError,match='No WebUI'): discover_webui(tmp_path,'default_1280-720')


@pytest.mark.browser
def test_webui_raw_to_real_html_feedback_and_recorded_evaluation(tmp_path,monkeypatch):
    root=tmp_path/'raw';raw_page(root/'sample')
    raw_page(root/'missing-metadata',metadata=False)
    html='<html><body style="margin:0;min-height:200px"><button style="position:absolute;left:20px;top:10px;width:100px;height:40px">Hello</button></body></html>'
    def generate(args,prompt,images,runtime):
        assert 'SECRET_REFERENCE' not in prompt
        if args.task=='extract-frames':
            tree=read_json(args.current)
            return json.dumps({'viewport':[320,200],'target':execute(tree,[320,200])}),{}
        assert len(images)==(2 if args.task=='revise-html' else 1)
        return html,{}
    monkeypatch.setattr('framediff.vlm.generate',generate)
    args=SimpleNamespace(root=str(root),manifest=None,out=str(tmp_path/'prepared'),dataset='webui',
        webui_view='default_320-200',rounds=1,max_nodes=16,max_pixels=65536,max_new_tokens=4096,
        limit=0,seed=42,resume=False,backend='openai-compatible',model='mock',revision='main',
        endpoint='unused',api_key_env='TEST',four_bit=False,initial_mode='direct')
    prepared=prepare(args)
    assert len(prepared)==2 and all(not r['errors'] for r in prepared)
    args.resume=True
    assert prepare(args)==prepared
    cfg=ModelConfig(hidden=16,layers=1,heads=2,max_nodes=16)
    model=EditDenoiser(cfg); checkpoint=tmp_path/'model.pt'
    torch.save({'config':asdict(cfg),'model':model.state_dict(),'objective':'edits','training_groups':[]},checkpoint)
    repair_args=SimpleNamespace(data=str(tmp_path/'prepared/prepared.jsonl'),checkpoint=str(checkpoint),
        out=str(tmp_path/'repair'),methods='model-feedback',device='cpu',cpu_threads=1,
        steps=1,beam=1,topk=1,budget=1,seed=42,resume=False,feedback_render='frames')
    repaired=repair_pages(repair_args)
    assert all(not r['methods']['model-feedback']['failed'] for r in repaired)
    eval_args=SimpleNamespace(data=str(tmp_path/'repair/results.jsonl'),out=str(tmp_path/'eval'),
        max_nodes=128,official_repo=None,resume=False)
    rows=evaluate_pages(eval_args)
    assert len(rows)==6 and all(r['evaluation_error'] is None for r in rows)
    assert all('pixel_mae' in r for r in rows)
    valid=[r for r in rows if r['id'].startswith('sample/')]
    assert all('webui_box_iou' in r for r in valid)
    assert all(r['reference_kind']=='webui_recorded_boxes' for r in rows)
    assert all('dom_box_iou' not in r for r in rows)
    summary=read_json(tmp_path/'eval/summary.json')
    assert all(s['n']==2 and s['webui_box_iou_n']==1 and s['reference_failures']==1 for s in summary)
    eval_args.official_repo='unused'
    with pytest.raises(ValueError,match='Omit --official-repo'): evaluate_pages(eval_args)

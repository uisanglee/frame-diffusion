from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from framediff import css_owners
from framediff.ir import read_jsonl, write_jsonl


@pytest.mark.browser
def test_frozen_webui_test_rollout_exports_html_png_and_metrics(tmp_path,monkeypatch):
    from framediff.cli import main
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate, elements
    import framediff.fixed_test as fixed

    html='''<!doctype html><html><head><style>body{margin:0}.card{width:200px;height:80px;background:#abc}</style></head>
    <body><div class="card">card</div></body></html>'''
    target=tmp_path/'target.png';current_png=tmp_path/'current.png';target_html=tmp_path/'target.html';current_html=tmp_path/'current.html'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[320,240],target,16);tree,target_boxes,_=dom_tree(dom);annotate(browser,tree)
        target_html.write_text(dom['html']);parsed=css_owners.read(browser,tree)
        owner=next(i for i,o in enumerate(parsed['owners']) if o.get('selector')=='.card')
        css_owners.execute(browser,parsed['owners'],[owner,'width','120px',''])
        current_html.write_text(browser.page.content());browser.page.screenshot(path=str(current_png))
        current_boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
        current_boxes[tree['nodes'][0]['id']]=[0,0,320,240]
        current_state=css_owners.read(browser,tree)['state']
    row=dict(id='page/online-0',group='held-out',split='test',source_sha='held-out',viewport=[320,240],
        current=tree,current_boxes=current_boxes,current_html=str(current_html),current_image=str(current_png),
        target_html=str(target_html),target_image=str(target),target_elements=elements(tree,target_boxes,[320,240]),
        css_owners=parsed['owners'],declaration_state=current_state,target_declaration_state=parsed['state'])
    data=tmp_path/'policy-test.jsonl';pages=tmp_path/'pages-test.jsonl'
    write_jsonl(data,[row]);write_jsonl(pages,[dict(id='page',group='held-out',split='test',source_sha='held-out',
        viewport=[320,240],evaluation_target_boxes=target_boxes)])
    policy_path=tmp_path/'policy.pt';detector_path=tmp_path/'detector.pt'
    policy_path.write_bytes(b'policy');detector_path.write_bytes(b'detector')
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.cfg=SimpleNamespace(mode='abstract')
    monkeypatch.setattr(fixed,'load_policy',lambda *a:(Policy(),{}))
    monkeypatch.setattr(fixed,'load_detector',lambda *a:(object(),{}))
    monkeypatch.setattr(fixed,'rollout',lambda browser,source,*a,**kw:(source,dict(actions=0,seconds=.1,
        browser_executions=0,browser_screenshots=0,history=[]),None))
    out=tmp_path/'out'
    main(['visual-test-rollout','--data',str(data),'--pages',str(pages),'--out',str(out),
          '--abstract-checkpoint',str(policy_path),'--detector-checkpoint',str(detector_path),
          '--device','cpu','--steps','1'])
    result=list(read_jsonl(out/'results.jsonl'))[0]
    assert Path(result['html']).is_file() and Path(result['png']).is_file()
    assert result['initial_box_iou']<1 and result['final_box_iou']==result['initial_box_iou']
    assert (out/'report.md').is_file() and (out/'summary.json').is_file()

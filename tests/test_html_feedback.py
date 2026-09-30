import copy

import pytest

from framediff.adapters import fit_observation
from framediff.html_bridge import HtmlBrowser
from framediff.html_feedback import action_unit, repair_html, refresh_geometry


FLOW_HTML = '''<html><head><style>
body{margin:0;font:16px/20px monospace}section{width:480px}
#card{width:300px;background:#ddd}#next{height:24px;background:#acf}
</style></head><body><section><div id="card">one two three four five six seven eight nine ten
eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen</div><div id="next">Next item</div>
</section></body></html>'''


def fixture(browser):
    viewport=[500,300]
    dom=browser.snapshot(FLOW_HTML,viewport)
    tree,_,_=fit_observation(dom)
    boxes={n['id']:n['box'] for n in dom['nodes']}
    boxes[tree['nodes'][0]['id']]=[0,0,*viewport]
    tree=refresh_geometry(tree,boxes)
    card=next(i for i,n in enumerate(tree['nodes']) if n['id']=='fd-2')
    old=tree['nodes'][card]['props']['width']; value=round(old/2)
    delta=(value-old)*action_unit(tree,boxes,card,'width')
    browser.edit_property(dom['html'],viewport,'fd-2','width',delta)
    target=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
    target[tree['nodes'][0]['id']]=[0,0,*viewport]
    return dom['html'],tree,boxes,[{'viewport':viewport,'target':target}],(card,'width',value)


@pytest.mark.browser
def test_reflow_updates_next_model_input_and_preserves_auto_height(tmp_path,monkeypatch):
    with HtmlBrowser() as browser:
        html,tree,initial,observations,edit=fixture(browser)
        # Keep a second unresolved error so another proposal sees the reflowed DOM.
        observations[0]['target']['fd-3'][1]+=8
        seen=[]
        def proposer(model,current,obs,boxes,k,method):
            seen.append(copy.deepcopy(boxes))
            return [edit] if len(seen)==1 else []
        monkeypatch.setattr('framediff.html_feedback.propose_edits',proposer)
        result,_,stats=repair_html(browser,html,tree,observations,steps=3,beam=1,
            render_mode='frames',trace_dir=tmp_path)
        assert len(seen)==2
        assert seen[1]['fd-2'][2]<seen[0]['fd-2'][2]  # width shrank
        assert seen[1]['fd-2'][3]>seen[0]['fd-2'][3]  # auto height increased from wrapping
        assert seen[1]['fd-3'][1]>seen[0]['fd-3'][1]  # next element moved without direct editing
        assert 'fd-3' in stats['edits'][0]['changed_nodes']
        assert stats['proxy_executions']==0 and stats['browser_executions']==2
        assert (tmp_path/'step-001.png').exists()
        browser.load(result,[500,300])
        assert browser.page.locator('#card').evaluate('e=>e.style.height')==''
        assert browser.page.locator('#next').evaluate('e=>e.getAttribute("style")') is None
        assert browser.page.locator('#card').inner_text().startswith('one two three')
        assert initial['fd-3'][1]==seen[0]['fd-3'][1]


@pytest.mark.browser
def test_frame_and_raster_modes_use_identical_actual_geometry(monkeypatch):
    with HtmlBrowser() as browser:
        html,tree,_,obs,edit=fixture(browser)
        monkeypatch.setattr('framediff.html_feedback.propose_edits',lambda *a:[edit])
        results=[]
        for mode in ('boxes','frames','raster'):
            _,_,stats=repair_html(browser,html,tree,obs,steps=1,beam=1,render_mode=mode)
            results.append(stats['history'][-1]['boxes'])
            assert stats['browser_executions']==2
            assert stats['feedback_images']==(0 if mode=='boxes' else 2)
            assert stats['browser_screenshots']==(2 if mode=='raster' else 0)
        assert results[0]==results[1]==results[2]


@pytest.mark.browser
def test_feedback_identity_and_failed_candidate_are_not_silently_lost(monkeypatch):
    with HtmlBrowser() as browser:
        html,tree,_,obs,edit=fixture(browser)
        monkeypatch.setattr('framediff.html_feedback.propose_edits',lambda *a:[edit])
        def fail(*a): raise ValueError('test render failure')
        monkeypatch.setattr(browser,'edit_property',fail)
        result,_,stats=repair_html(browser,html,tree,obs,steps=2,beam=1,render_mode='boxes')
        assert len(stats['candidate_failures'])==1
        assert stats['edits']==[]
        assert 'Next item' in result
        with pytest.raises(ValueError,match='identities'):
            browser.tagged_boxes(['not-present'])


@pytest.mark.browser
def test_policy_tensor_contains_reflowed_boxes_not_proxy_geometry():
    import torch
    from types import SimpleNamespace
    from framediff.ir import FIELDS
    class RecordingPolicy(torch.nn.Module):
        def __init__(self,edit):
            super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(1))
            self.cfg=SimpleNamespace(max_nodes=16); self.edit=edit; self.seen=[]
        def forward(self,batch):
            self.seen.append(batch['geometry'].clone())
            legal=batch['legal']; logits=torch.full(legal.shape,float('-inf'))
            i,field,value=self.edit
            # First propose width; subsequent invocation explicitly stops.
            if len(self.seen)==1: logits[0,i,FIELDS.index(field),value]=10
            return {'logits':logits,'stop':torch.tensor([0.])}
    with HtmlBrowser() as browser:
        html,tree,_,obs,edit=fixture(browser)
        obs[0]['target']['fd-3'][1]+=8
        model=RecordingPolicy(edit)
        _,_,stats=repair_html(browser,html,tree,obs,model=model,steps=3,topk=1,beam=1,render_mode='boxes')
        assert len(model.seen)==2
        index=next(i for i,n in enumerate(tree['nodes']) if n['id']=='fd-3')
        actual_y=stats['history'][1]['boxes']['fd-3'][1]/300
        assert model.seen[1][0,0,index,1].item()==pytest.approx(actual_y)
        assert model.seen[1][0,0,index,1]>model.seen[0][0,0,index,1]

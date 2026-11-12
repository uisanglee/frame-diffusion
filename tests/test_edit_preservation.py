import numpy as np
import pytest
from framediff.edit_preservation import matching_regions, violated_regions, editable_owners


def test_regions_use_all_pixels_and_shared_rules():
    a=np.zeros((30,40,3),dtype=np.uint8);b=a.copy();b[10,10]=255
    boxes={'a':[5,5,10,10],'b':[25,5,10,10]}
    regions=matching_regions(a,b,boxes,padding=0)
    assert set(regions)=={'b'}
    owners=[{}, {'matches':['a']},{'matches':['b']},{'matches':['a','b']}]
    assert editable_owners(owners,regions)==[1,3]
    assert editable_owners(owners,regions,{1})==[3]
    c=b.copy();c[6,26]=255
    assert violated_regions(a,c,regions)==['b']


@pytest.mark.browser
@pytest.mark.parametrize('scenario',['clean','repair','reflow','noop'])
def test_rollout_protection(tmp_path,scenario):
    from framediff import css_owners
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.tree_policy import TreePolicy,TreeConfig
    from framediff.visual import annotate
    from framediff.visual_experiment import rollout
    html='''<style>body{margin:0}.first{width:120px;height:30px;background:red;margin-bottom:0px}
    .second{width:100px;height:30px;background:blue;margin-top:20px}</style><div class="first"></div><div class="second"></div>'''
    target=tmp_path/'target.png'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[320,180],max_nodes=16)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        browser.page.screenshot(path=str(target),animations='disabled')
        clean=browser.page.content();parsed=css_owners.read(browser,tree)
        first=next(i for i,o in enumerate(parsed['owners']) if o.get('selector')=='.first')
        second=next(i for i,o in enumerate(parsed['owners']) if o.get('selector')=='.second')
        net=TreePolicy(TreeConfig(mode='screenshot',size=32,hidden=16,layers=1,heads=2,
            token_grid=2,max_nodes=16,stylesheets=True,action_contract=css_owners.CONTRACT,
            existing_values_only=True),False).eval()
        if scenario!='clean':css_owners.execute(browser,parsed['owners'],[first,'width','80px',''])
        initial=browser.page.content();calls=[]
        def predict(*args,allowed_owners=None,**kwargs):
            calls.append(allowed_owners)
            if scenario=='clean':pytest.fail('Clean image should have no unprotected editable owner')
            assert first in allowed_owners
            assert second not in allowed_owners
            return [first,'margin-bottom','40px',''] if scenario=='reflow' else [first,'width','80px' if scenario=='noop' else '120px','']
        net.predict=predict
        final,stats,_=rollout(browser,initial,tree,[320,180],str(target),net,steps=1,
            pixel_goal_fraction=-1,preserve_matched_regions=True)
        assert not stats['failed'],stats.get('error')
        if scenario=='clean':
            assert stats['actions']==0 and not calls
            assert stats['stop_reason']=='no_unprotected_editable_owners'
        elif scenario=='repair':assert stats['actions']==1
        elif scenario=='noop':assert stats['noop_skips']==1 and stats['actions']==0 and final==initial
        else:
            assert stats['preservation_rollbacks']==1
            assert stats['actions']==0 and final==initial

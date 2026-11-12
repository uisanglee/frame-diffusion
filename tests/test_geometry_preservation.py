import copy
import pytest
from framediff.geometry_preservation import match_geometry, violated_boxes


def item(id,box,label=1):return {'owner_id':id,'box':box,'label':label}


def test_geometry_ignores_style_and_target_ids():
    target=[item('different',[10,20,110,70])]
    current=[dict(item('a',[11,20,110,70]),color='red',font='other')]
    r=match_geometry(target,current,{'a':[11,20,99,50]},[1280,720])
    assert r['protected_ids']==['a'] and r['goal']
    assert r['tolerance_xywh']==[2.56,2.,2.56,2.]


def test_ambiguous_missing_wrong_class_not_correct():
    current=[item('a',[10,10,100,100])];boxes={'a':[10,10,90,90]}
    for target in ([],[item('x',[10,10,100,100],2)], [item('x',[10,10,100,100])]*2):
        result=match_geometry(target,current,boxes,[200,200])
        assert not result['protected_ids'] and not result['goal']
    result=match_geometry([item('x',[10,10,100,100]),item('y',[120,120,180,180])],current,boxes,[200,200])
    assert result['protected_ids']==['a'] and not result['goal']


def test_owner_all_fragments_and_size_error():
    current=[item('a',[0,0,100,20]),item('a',[0,30,100,50])]
    target=[item('x',[0,0,100,20]),item('x',[0,30,110,50])]
    r=match_geometry(target,current,{'a':[0,0,100,50]},[200,200])
    assert len(r['matches'])==2 and not r['protected_ids']
    assert violated_boxes({'a':[0,0,100,50]},{'a':[0,1,100,50]})==['a']
    assert violated_boxes({'a':[0,0,100,50]}, {})==['a']


@pytest.mark.browser
@pytest.mark.parametrize('scenario',['style','repair','reflow'])
def test_geometry_rollout_without_rgb(tmp_path,scenario):
    from framediff import css_owners
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.tree_policy import TreePolicy,TreeConfig
    from framediff.visual import annotate,elements,VISUAL_GEOMETRY_CONTRACT
    from framediff.visual_experiment import rollout
    html='''<style>body{margin:0}.first{width:120px;height:30px;background:red;margin-bottom:0px}
    .second{width:100px;height:30px;background:blue;margin-top:20px}</style><div class="first"></div><div class="second"></div>'''
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[320,180],max_nodes=16)
        tree,boxes,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        target=copy.deepcopy(elements(tree,boxes,[320,180]))
        parsed=css_owners.read(browser,tree)
        first=next(i for i,o in enumerate(parsed['owners']) if o.get('selector')=='.first')
        second=next(i for i,o in enumerate(parsed['owners']) if o.get('selector')=='.second')
        if scenario=='style':
            browser.page.evaluate("document.querySelector('.first').style.background='green'")
        else:css_owners.execute(browser,parsed['owners'],[first,'width','80px',''])
        initial=browser.page.content()
        net=TreePolicy(TreeConfig(mode='abstract',size=32,hidden=16,layers=1,heads=2,
            token_grid=2,max_nodes=16,stylesheets=True,action_contract=css_owners.CONTRACT,
            visual_geometry_contract=VISUAL_GEOMETRY_CONTRACT,existing_values_only=True),False).eval()
        calls=[]
        def predict(*args,allowed_owners=None,**kwargs):
            calls.append(allowed_owners)
            assert scenario!='style'
            assert first in allowed_owners and second not in allowed_owners
            return [first,'width','120px',''] if scenario=='repair' else [first,'margin-bottom','40px','']
        net.predict=predict
        # No target RGB file exists: geometry decisions must never read it.
        final,stats,_=rollout(browser,initial,tree,[320,180],str(tmp_path/'absent.png'),net,
            oracle_elements=target,steps=1,preserve_matched_geometry=True)
        assert not stats['failed'],stats.get('error')
        assert stats['pixel_goal_checks']==0 and stats['pixel_goal_fraction']==-1
        if scenario=='style':
            assert stats['geometry_goal_reached'] and not calls and stats['actions']==0
            assert final==initial
        elif scenario=='repair':
            assert stats['actions']==1 and stats['geometry_goal_reached']
        else:
            assert stats['preservation_rollbacks']==1 and stats['actions']==0
            assert not stats['geometry_goal_reached'] and final==initial

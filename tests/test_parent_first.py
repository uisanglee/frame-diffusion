import copy

import pytest
import torch

from framediff import css_owners
from framediff.html_bridge import HtmlBrowser
from framediff.parent_first import owner_order,improving_teacher
from framediff.plans import dom_tree
from framediff.tree_online import abstraction
from framediff.tree_policy import TreeConfig,TreePolicy
from framediff.visual import annotate
from framediff.visual_experiment import rollout


def test_order_uses_hierarchy_not_dom_array_order():
    tree={'nodes':[{'id':'root'},{'id':'child','parent':'parent'},
                   {'id':'parent','parent':'root'}]}
    assert owner_order(tree)==[2,1]
    owners=[{'kind':'root'},{'kind':'inline','id':'child'},
            {'kind':'inline','id':'parent'},{'kind':'rule','matches':['child','parent']}]
    assert owner_order(tree,owners)==[2,3,1]


@pytest.mark.browser
@pytest.mark.parametrize('parent_wrong',[False,True])
@pytest.mark.parametrize('mode,representation', [('screenshot','occupancy-boundary'),
    *[('abstract',r) for r in ('occupancy-boundary','abstract-rgb','occupancy','boundary')]])
def test_rollout_skips_correct_parent_and_repairs_parent_before_child(tmp_path,parent_wrong,mode,representation):
    torch.set_num_threads(1)
    html=f'''<body style="margin:0"><div id="parent" style="width:{180 if parent_wrong else 200}px;
    height:100px;background:red"><div id="child" style="width:40px;height:30px;background:blue"></div></div></body>'''
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[300,180],max_nodes=16);tree,_,_=dom_tree(dom);annotate(browser,tree)
        initial=browser.page.content()
        parsed=css_owners.read(browser,tree)
        p=next(i for i,o in enumerate(parsed['owners']) if o.get('id')==browser.page.locator('#parent').get_attribute('data-fd-id'))
        c=next(i for i,o in enumerate(parsed['owners']) if o.get('id')==browser.page.locator('#child').get_attribute('data-fd-id'))
        css_owners.execute(browser,parsed['owners'],[p,'width','200px',''])
        css_owners.execute(browser,parsed['owners'],[c,'width','80px',''])
        target=tmp_path/'target.png';browser.page.screenshot(path=str(target))
        clean_mask,_,clean_items=abstraction(browser,tree,[300,180],64)
        browser.load(initial,[300,180])
        edit,gain=improving_teacher(browser,tree,[300,180],64,parsed['state'],
            [[c,'width','80px',''],[p,'width','200px','']],clean_mask,parsed['owners'])
        assert edit[0]==(p if parent_wrong else c) and gain>0
        net=TreePolicy(TreeConfig(mode=mode,size=64,hidden=16,layers=1,heads=2,
            token_grid=2,max_nodes=16,stylesheets=True,existing_values_only=True,parent_first=True,
            action_contract=css_owners.CONTRACT,abstraction_representation=representation),False)
        net.eval()
        def predict(*args,allowed_owners=None,**kwargs):
            owner=allowed_owners[0]
            return [owner,'width','200px' if owner==p else '80px','']
        net.predict=predict
        final,stats,_=rollout(browser,initial,copy.deepcopy(tree),[300,180],str(target),net,steps=10,
                            oracle_elements=clean_items if mode=='abstract' else None)
        assert not stats['failed'],stats
        accepted=[h['action'][0] for h in stats['history'] if h.get('accepted')]
        assert accepted==([p,c] if parent_wrong else [c]),stats
        browser.load(final,[300,180])
        assert browser.page.locator('#parent > #child').count()==1
        assert browser.page.locator('#child').bounding_box()['width']==80
        assert stats['parent_first_probes']>=len(accepted)


@pytest.mark.browser
def test_validation_relabel_preserves_source_and_uses_improving_parent(tmp_path):
    from framediff.parent_first import relabel_validation
    with HtmlBrowser() as browser:
        dom=browser.snapshot('<div id="parent" style="width:200px;height:100px;background:red"><div style="width:80px;height:30px;background:blue"></div></div>',[300,180],max_nodes=16)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        clean=tmp_path/'clean.html';clean.write_text(browser.page.content())
        parsed=css_owners.read(browser,tree)
        pid=browser.page.locator('#parent').get_attribute('data-fd-id')
        owner=next(i for i,o in enumerate(parsed['owners']) if o.get('id')==pid)
        css_owners.execute(browser,parsed['owners'],[owner,'width','160px',''])
        current=tmp_path/'current.html';current.write_text(browser.page.content())
    row={'id':'val/a','viewport':[300,180],'current':tree,'css_owners':parsed['owners'],
         'target_html':str(clean),'current_html':str(current),'replacement_edit':[2,'width','80px','']}
    before=copy.deepcopy(row)
    relabeled,excluded=relabel_validation([row],64)
    assert not excluded and row==before
    assert relabeled[0]['replacement_edit']==[owner,'width','200px','']

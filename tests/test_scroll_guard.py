import pytest
from framediff.html_bridge import HtmlBrowser
from framediff.scroll_guard import snapshot,compare
from framediff.explicit_html import convert


def test_horizontal_guard_only_rejects_new_offscreen_state():
    import copy
    before=dict(viewport_height=720,viewport_width=1280,elements=[
        dict(id='85',tag='DIV',parent=82,box=[-4912,1520,1200,228],visible_fraction=1)])
    assert compare(before,copy.deepcopy(before))['accepted']
    before['elements'][0]['box'][0]=1400
    assert compare(before,copy.deepcopy(before))['accepted']
    for x in (0, -100):  # Fully and partially horizontally visible before.
        before['elements'][0]['box'][0]=x
        after=copy.deepcopy(before)
        after['elements'][0]['box'][0]=-4912
        assert compare(before,after)['failures'][0]['reason']=='outside_horizontal_view'
        after['elements'][0]['box'][0]=1400
        assert not compare(before,after)['accepted']


@pytest.mark.browser
def test_inline_field_probes_keep_horizontal_margin_not_sizes(tmp_path):
    with HtmlBrowser() as b:
        qa=convert(b,'<a data-fd-id="a" style="margin-left:0px">Link</a>',[400,200],tmp_path,mode='sizes')
        assert qa['accepted'],qa
        fields=b.page.locator('a').get_attribute('data-tuide-editable-fields').split()
        assert 'margin-left' in fields and 'width' not in fields and 'height' not in fields
        assert 'margin-top' not in fields and 'margin-bottom' not in fields
        assert b.page.locator('a').evaluate('e=>getComputedStyle(e).display')=='inline'


@pytest.mark.browser
def test_table_fields_probed_without_display_change(tmp_path):
    with HtmlBrowser() as b:
        qa=convert(b,'<table data-fd-id="t" style="width:200px;margin-left:0px"><tr><td>A</td></tr></table>',
                   [400,200],tmp_path,mode='sizes')
        assert qa['accepted'],qa
        assert 'margin-left' in b.page.locator('table').get_attribute('data-tuide-editable-fields').split()
        assert b.page.locator('table').evaluate('e=>getComputedStyle(e).display')=='table'


@pytest.mark.browser
def test_scroll_guard_accepts_translation_rejects_disappearance_clip_overlap():
    html='''<style>body{margin:0}.spacer{height:400px}.section{height:100px}
    p{margin:0;height:40px}</style><div class="spacer"></div>
    <section class="section"><p>A</p><p>B</p></section>'''
    with HtmlBrowser() as b:
        b.load(html,[400,200]);before=snapshot(b)
        b.page.locator('.spacer').evaluate("e=>e.style.height='500px'")
        assert compare(before,snapshot(b))['accepted']
        b.page.locator('.section').evaluate("e=>{e.style.height='10px';e.style.overflow='hidden'}")
        assert not compare(before,snapshot(b))['accepted']
        b.load(html,[400,200])
        b.page.locator('p').first.evaluate("e=>e.style.display='none'")
        assert any(r['reason']=='disappeared' for r in compare(before,snapshot(b))['failures'])
        b.load(html,[400,200])
        b.page.locator('p').last.evaluate("e=>e.style.marginTop='-30px'")
        assert any(r['reason']=='new_sibling_overlap' for r in compare(before,snapshot(b))['failures'])


@pytest.mark.browser
def test_rollout_returns_initial_when_shared_rule_hides_below_fold(tmp_path):
    import torch
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff import css_owners
    from framediff.tree_policy import TreeConfig,TreePolicy
    from framediff.visual_experiment import rollout
    torch.set_num_threads(2)
    html='''<style>.card{width:100px;height:40px;background:red;overflow:hidden}
    .spacer{height:400px}</style><div class="card">A</div><div class="spacer"></div><div class="card">B</div>'''
    image=tmp_path/'target.png'
    with HtmlBrowser() as b:
        dom=b.snapshot(html,[400,200],image,16);tree,_,_=dom_tree(dom);annotate(b,tree)
        state=css_owners.read(b,tree)
        owner=next(i for i,o in enumerate(state['owners']) if o.get('selector')=='.card')
        policy=TreePolicy(TreeConfig(mode='screenshot',size=32,hidden=16,layers=1,heads=2,
            token_grid=2,max_nodes=32,stylesheets=True,action_contract=css_owners.CONTRACT),False)
        policy.eval();policy.predict=lambda *a,**k:[owner,'height','0px','']
        result,stats,_=rollout(b,dom['html'],tree,[400,200],str(image),policy,steps=1)
        assert not stats['failed'],stats
        assert stats['scroll_guard_rolled_back'] and result==dom['html']
        assert stats['actions']==0 and stats['actions_before_scroll_rollback']==1

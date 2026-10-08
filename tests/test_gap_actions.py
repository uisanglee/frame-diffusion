import pytest

from framediff.tree_edits import FIELDS, EditTokenizer, value_valid, value_prefix


@pytest.mark.parametrize('field', ['row-gap', 'column-gap'])
def test_gap_excluded_from_action_grammar_and_diffusion(field):
    from framediff.css_diffusion import grid
    assert field not in FIELDS
    assert len(FIELDS)==6
    assert not value_valid('10px',field)
    assert not value_prefix('1',field)
    with pytest.raises(ValueError):
        EditTokenizer(4).encode([1,field,'10px',''],2)
    with pytest.raises(ValueError):
        grid(field,'px',129)


@pytest.mark.browser
def test_size_edit_preserves_gap_grid_and_parent():
    from framediff import css_owners
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    html='''<html><head><style>
      .parent {display:grid;width:420px;grid-template-columns:1fr 1fr;gap:10px 20px!important}
      .child {height:30px;background:red}
      </style></head><body><div class="parent" id="p">
      <div class="child" id="a"></div><div class="child" id="b"></div>
      <div class="child" id="c"></div></div></body></html>'''
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[600,300],max_nodes=16)
        tree,_,_=dom_tree(dom);target=css_owners.read(browser,tree)
        fixed=css_owners.read(browser,tree,html=browser.page.content(),fixed=True)['fixed']
        owner=next(i for i,o in enumerate(target['owners']) if o.get('selector')=='.parent')
        with pytest.raises(ValueError):
            css_owners.execute(browser,target['owners'],[owner,'row-gap','50px',''])
        css_owners.execute(browser,target['owners'],[owner,'width','500px',''])
        browser.load(browser.page.content(),[600,300])
        assert browser.page.locator('#p').evaluate('e=>getComputedStyle(e).width')=='500px'
        assert browser.page.locator('#p').evaluate('e=>getComputedStyle(e).rowGap')=='10px'
        assert browser.page.locator('#p').evaluate('e=>getComputedStyle(e).columnGap')=='20px'
        assert browser.page.locator('#p').evaluate('e=>getComputedStyle(e).display')=='grid'
        assert browser.page.locator('#c').evaluate('e=>e.parentElement.id')=='p'
        assert css_owners.read(browser,tree,html=browser.page.content(),fixed=True)['fixed']==fixed
        # Gap is immutable context, not silently discarded from compatibility checks.
        altered=browser.page.content().replace('10px 20px','11px 20px')
        assert css_owners.read(browser,tree,html=altered,fixed=True)['fixed']!=fixed


@pytest.mark.browser
def test_gap_only_page_has_no_mutation_sites():
    from framediff import css_owners
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.tree_online import mutation_sites
    with HtmlBrowser() as browser:
        dom=browser.snapshot('<div style="display:flex;gap:10px"><button>A</button><button>B</button></div>',[400,300],max_nodes=16)
        tree,_,_=dom_tree(dom)
        parsed=css_owners.read(browser,tree)
        assert not mutation_sites(parsed['state'])

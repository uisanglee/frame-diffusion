import pytest
from framediff.html_bridge import HtmlBrowser
from framediff.plans import dom_tree
from framediff.visual import annotate
from framediff.safe_sizes import convert


@pytest.mark.parametrize('body,css',[
 ('<div class="p"><div class="c">text</div><div style="height:100px">sibling</div></div>',
  '.p{height:auto;width:200px}.c{height:100%;background:red}'),
 ('<div class="p"><div class="c">A</div><div class="c">B</div></div>',
  '.p{display:grid;grid-template-columns:1fr 1fr;gap:20px;width:300px}.c{height:40px;background:red}'),
 ('<div class="c">fixed</div><div style="height:500px"></div><div>before <span>middle</span> after</div>',
  '.c{position:fixed;top:0;background:red;width:80px;height:30px}'),
])
def test_flow_and_scroll_preserved(tmp_path,body,css):
    with HtmlBrowser() as browser:
        html=f'<html><head><style>body{{margin:0}}{css}</style></head><body>{body}</body></html>'
        dom=browser.snapshot(html,[400,300],None,32)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        source=browser.page.content()
        structure=browser.page.evaluate("[...document.querySelectorAll('body *')].map(e=>[e.tagName,e.parentElement.tagName,getComputedStyle(e).position,getComputedStyle(e).display])")
        qa=convert(browser,source,[400,300],tmp_path,action_ids=[n['id'] for n in tree['nodes'][1:]],max_pixel_mae=.001)
        assert qa['accepted'],qa
        assert qa['scroll_pixel_preservation']['accepted']
        assert structure==browser.page.evaluate("[...document.querySelectorAll('body *')].map(e=>[e.tagName,e.parentElement.tagName,getComputedStyle(e).position,getComputedStyle(e).display])")
        assert not browser.page.locator('[data-tuide-box-layout]').count()


def test_unsafe_height_is_rolled_back_not_a_corruption_label(tmp_path):
    with HtmlBrowser() as browser:
        html='<html><body style="margin:0"><div id="parent" style="width:200px;height:auto"><div style="height:100%">text</div><div style="height:100px">sibling</div></div></body></html>'
        dom=browser.snapshot(html,[400,300],None,32)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        parent_id=browser.page.locator('#parent').get_attribute('data-fd-id')
        qa=convert(browser,browser.page.content(),[400,300],tmp_path,action_ids=[parent_id],max_pixel_mae=.001)
        assert qa['accepted'] and qa['partial_normalization'],qa
        assert qa['full_scroll_fallback'] and qa['scroll_verification_trials']>2,qa
        assert any(r['field']=='height' and r['reason']=='changes_render_or_scroll' for r in qa['rejected_fields'])
        assert browser.page.locator('#parent').evaluate('e=>e.style.height')=='auto'
        assert 'height' not in browser.page.locator('#parent').get_attribute('data-tuide-editable-fields').split()


def test_scroll_only_failure_uses_full_fallback(tmp_path):
    with HtmlBrowser() as browser:
        html='''<html><head><style>body{margin:0}
        #scroll{width:150px;height:100px;overflow:auto}
        .spacer{height:400px}.danger{height:30px;background:red}
        #scroll[data-tuide-explicit-id] .danger{background:blue!important}</style></head>
        <body><div id="scroll"><div class="spacer"></div><div class="danger">below</div></div></body></html>'''
        dom=browser.snapshot(html,[400,200],None,32)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        target=browser.page.locator('#scroll').get_attribute('data-fd-id')
        qa=convert(browser,browser.page.content(),[400,200],tmp_path,
                   action_ids=[target],max_pixel_mae=.001)
        assert qa['accepted'] and qa['partial_normalization'],qa
        assert qa['full_scroll_fallback'] and qa['scroll_verification_trials']>2,qa
        assert any(r['field']=='width' for r in qa['rejected_fields'])

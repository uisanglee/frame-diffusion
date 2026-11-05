import pytest

from framediff.app_edits import apply_application_edits


BEFORE = '''<html><head><style>.card{color:red}</style></head><body data-tuide-app-node="body"><div id="root" data-tuide-app-node="body > div:nth-of-type(1)"><section id="card" style="width:100px" data-tuide-app-node="body > div:nth-of-type(1) > section:nth-of-type(1)"><span data-tuide-app-node="body > div:nth-of-type(1) > section:nth-of-type(1) > span:nth-of-type(1)">0</span></section></div></body></html>'''
AFTER = BEFORE.replace('width:100px', 'width:180px!important;min-width:0px!important')
SOURCE = '''<html><head><style>.card{color:red}</style></head><body><div id="root"></div><button id="increment">Increase</button><script>
let count=0;
function render(){document.getElementById('root').innerHTML='<section id="card" style="width:100px"><span>'+count+'</span></section>';}
render();document.getElementById('increment').addEventListener('click',()=>{count++;render();});
</script></body></html>'''


def test_app_edit_transfer_preserves_source_and_reports_normalization():
    result, edits = apply_application_edits(SOURCE, BEFORE, AFTER)
    assert result.replace(result[result.index('<script id="tuide-application-layout-edits">'):result.rindex('</body>')], '') == SOURCE
    assert edits[0]['selector'].endswith('section:nth-of-type(1)')
    assert {d['kind'] for d in edits[0]['declarations']} == {'normalization', 'layout'}


@pytest.mark.parametrize('after', [
    AFTER.replace('>0<', '>wrong<'),
    AFTER.replace('color:red', 'color:blue'),
    AFTER.replace('width:180px', 'padding:180px'),
    AFTER.replace('id="card"', 'id="other"'),
    AFTER.replace('data-tuide-app-node="body > div:nth-of-type(1) > section:nth-of-type(1)"', ''),
])
def test_app_transfer_rejects_unsupported_or_unbound_edits(after):
    with pytest.raises(ValueError):
        apply_application_edits(SOURCE, BEFORE, after)


def test_no_edits_returns_exact_original():
    assert apply_application_edits(SOURCE, BEFORE, BEFORE) == (SOURCE, [])


def test_margin_cssom_shorthand_and_doctype_serialization_are_not_other_edits():
    before='<!DOCTYPE html>\n'+BEFORE.replace('width:100px','width:100px;margin:calc(2px + 3px) 6px')
    after='<!DOCTYPE html>'+BEFORE.replace('width:100px',
        'width:100px;margin-top:5px!important;margin-right:6px!important;'
        'margin-bottom:5px!important;margin-left:6px!important')
    _,edits=apply_application_edits(SOURCE,before,after)
    assert {d['property'] for d in edits[0]['declarations']}=={
        'margin-top','margin-right','margin-bottom','margin-left'}


@pytest.mark.browser
def test_normalized_tagged_snapshot_can_be_restored_to_live_application(tmp_path):
    from pathlib import Path
    from framediff.app_browser import freeze_application
    from framediff.application_repair import restore_application
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.safe_sizes import convert
    from framediff.visual import annotate
    app='''<!DOCTYPE html><html><head><style>body{margin:0}.card{width:200px;height:120px;min-width:100px;background:#88ccdd;margin:15px}button{width:80px;height:30px}</style></head><body><div id="root"></div><script>
const root=document.getElementById('root');root.innerHTML='<section id="card" class="card" style="padding:0;color:#ff0000"><button id="count">Count 0</button></section>';let count=0;root.firstChild.firstChild.onclick=()=>{count++;document.getElementById('count').textContent='Count '+count;};
</script></body></html>'''
    with HtmlBrowser() as browser:
        frozen=freeze_application(app,[320,240],tmp_path/'before',tmp_path/'assets',browser_host=browser)
        initial={'html':frozen['static_html'],'application_html':frozen['source_html']}
        dom=browser.snapshot(Path(initial['html']).read_text(),[320,240],max_nodes=100)
        tree,_,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        qa=convert(browser,dom['html'],[320,240],tmp_path/'normalized',
                   action_ids=[n['id'] for n in tree['nodes'][1:] if n['visual_class']>0],target_tree=tree)
        assert qa['accepted']
        browser.load((tmp_path/'normalized/normalized.html').read_text(),[320,240])
        browser.page.locator('#card').evaluate("e=>e.style.setProperty('width','180px','important')")
        final=browser.page.content()
        result=restore_application(initial,final,[320,240],tmp_path/'patched',browser_host=browser)
        assert result['application_patch_verification']['accepted']
        assert result['application_patch_verification']['pixel_mae']==0
        assert 'root.firstChild.firstChild.onclick' in Path(result['application_html']).read_text()


@pytest.mark.browser
def test_svg_mixed_case_bindings_are_valid_and_keep_runtime_identity(tmp_path):
    from pathlib import Path
    from framediff.app_browser import freeze_application
    app='''<!DOCTYPE html><html><head></head><body><svg width="100" height="100"><defs><linearGradient id="grad"><stop stop-color="red"/></linearGradient></defs><rect width="100" height="100" fill="url(#grad)"/></svg><p style="width:100px">Text</p></body></html>'''
    frozen=freeze_application(app,[320,240],tmp_path/'snapshot',tmp_path/'assets')
    before=Path(frozen['static_html']).read_text()
    assert 'linearGradient:nth-of-type(1)' in before
    patched,edits=apply_application_edits(app,before,before.replace('width: 100px','width: 110px'))
    assert edits[0]['tag']=='p'
    result=freeze_application(patched,[320,240],tmp_path/'patched',tmp_path/'assets')
    assert result['pixel_mae']==0


@pytest.mark.browser
def test_state_update_survives_and_replacement_nodes_receive_edits():
    from playwright.sync_api import sync_playwright
    patched, _ = apply_application_edits(SOURCE, BEFORE, AFTER.replace('180px','180.0px'))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_content(patched)
            assert page.locator('#card').evaluate('(e)=>e.style.width') == '180px'
            page.locator('#increment').click()
            page.wait_for_function("document.querySelector('#card span').textContent==='1' && document.querySelector('#card').style.width==='180px'")
            assert page.evaluate('window.__tuideApplicationEdits.errors') == []
            # A reused structural slot with different stable identity is not edited.
            page.evaluate("document.querySelector('#root').innerHTML='<section id=unrelated style=width:75px></section>'")
            page.wait_for_function('window.__tuideApplicationEdits.errors.length>0')
            assert page.locator('#unrelated').evaluate('(e)=>e.style.width') == '75px'
        finally:
            browser.close()

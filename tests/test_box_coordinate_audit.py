"""Preservation regressions for coordinate conversion and scroll behavior."""
import io
import json
import numpy as np
import pytest
from PIL import Image
from framediff.html_bridge import HtmlBrowser
from framediff.plans import dom_tree
from framediff.visual import annotate
from framediff.safe_sizes import convert

CASES={
 'fractional_border':('<div class="p"><div class="c">child</div></div>',
     '.p{width:200px;height:100px;border:1.5px solid red;padding:7.25px}.c{width:60px;height:30px;background:blue}'),
 'collapsed_body_margin':('<div class="p">child</div>',
     'body{margin:8px}.p{margin-top:40px;width:120px;height:50px;background:red}'),
 'zero_height_positioned_parent':('<div class="p"><div class="c">child</div></div>',
     '.p{position:relative;left:70px;top:40px;width:200px;height:0}.c{position:absolute;left:15px;top:10px;width:80px;height:30px;background:red}'),
 'inline_text':('<div class="p">before <span>middle</span> after</div>',
     '.p{width:280px;height:50px;background:#eee}span{color:red}'),
 'below_fold_inline_text':('<div style="width:80px;height:30px;background:blue">top</div><div class="p">before <span>middle</span> after</div>',
     '.p{margin-top:450px;width:280px;height:50px;background:#eee}span{color:red}'),
 'table':('<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
     'table{border-collapse:collapse}td{border:2px solid red;padding:10px}'),
 'pseudo_element':('<div class="p">text</div>',
     '.p{width:160px;height:60px;background:#eee}.p::before{content:"before";display:block;height:20px;background:red}'),
 'logical_inset':('<div class="p"><div class="c">child</div></div>',
     '.p{width:200px;height:100px;position:relative;direction:rtl}.c{position:absolute;inset-inline-start:10px!important;width:80px;height:30px;background:red}'),
 'fixed_below_scroll':('<div class="fixed">fixed</div><div class="long">long</div>',
     '.fixed{position:fixed;z-index:10;top:10px;left:10px;width:80px;height:30px;background:red}.long{height:1000px;width:200px;background:#eee}'),
}

@pytest.mark.parametrize('name',CASES)
def test_coordinate_audit(tmp_path,name):
    body,css=CASES[name]
    html=f'<html><head><style>body{{margin:0}}{css}</style></head><body>{body}</body></html>'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[400,300],None,64)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        source=browser.page.content()
        before=browser.page.screenshot(full_page=True)
        browser.page.evaluate('window.scrollTo(0,200)')
        before_scroll=browser.page.screenshot()
        browser.page.evaluate('window.scrollTo(0,0)')
        qa=convert(browser,source,[400,300],tmp_path/name,target_tree=tree,max_pixel_mae=.001,max_box_error=.1)
        after=browser.page.screenshot(full_page=True)
        browser.page.evaluate('window.scrollTo(0,200)')
        after_scroll=browser.page.screenshot()
    def mae(a,b):
        a=np.asarray(Image.open(io.BytesIO(a)).convert('RGB'),dtype=float)
        b=np.asarray(Image.open(io.BytesIO(b)).convert('RGB'),dtype=float)
        return float(np.abs(a-b).mean()/255) if a.shape==b.shape else 'shape_changed'
    result=dict(name=name,accepted=qa['accepted'],viewport_mae=qa.get('pixel_mae'),
        max_box_error=qa.get('max_box_error_px'),full_page_mae=mae(before,after),
        scrolled_mae=mae(before_scroll,after_scroll),report=str(tmp_path/name/'report.json'))
    (tmp_path/name/'audit.json').write_text(json.dumps(result,indent=2))
    for filename,png in [('full-before.png',before),('full-after.png',after),
                         ('scrolled-before.png',before_scroll),('scrolled-after.png',after_scroll)]:
        (tmp_path/name/filename).write_bytes(png)
    print('COORDINATE_AUDIT '+json.dumps(result))
    assert qa['accepted'],qa
    assert result['viewport_mae']==0
    assert result['full_page_mae']==0
    assert result['scrolled_mae']==0
    assert (tmp_path/name/'report.json').is_file()


@pytest.mark.parametrize('kind',['below_text','fixed','nested'])
def test_scroll_guard_rejects_hidden_regressions(tmp_path,kind):
    from framediff.box_scroll_guard import capture,compare
    html='''<html><body style="margin:0"><div id="fixed" style="position:fixed;z-index:10;top:0;background:red">fixed</div>
    <div style="height:500px">top</div><div id="below">original text</div>
    <div style="height:80px;width:200px;overflow:auto"><div style="height:300px">inside</div><div id="nested">nested text</div></div>
    <div style="height:400px"></div></body></html>'''
    with HtmlBrowser() as browser:
        browser.load(html,[400,300])
        browser.page.evaluate("[...document.querySelectorAll('*')].forEach((e,i)=>e.dataset.tuideVerifyId=String(i))")
        before=capture(browser,tmp_path/'before')
        if kind=='fixed':browser.page.evaluate("document.querySelector('#fixed').style.position='absolute'")
        else:browser.page.evaluate("id=>document.querySelector(id).textContent='CHANGED CONTENT'",'#below' if kind=='below_text' else '#nested')
        after=capture(browser,tmp_path/'after',before)
        result=compare(before,after,[],1,.001)
        assert not result['accepted'],result


def test_scroll_budget_is_not_silent_acceptance(tmp_path):
    from framediff.box_scroll_guard import capture
    with HtmlBrowser() as browser:
        browser.load('<html><body style="height:10000px">long</body></html>',[400,300])
        with pytest.raises(ValueError,match='not verified'):
            capture(browser,tmp_path,max_frames=2)


def test_sticky_flow_is_preserved(tmp_path):
    with HtmlBrowser() as browser:
        dom=browser.snapshot('<html><body><div style="position:sticky;top:0">sticky</div><div style="height:500px"></div></body></html>',[400,300],None,32)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        report=convert(browser,browser.page.content(),[400,300],tmp_path,target_tree=tree)
        assert report['accepted'], report
        assert browser.page.locator('div').first.evaluate('e=>getComputedStyle(e).position')=='sticky'

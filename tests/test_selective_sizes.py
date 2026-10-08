import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from framediff.explicit_html import convert
from framediff.html_bridge import HtmlBrowser


@pytest.mark.browser
def test_preserves_complex_content_and_only_freezes_safe_nodes(tmp_path):
    html='''<html><head><style>
    body{margin:0}.card{width:250px;max-width:180px;height:50px;background:red}
    .card::before{content:"icon";color:blue}
    .transformed{transform:scale(.8);width:200px;height:40px}
    .hidden{display:none}.hidden::before{content:"hidden"}
    </style></head><body>
    <div class="card"><span>Text</span></div>
    <div class="transformed"><div id="nested" style="width:80px;height:30px">T</div></div>
    <div class="hidden"></div><div style="width:70px"><span id="multiline">one two three four five six seven</span></div>
    <canvas width="30" height="20"></canvas></body></html>'''
    with HtmlBrowser() as b:
        report=convert(b,html,[500,300],tmp_path,mode='sizes')
        assert report['accepted'],report
        assert report['normalized_elements']>0
        assert b.page.locator('style:not([data-framediff-static])').count()==1
        assert b.page.locator('.card').evaluate('e=>getComputedStyle(e,"::before").content')=='"icon"'
        assert b.page.locator('#nested').get_attribute('data-tuide-explicit-id') is None
        assert b.page.locator('#multiline').get_attribute('data-tuide-explicit-id') is None
        assert b.page.locator('.card').bounding_box()['width']==180
        assert b.page.locator('.card').evaluate('e=>getComputedStyle(e).maxWidth')=='none'
        b.page.locator('.card').evaluate("e=>e.style.setProperty('width','220px','important')")
        assert b.page.locator('.card').bounding_box()['width']==220


@pytest.mark.browser
def test_failed_element_rolls_back_without_discarding_page(tmp_path):
    html='''<html><head><style>
    .safe,.unsafe{position:absolute;width:80px;height:40px;background:red}
    .unsafe{top:100px}
    .unsafe[style*="box-sizing"]{padding-left:200px!important}
    </style></head><body><div class="safe">A</div><div class="unsafe">B</div></body></html>'''
    with HtmlBrowser() as b:
        report=convert(b,html,[400,240],tmp_path,mode='sizes')
        assert report['accepted'],report
        assert report['normalized_elements']==1
        assert any(r['reason']=='render_changed' for r in report['preserved_elements'])
        assert b.page.locator('.unsafe').get_attribute('style') is None
        assert b.page.locator('.unsafe').get_attribute('data-tuide-explicit-id') is None
        assert b.page.locator('.safe').get_attribute('data-tuide-explicit-id') is not None


@pytest.mark.browser
def test_no_eligible_elements_is_explicit_unchanged_fallback(tmp_path):
    with HtmlBrowser() as b:
        report=convert(b,'<span style="transform:none">Hello</span>',[300,100],tmp_path,mode='sizes')
        assert report['accepted'] and report['normalized_elements']==0
        assert report['partial_normalization']


def test_preparation_records_errors_before_split_completion(tmp_path,monkeypatch):
    from framediff import tree_normalize as mod
    from framediff.ir import write_jsonl
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out'
    html=source/'clean.html';html.write_text('<div>Hello</div>')
    for split in ('train','val','test'):
        row=dict(id=split+'/clean',split=split,group=split,source_sha=split,
                 current_html=str(html),viewport=[300,100])
        write_jsonl(source/f'policy-{split}.jsonl',[row])
        write_jsonl(source/f'pages-{split}.jsonl',[dict(id=split)])
    monkeypatch.setattr(mod,'HtmlBrowser',lambda:nullcontext(object()))
    calls=[]
    def fail(*args,**kwargs):
        if calls:
            previous=calls[-1]
            entries=[json.loads(l) for l in (out/f'rejections-{previous}.jsonl').read_text().splitlines()]
            assert entries[0]['stage']=='normalization'
            assert json.loads((out/f'progress-{previous}.json').read_text())['failures']==1
        calls.append(('train','val','test')[len(calls)])
        raise RuntimeError('diagnostic fixture')
    monkeypatch.setattr(mod,'convert',fail)
    args=SimpleNamespace(rendered=str(source),out=str(out),resume=True,seed=42,
                         max_css_owners=64,observation_size=96,max_pixel_mae=.01,max_box_error=1.)
    with pytest.raises(ValueError,match='Empty normalized split'):
        mod.prepare(args)
    assert len(list(out.glob('normalized-pages/*/*/page-status.json')))==3

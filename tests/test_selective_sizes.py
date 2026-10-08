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
def test_changed_batch_is_rejected_without_partial_freezing(tmp_path):
    html='''<html><head><style>
    .safe,.unsafe{position:absolute;width:80px;height:40px;background:red}
    .unsafe{top:100px}
    .unsafe[style*="box-sizing"]{padding-left:200px!important}
    </style></head><body><div class="safe">A</div><div class="unsafe">B</div></body></html>'''
    with HtmlBrowser() as b:
        report=convert(b,html,[400,240],tmp_path,mode='sizes')
        assert not report['accepted'],report
        assert report['verification_trials']==1
        assert not (tmp_path/'normalized.html').exists()
        assert (tmp_path/'rejected.html').exists()


@pytest.mark.browser
def test_no_eligible_elements_is_explicit_unchanged_fallback(tmp_path):
    with HtmlBrowser() as b:
        report=convert(b,'<span style="transform:none">Hello</span>',[300,100],tmp_path,mode='sizes')
        assert report['accepted'] and report['normalized_elements']==0
        assert not report['partial_normalization']


@pytest.mark.browser
def test_snapshot_includes_untagged_percentage_dependencies(tmp_path):
    html='''<!DOCTYPE html><style>body{margin:0}.parent{height:auto;width:200px}
    .child{height:100%;width:100%}.sibling{height:100px}</style>
    <div class="parent" data-fd-id="parent"><div class="child">Text</div>
    <div class="sibling"></div></div>'''
    with HtmlBrowser() as b:
        report=convert(b,html,[400,300],tmp_path,mode='sizes')
        assert report['accepted'],report
        assert b.page.locator('.child').get_attribute('data-tuide-explicit-id') is not None
        assert b.page.locator('.child').bounding_box()['height']<30
        assert b.page.locator('.parent > .child').count()==1


@pytest.mark.browser
def test_logical_css_and_shadowed_percentage_rules_are_readonly(tmp_path):
    from framediff import css_owners
    html='''<!DOCTYPE html><style>body{margin:0}.card{width:50%;height:40px;
    margin-inline:4px;background:red}</style><div class="card" data-fd-id="card">Text</div>'''
    tree={'nodes':[{'id':'root'},{'id':'card'}]}
    with HtmlBrowser() as b:
        report=convert(b,html,[400,200],tmp_path,mode='sizes')
        assert report['accepted'],report
        parsed=css_owners.read(b,tree)
        assert [o['kind'] for o in parsed['owners']]==['root','inline']
        assert parsed['state'][1][0][0]=='200px'
        css_owners.execute(b,parsed['owners'],[1,'width','180px','important'])
        assert b.page.locator('.card').bounding_box()['width']==180
        assert '50%' in b.page.locator('style:not([data-framediff-static])').text_content()
        assert 'margin-inline' in b.page.locator('style:not([data-framediff-static])').text_content()


@pytest.mark.browser
def test_clipping_change_is_rejected_even_on_white_background(tmp_path):
    html='''<!DOCTYPE html><style>body{margin:0}.parent{position:relative;width:100px;height:40px}
    .parent[style*="box-sizing"]{overflow:hidden}.child{position:absolute;left:90px;width:20px;height:10px}</style>
    <div class="parent"><div class="child"></div></div>'''
    with HtmlBrowser() as b:
        report=convert(b,html,[400,200],tmp_path,mode='sizes')
        assert not report['accepted'],report
        assert report['pixel_mae']==0
        assert any(f['clip_error']>=10 for f in report['element_failures'])


@pytest.mark.browser
def test_generic_logical_css_masks_only_ambiguous_fields():
    from framediff import css_owners
    with HtmlBrowser() as b:
        b.load('<div data-fd-id="card" style="width:100px;margin-inline:4px;margin-top:2px"></div>',[400,200])
        parsed=css_owners.read(b,{'nodes':[{'id':'root'},{'id':'card'}]})
        assert parsed['state'][1][0][0]=='100px'
        assert all(not value for value,priority in parsed['state'][1][2:])


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

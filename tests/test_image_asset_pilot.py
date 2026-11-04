import io
import json
from types import SimpleNamespace

from PIL import Image

from framediff.image_asset_pilot import image_references, embed, fetch_image, report


def test_image_sources_resolve_without_stylesheet_or_script_fetch():
    text='''<base href="https://example.org/path/"><link rel="stylesheet" href="style.css">
    <style>@font-face{src:url(font.woff2)}</style><script src="app.js"></script><img src="logo.png"><picture><source srcset="a.png 1x, b.png 2x"></picture>
    <video><source src="movie.mp4"></video><div style="background-image:url(bg.jpg)"></div>'''
    refs=image_references(text,'https://other.org/')
    assert refs=={x:'https://example.org/path/'+x for x in ('logo.png','a.png','b.png','bg.jpg')}


def test_embedding_preserves_css_corruption_and_handles_html_entities():
    source='<img src="https://example.org/a?x=1&amp;y=2" style="width:240px!important">'
    result=embed(source,{'https://example.org/a?x=1&y=2':'data:image/png;base64,ABC'})
    assert 'src="data:image/png;base64,ABC"' in result
    assert 'width:240px!important' in result


def test_download_validates_image_and_caches_success_and_failure(tmp_path,monkeypatch):
    buf=io.BytesIO();Image.new('RGB',(3,3),'red').save(buf,format='PNG')
    calls=[]
    def open_image(*a,**kw):calls.append(1);return io.BytesIO(buf.getvalue())
    monkeypatch.setattr('framediff.image_asset_pilot.urlopen',open_image)
    first=fetch_image('https://example.org/image',tmp_path,2)
    assert first['ok'] and first['uri'].startswith('data:image/png;')
    assert fetch_image('https://example.org/image',tmp_path,2)==first and len(calls)==1
    monkeypatch.setattr('framediff.image_asset_pilot.urlopen',lambda *a,**kw:io.BytesIO(b'<html>Error</html>'))
    assert not fetch_image('https://example.org/bad',tmp_path,2)['ok']


def test_report_paired_deltas_and_html(tmp_path):
    from framediff.ir import write_jsonl
    for mode,final in [('blocked',.8),('images',.7)]:
        folder=tmp_path/mode/'repair';folder.mkdir(parents=True)
        write_jsonl(folder/'results.jsonl',[dict(id='p/v/state',png=str(folder/'page/final.png'),
            initial_box_iou=.3,final_box_iou=final,failed=False)])
    report(SimpleNamespace(out=str(tmp_path)))
    row=json.loads((tmp_path/'paired-results.jsonl').read_text())
    assert abs(row['transfer_delta']+.1)<1e-6
    assert 'images/repair/page/final.png' in (tmp_path/'index.html').read_text()


def test_prepare_remeasures_paired_images_with_same_frozen_edit(tmp_path,monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.image_asset_pilot import prepare
    from framediff import css_owners
    data=tmp_path/'data/7000/css';data.mkdir(parents=True)
    source_dir=data.parent/'source';source_dir.mkdir()
    raw=tmp_path/'raw/page/default_320-240-html.html';raw.parent.mkdir(parents=True)
    raw.write_text('unused');raw.with_name('default_320-240-url.txt').write_text('https://example.org/')
    text='''<html><body style="margin:0"><img src="https://example.org/a.png" style="width:100px;height:80px"><p>Caption</p></body></html>'''
    target=tmp_path/'target.html';initial=tmp_path/'initial.html'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(text,[320,240],max_nodes=16)
        tree,_,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        target.write_text(dom['html'])
        initial.write_text(dom['html'].replace('width:100px','width:50px'))
    row=dict(id='page/default_320-240/sample-0',page_id='page',group='heldout',split='test',
             viewport=[320,240],current=tree,target_html=str(target),current_html=str(initial))
    write_jsonl(data/'policy-test.jsonl',[row])
    write_jsonl(source_dir/'manifest-test.jsonl',[dict(id='page/default_320-240',page_id='page',html=str(raw))])
    buf=io.BytesIO();Image.new('RGB',(100,80),'red').save(buf,format='PNG')
    monkeypatch.setattr('framediff.image_asset_pilot.urlopen',lambda *a,**kw:io.BytesIO(buf.getvalue()))
    out=tmp_path/'pilot'
    args=SimpleNamespace(data=str(data),out=str(out),count=1,max_candidates=2,max_images=4,timeout=2)
    prepare(args)
    assert json.loads((out/'complete.json').read_text())['selected']==1
    left=list(read_jsonl(out/'blocked/policy-test.jsonl'))[0]
    right=list(read_jsonl(out/'images/policy-test.jsonl'))[0]
    assert left['declaration_state']==right['declaration_state']
    assert left['target_declaration_state']==right['target_declaration_state']
    assert 'data:image/png' in __import__('pathlib').Path(right['target_html']).read_text()
    assert json.loads((out/'screening.jsonl').read_text())['target_image_change_mae']>0
    # Preparation can resume without network or browser work.
    monkeypatch.setattr('framediff.image_asset_pilot.urlopen',lambda *a,**kw:(_ for _ in ()).throw(AssertionError()))
    prepare(args)


def test_short_image_reference_cannot_replace_text_or_css_selectors():
    text='<p>a</p><img src="a"><style>.a{width:240px;background:url(a)}</style>'
    result=embed(text,{'a':'data:image/png;base64,ABC'})
    assert '<p>a</p>' in result and '.a{width:240px;' in result
    assert result.count('data:image/png;base64,ABC')==2

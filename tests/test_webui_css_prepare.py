import base64
import hashlib
import io
import json
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image
import pytest

from framediff.css_archive import CSSArchive
from framediff.webui_css_prepare import Stylesheets, Images, prepare_view
from test_css_archive import fixtures


def test_css_first_imports_relative_images_and_order(tmp_path):
    calls=[]
    archive=Stylesheets(tmp_path, fetcher=fixtures({
        'https://site.test/a.css':('@import "nested.css" layer(cards) supports(display: flex) screen and (min-width: 200px);'
            '.a{background-image:var(--art);--art:url(../picture.png)}', 'https://cdn.test/css/a.css'),
        'https://cdn.test/css/nested.css':'.a{display:flex}',
    }, calls))
    result=archive.apply('<html><head><link rel="stylesheet" href="a.css" media="screen">'
                         '<style>.a{color:blue}</style></head><body><div class="a">A</div></body></html>',
                         'https://site.test/index.html')
    soup=BeautifulSoup(result,'html.parser')
    assert not soup.find('link',rel='stylesheet')
    assert soup.style['media']=='screen'
    assert '@layer cards{' in result and '@supports (display: flex)' in result
    assert 'screen and (min-width: 200px)' in result
    assert result.index('display:flex')<result.index('color:blue')
    assert 'https://cdn.test/picture.png' in archive.image_refs
    assert len(calls)==2


def test_missing_css_rejected_not_silently_ignored(tmp_path):
    archive=Stylesheets(tmp_path,fetcher=fixtures({'https://site.test/a.css':OSError('404')},[]))
    with pytest.raises(OSError,match='404'):
        archive.apply('<html><head><link rel="stylesheet" href="a.css"></head><body>hi</body></html>',
                      'https://site.test/')


def test_css_raw_cache_reuse_validates_hash_and_does_not_modify_old(tmp_path):
    calls=[]
    old=CSSArchive(tmp_path/'old',fetcher=fixtures({'https://site.test/a.css':'.a{width:100px}'},calls))
    old.fetch('https://site.test/a.css')
    previous={p:p.read_bytes() for p in (tmp_path/'old/downloads').iterdir()}
    def forbidden(*a):raise AssertionError('Must reuse existing CSS download')
    new=Stylesheets(tmp_path/'new',reuse=[tmp_path/'old/downloads'],fetcher=forbidden)
    assert '.a{width:100px}' in new.sheet('https://site.test/a.css','')
    assert new.events[0]['cache_hit']
    assert all(p.read_bytes()==b for p,b in previous.items())


def image_cache(root,url):
    root.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO();Image.new('RGB',(60,30),'purple').save(buffer,format='PNG')
    data=buffer.getvalue()
    record=dict(url=url,ok=True,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
                uri='data:image/png;base64,'+base64.b64encode(data).decode())
    path=root/(hashlib.sha256(url.encode()).hexdigest()+'.json')
    path.write_text(json.dumps(record))
    return record


def test_image_download_reuse_without_network(tmp_path,monkeypatch):
    from framediff import image_asset_pilot
    record=image_cache(tmp_path/'old','https://site.test/image.png')
    def forbidden(*a):raise AssertionError('Unexpected download')
    monkeypatch.setattr(image_asset_pilot,'_fetch_image_locked',forbidden)
    assert Images([tmp_path/'old'])('https://site.test/image.png',tmp_path/'new',1)==record


@pytest.mark.browser
def test_css_geometry_before_placeholder_and_resume(tmp_path,monkeypatch):
    from framediff import webui_css_prepare as module
    from framediff.html_bridge import HtmlBrowser
    raw=tmp_path/'raw/view-html.html';raw.parent.mkdir()
    raw.write_text('<html><head><link rel="stylesheet" href="a.css"></head><body>'
                   '<div class="row"><img src="image.png"><div class="card">Text</div></div>'
                   '<script>document.body.remove()</script></body></html>')
    raw.with_name('view-url.txt').write_text('https://site.test/page')
    old=tmp_path/'old-images';image_cache(old,'https://site.test/image.png')
    css_cache=CSSArchive(tmp_path/'old-css',fetcher=fixtures({'https://site.test/a.css':
        '@import "child.css" screen and (min-width:200px);body{margin:0}.row{display:flex;gap:12px}img{width:120px;height:60px}',
        'https://site.test/child.css':'.card{width:100px;height:50px}'},[]))
    css_cache.fetch('https://site.test/a.css');css_cache.fetch('https://site.test/child.css')
    row=dict(id='p/view',page_id='p',split='test',group='webui-page:p',original_html=str(raw),viewport=[320,200])
    args=(row,tmp_path/'out',[old],[tmp_path/'old-css/downloads'])
    saved,error,hit=prepare_view(args)
    assert error is None and not hit
    assert saved['split']=='test'
    mapping=json.loads((Path(saved['html']).parent/'images.json').read_text())
    assert mapping[0]['width']==60 and mapping[0]['height']==30
    with HtmlBrowser() as browser:
        browser.load(Path(saved['html']).read_text(),[320,200]);browser.stable_capture()
        assert browser.page.locator('img').evaluate('e=>[e.offsetWidth,e.offsetHeight]')==[120,60]
        assert browser.page.locator('.card').evaluate('e=>[e.offsetLeft,e.offsetTop,e.offsetWidth]')==[132,0,100]
    def no_browser(*a,**kw):raise AssertionError('Completed page must not rerender')
    monkeypatch.setattr('framediff.html_bridge.HtmlBrowser',no_browser)
    assert prepare_view(args)==(saved,None,True)
    raw.write_text(raw.read_text().replace('Text','Changed'))
    with pytest.raises(ValueError,match='changed'):prepare_view(args)


def test_image_set_url_inventory_and_css_variables(tmp_path):
    archive=Stylesheets(tmp_path)
    result=archive.apply('<html><head><style>.x{--picture:url(a.png);background:var(--picture);'
                         'mask-image:image-set("b.png" 1x,url(c.png) 2x)}</style></head><body></body></html>',
                         'https://site.test/path/')
    assert set(archive.image_refs)=={'https://site.test/path/'+name for name in ('a.png','b.png','c.png')}
    assert 'url("https://site.test/path/b.png")' in result


@pytest.mark.browser
def test_full_css_first_pipeline_preserves_three_splits(tmp_path):
    import subprocess
    import sys
    from types import SimpleNamespace
    from framediff.webui_css_prepare import prepare
    from framediff.webui_final_split import publish
    from framediff.ir import write_jsonl, read_jsonl
    old=tmp_path/'old';old.mkdir()
    css=CSSArchive(old/'css-cache',fetcher=fixtures({'https://site.test/a.css':
        'body{margin:0}h1{font-size:22px}.card{width:150px;height:60px;margin:10px;background:#eee}'},[]))
    css.fetch('https://site.test/a.css')
    rows=[]
    for split in ('train','val','test'):
        raw=old/f'{split}-html.html'
        raw.write_text('<html><head><link rel="stylesheet" href="https://site.test/a.css"></head>'
            f'<body><h1>{split} example</h1><div class="card">One {split}</div>'
            '<div class="card" style="width:180px">Two</div><div class="card">Three</div></body></html>')
        rows.append(dict(id=split+'/view',page_id=split,view='view',split=split,group=split,
                         original_html=str(raw),html=str(raw),viewport=[400,400]))
    source=old/'published.jsonl';write_jsonl(source,rows)
    root=tmp_path/'new';out=root/'master/source'
    args=SimpleNamespace(manifest=str(source),out=str(out),workers=1,image_cache=[],css_cache=[str(old/'css-cache/downloads')])
    prepare(args)
    commands=[
        ['visual-build-data','--manifest',str(out/'manifest.jsonl'),'--out',str(root/'master/rendered'),
         '--workers','1','--clean-only','--min-elements','3','--max-source-mae','1','--resume'],
        ['visual-tree-normalize','--rendered',str(root/'master/rendered'),'--out',str(root/'master/labels'),
         '--workers','1','--latest-visual-labels','--clean-targets-only','--final-repartition',
         '--max-css-owners','512','--observation-size','384','--seed','42','--resume']]
    for command in commands:
        completed=subprocess.run([sys.executable,'-m','framediff',*command],capture_output=True,text=True,timeout=120)
        assert completed.returncode==0,completed.stdout+completed.stderr
    report=publish(root,workers=1,preserve_splits_from=out/'preserved-inputs.jsonl')
    assert report['sizes']['7000']==dict(train=1,val=1,test=1)
    for split in ('train','val','test'):
        policy=list(read_jsonl(root/f'7000/css/policy-{split}.jsonl'))
        assert len(policy)==(1 if split=='train' else 5)
        assert all(r['split']==split and r['page_id']==split for r in policy)
        if split!='train':
            assert {r['corruption_geometry']['requested_profile'] for r in policy}=={'clean','mild','medium','strong','multi'}
        assert bool(policy[0].get('clean_target_only'))==(split=='train')
        assert Path(policy[0]['target_image']).is_file()
        if split!='train':assert Path(policy[0]['current_image']).is_file()
    before={p:p.stat().st_mtime_ns for p in (root/'master/source/pages').rglob('*.png')}
    prepare(args)
    assert all(p.stat().st_mtime_ns==value for p,value in before.items())
    artifacts={p:p.stat().st_mtime_ns for p in (root/'master').rglob('*') if p.suffix in ('.png','.html')}
    for command in commands:
        completed=subprocess.run([sys.executable,'-m','framediff',*command],capture_output=True,text=True,timeout=120)
        assert completed.returncode==0,completed.stdout+completed.stderr
    assert publish(root,workers=1,preserve_splits_from=out/'preserved-inputs.jsonl')==report
    assert all(p.stat().st_mtime_ns==value for p,value in artifacts.items())

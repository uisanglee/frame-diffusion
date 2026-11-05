import json
import pytest
from framediff.app_browser import AppBrowser, allowed_dependency, embed_application_assets, freeze_application

APP = '''<!DOCTYPE html><html><head><style>body{margin:0}button{width:140px;height:40px}</style></head>
<body><div id="root"></div><script>
const root=document.getElementById('root');
setTimeout(()=>{root.innerHTML='<button id="count">Count 0</button>';let n=0;
root.firstChild.onclick=()=>{n++;root.firstChild.textContent='Count '+n;};},30);
</script></body></html>'''


def test_dependency_allowlist():
    assert allowed_dependency('https://cdn.jsdelivr.net/npm/react@18.0.0/umd/react.development.js')
    assert allowed_dependency('https://cdn.jsdelivr.net/npm/@babel/standalone@7.28.0/babel.min.js')
    assert allowed_dependency('https://cdn.tailwindcss.com')
    for url in ('http://cdn.tailwindcss.com', 'https://cdn.jsdelivr.net/npm/evil@1/x.js',
                'https://example.org/x.js', 'http://127.0.0.1/a', 'file:///etc/passwd',
                'https://cdn.tailwindcss.com:444/x', 'https://user@cdn.tailwindcss.com'):
        assert not allowed_dependency(url)


def test_reserved_assets_in_jsx_and_javascript_are_embedded_idempotently():
    from framediff.webui_placeholders import image_uri
    source='''<html><body><script type="text/babel">
const view=<img src="rick2.jpg"/>;
const asset=`rick3.jpg`;
const unrelated="not-rick1.jpg";
</script></body></html>'''
    result=embed_application_assets(source)
    assert f'src="{image_uri(1)}"' in result
    assert f'`{image_uri(2)}`' in result
    assert '"not-rick1.jpg"' in result
    assert embed_application_assets(result)==result


def test_dependency_redirects_are_validated_and_cached(tmp_path):
    from types import SimpleNamespace
    class Response:
        def __init__(self,status,headers,body=b'library'):
            self.status=status;self.headers=headers;self.ok=status==200;self.data=body
        def body(self):return self.data
        def dispose(self):pass
    calls=[]
    replies=iter([Response(302,{'location':'/3.4.17'}),Response(200,{'content-type':'text/javascript'})])
    def get(url,**kwargs):calls.append(url);return next(replies)
    browser=AppBrowser(tmp_path/'assets')
    browser.context=SimpleNamespace(request=SimpleNamespace(get=get))
    resource=browser._resource('https://cdn.tailwindcss.com/')
    assert calls==['https://cdn.tailwindcss.com/','https://cdn.tailwindcss.com/3.4.17']
    assert resource['resolved_url']==calls[-1]
    again=AppBrowser(tmp_path/'assets')
    assert again._resource(calls[0])['body']==b'library'
    browser.resources={}
    browser.context.request.get=lambda *a,**k:Response(302,{'location':'http://127.0.0.1/private'})
    with pytest.raises(ValueError,match='Unsafe dependency redirect'):
        browser._resource('https://cdn.tailwindcss.com/3.4.16')


@pytest.mark.browser
def test_app_executes_and_snapshot_has_real_static_dom(tmp_path):
    with AppBrowser(tmp_path/'assets') as browser:
        data=browser.snapshot(APP,[320,240],tmp_path/'live.png',require_content=True)
        assert 'Count 0' in data['html']
        assert '<script' not in data['html']
        assert 'data-tuide-app-node="body &gt; div:nth-of-type(1) &gt; button:nth-of-type(1)"' in data['html']
        browser.page.click('#count')
        assert browser.page.locator('#count').inner_text()=='Count 1'


@pytest.mark.browser
def test_freeze_verifies_current_dom_and_runtime_css_rules(tmp_path):
    app=APP.replace("const root=", "document.styleSheets[0].insertRule('button{background:rgb(10,20,30);color:white}',2);const root=")
    result=freeze_application(app,[320,240],tmp_path/'snapshot',tmp_path/'assets')
    assert result['accepted'] is True
    assert result['pixel_mae']==0
    assert result['max_box_error_px']==0
    assert json.loads((tmp_path/'snapshot/freeze.json').read_text())['contract']==result['contract']
    assert (tmp_path/'snapshot/application.html').read_text()==app
    assert 'rgb(10, 20, 30)' in (tmp_path/'snapshot/static.html').read_text()


@pytest.mark.browser
def test_static_inline_styles_use_cssom_canonical_values(tmp_path):
    from bs4 import BeautifulSoup
    source='<html><body><p style="padding:0;color:#ff0000;width:100px!important">Text</p></body></html>'
    with AppBrowser(tmp_path/'assets') as browser:
        data=browser.snapshot(source,[320,240])
        expected=browser.page.locator('p').evaluate('e=>e.style.cssText')
        assert BeautifulSoup(data['html'],'html.parser').p['style']==expected
        assert '!important' in expected


@pytest.mark.browser
def test_picture_source_is_materialized_without_identity_mismatch(tmp_path):
    from urllib.parse import quote
    uri='data:image/svg+xml,'+quote('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="80"><rect width="100" height="80" fill="red"/></svg>')
    source=f'<html><body><picture><source media="(min-width:1px)" srcset="{uri}"><img src="{uri}"></picture></body></html>'
    result=freeze_application(source,[320,240],tmp_path/'snapshot',tmp_path/'assets')
    assert result['accepted'] is True
    assert result['pixel_mae']==0
    assert '<source' not in (tmp_path/'snapshot/static.html').read_text()


@pytest.mark.browser
def test_javascript_image_uses_local_reserved_asset_and_saved_source(tmp_path,monkeypatch):
    from framediff.local_assets import ORIGIN
    monkeypatch.setenv('TUIDE_LOCAL_ASSET_CACHE',str(tmp_path/'file-assets'))
    source='''<html><body><div id="root"></div><script>
const image=new Image();image.src='rick1.jpg';image.style.cssText='width:80px;height:60px';
document.getElementById('root').appendChild(image);
</script></body></html>'''
    with AppBrowser(tmp_path/'assets') as browser:
        browser.snapshot(source,[320,240])
        assert browser.page.locator('img').evaluate('e=>e.naturalWidth')==1
        assert browser.resources=={}
    result=freeze_application(source,[320,240],tmp_path/'snapshot',tmp_path/'assets')
    assert result['accepted'] is True
    assert result['pixel_mae']==0
    saved=(tmp_path/'snapshot/application.html').read_text()
    assert f"image.src='{ORIGIN}/" in saved
    assert 'data:image/' not in saved
    assert 'rick1.jpg' not in saved
    assert '<script>' in saved


@pytest.mark.browser
def test_localized_images_and_css_background_preserve_pixels_after_copy(tmp_path,monkeypatch):
    import base64
    import shutil
    from PIL import Image
    from framediff.html_bridge import HtmlBrowser
    from framediff.local_assets import ORIGIN
    monkeypatch.setenv('TUIDE_LOCAL_ASSET_CACHE',str(tmp_path/'file-assets'))
    def image_uri(color):
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="30"><rect width="40" height="30" fill="{color}"/></svg>'
        return 'data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()
    source=f'''<!DOCTYPE html><html><head><style>
body{{margin:0}}#background{{width:80px;height:60px;background-image:url("{image_uri('blue')}")}}
</style></head><body><img width="80" height="60" src="{image_uri('red')}"><div id="background"></div></body></html>'''
    baseline=tmp_path/'baseline.png'
    with HtmlBrowser() as browser:
        browser.snapshot(source,[320,240],baseline)
    result=freeze_application(source,[320,240],tmp_path/'snapshot',tmp_path/'dependencies')
    assert result['accepted'] is True
    assert result['pixel_mae']==0
    assert result['max_box_error_px']==0
    for filename in ('application.html','static.html'):
        saved=(tmp_path/'snapshot'/filename).read_text()
        assert ORIGIN+'/' in saved
        assert 'data:image/' not in saved
    copied=tmp_path/'other-directory'/'copied.html'
    copied.parent.mkdir()
    shutil.copyfile(result['static_html'],copied)
    rendered=tmp_path/'copied.png'
    with HtmlBrowser() as browser:
        browser.snapshot(copied.read_text(),[320,240],rendered)
        assert browser.page.locator('img').evaluate('e=>e.naturalWidth')==40
        assert ORIGIN in browser.page.locator('#background').evaluate('e=>getComputedStyle(e).backgroundImage')
    with Image.open(baseline) as original, Image.open(rendered) as actual:
        assert original.size==actual.size
        assert original.convert('RGB').tobytes()==actual.convert('RGB').tobytes()


@pytest.mark.browser
def test_cached_stylesheet_and_font_freeze_to_offline_files(tmp_path,monkeypatch):
    import base64
    import hashlib
    from pathlib import Path
    from PIL import Image
    from playwright.sync_api import APIRequestContext
    from framediff.html_bridge import HtmlBrowser
    from framediff.local_assets import ORIGIN
    monkeypatch.setenv('TUIDE_LOCAL_ASSET_CACHE',str(tmp_path/'file-assets'))
    dependencies=tmp_path/'dependencies'
    dependencies.mkdir()
    stylesheet='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fixture.css'
    font_url='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/webfonts/fixture.ttf'
    font=(Path(__file__).resolve().parents[1]/'figures/assets/InterVariable.ttf').read_bytes()
    svg=b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8" fill="blue"/></svg>'
    image='data:image/svg+xml;base64,'+base64.b64encode(svg).decode()
    css=('@font-face{font-family:FixtureFont;src:url("../webfonts/fixture.ttf") format("truetype");}'
         'body{margin:0}p{font-family:FixtureFont;font-size:30px;line-height:40px;}'
         f'.image{{width:40px;height:40px;background-image:url("{image}");}}').encode()
    for url,body,mime in ((stylesheet,css,'text/css'),(font_url,font,'font/ttf')):
        digest=hashlib.sha256(body).hexdigest()
        (dependencies/digest).write_bytes(body)
        (dependencies/(hashlib.sha256(url.encode()).hexdigest()+'.json')).write_text(json.dumps(
            dict(url=url,resolved_url=url,sha256=digest,content_type=mime)))
    def no_network(*args,**kwargs):
        pytest.fail('Seeded dependency fixture unexpectedly attempted a network download')
    monkeypatch.setattr(APIRequestContext,'get',no_network)
    source=f'<html><head><link rel="stylesheet" href="{stylesheet}"></head><body><p>Offline font</p><div class="image"></div></body></html>'
    with AppBrowser(dependencies) as browser:
        browser.snapshot(source,[320,240])
        assert browser.page.evaluate("[...document.fonts].some(f=>f.family==='FixtureFont'&&f.status==='loaded')")
    result=freeze_application(source,[320,240],tmp_path/'snapshot',dependencies)
    assert result['accepted'] and result['pixel_mae']==0 and result['max_box_error_px']==0
    static=Path(result['static_html']).read_text()
    assert ORIGIN+'/' in static and '.ttf' in static
    assert 'data:' not in static
    assert font_url not in static
    rendered=tmp_path/'offline-font.png'
    with HtmlBrowser() as browser:
        browser.load(static,[320,240])
        browser.stable_capture()
        assert browser.page.evaluate("[...document.fonts].some(f=>f.family==='FixtureFont'&&f.status==='loaded')")
        browser.page.screenshot(path=str(rendered),animations='disabled')
    with Image.open(result['screenshot']) as expected, Image.open(rendered) as actual:
        assert expected.size==actual.size
        assert expected.convert('RGB').tobytes()==actual.convert('RGB').tobytes()


@pytest.mark.browser
@pytest.mark.parametrize('application',[False,True])
def test_missing_local_asset_is_a_render_failure(tmp_path,monkeypatch,application):
    from urllib.parse import urlsplit
    from framediff.html_bridge import HtmlBrowser
    from framediff.local_assets import store_asset
    cache=tmp_path/'file-assets'
    monkeypatch.setenv('TUIDE_LOCAL_ASSET_CACHE',str(cache))
    asset=store_asset(b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="30"/>','image/svg+xml')
    (cache/urlsplit(asset).path.lstrip('/')).unlink()
    source=f'<html><body><p>Visible text must not hide missing assets</p><img src="{asset}"></body></html>'
    browser=AppBrowser(tmp_path/'dependencies') if application else HtmlBrowser()
    with browser:
        with pytest.raises(ValueError,match='asset loading failed|resource loading failed'):
            browser.snapshot(source,[320,240])


@pytest.mark.browser
def test_app_unavailable_dependency_is_failure(tmp_path):
    app=APP.replace('<head>','<head><script src="https://example.org/not-allowed.js"></script>')
    with AppBrowser(tmp_path/'assets') as browser:
        with pytest.raises(ValueError, match='resource loading failed'):
            browser.snapshot(app,[320,240])


@pytest.mark.browser
def test_empty_root_does_not_become_ready(tmp_path):
    with AppBrowser(tmp_path/'assets',timeout_ms=100) as browser:
        with pytest.raises(ValueError,match='did not produce visible content'):
            browser.snapshot('<html><body><div id="root"></div></body></html>',[320,240])


@pytest.mark.browser
def test_freeze_borrows_browser_without_changing_static_context(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    reference='<html><body><h1>Reference</h1><script>document.body.textContent="WRONG"</script></body></html>'
    with HtmlBrowser() as host:
        context=host.context
        host.snapshot(reference,[320,240])
        assert host.page.locator('h1').inner_text()=='Reference'
        report=freeze_application(APP,[320,240],tmp_path/'snapshot',tmp_path/'assets',browser_host=host)
        assert report['pixel_mae']==0
        assert report['browser_executions']==2
        assert report['host_browser_executions']==1
        assert report['host_browser_screenshots']==1
        assert host.context is context and host._runtime.active
        host.snapshot(reference,[320,240])
        assert host.page.locator('h1').inner_text()=='Reference'
        assert host.browser.is_connected()


@pytest.mark.browser
def test_overlay_mapping_errors_are_rejected(tmp_path):
    app=APP.replace('const root=', 'window.__tuideApplicationEdits={errors:["target identity changed"]};const root=')
    with AppBrowser(tmp_path/'assets') as browser:
        with pytest.raises(ValueError,match='overlay mapping failed'):
            browser.snapshot(app,[320,240])


@pytest.mark.browser
def test_freeze_records_failed_loading_costs(tmp_path):
    app=APP.replace('<head>','<head><script src="https://example.org/not-allowed.js"></script>')
    with pytest.raises(ValueError,match='resource loading failed') as failure:
        freeze_application(app,[320,240],tmp_path/'snapshot',tmp_path/'assets')
    report=json.loads((tmp_path/'snapshot/freeze.json').read_text())
    assert report==failure.value.application_capture
    assert report['accepted'] is False
    assert report['error_type']=='ValueError'
    assert report['seconds']>0
    assert report['browser_executions']==1
    assert report['browser_screenshots']==0
    assert report['host_browser_executions']==0
    assert report['host_browser_screenshots']==0


@pytest.mark.browser
def test_freeze_counts_captures_before_stability_failure(tmp_path,monkeypatch):
    error=ValueError('Unstable application')
    def fail_after_capture(self,**kwargs):
        self.page.screenshot()
        raise error
    monkeypatch.setattr(AppBrowser,'stable_capture',fail_after_capture)
    with pytest.raises(ValueError,match='Unstable application') as failure:
        freeze_application(APP,[320,240],tmp_path/'snapshot',tmp_path/'assets')
    assert failure.value is error
    report=json.loads((tmp_path/'snapshot/freeze.json').read_text())
    assert report==error.application_capture
    assert report['browser_executions']==1
    assert report['browser_screenshots']==1


@pytest.mark.browser
def test_freeze_records_failed_visual_verification_and_host_costs(tmp_path,monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    capture=AppBrowser._static_html
    def change_layout(self):
        return capture(self).replace('</head>','<style>button{width:180px!important}</style></head>')
    monkeypatch.setattr(AppBrowser,'_static_html',change_layout)
    with HtmlBrowser() as host:
        with pytest.raises(ValueError,match='failed visual preservation') as failure:
            freeze_application(APP,[320,240],tmp_path/'snapshot',tmp_path/'assets',browser_host=host)
        report=json.loads((tmp_path/'snapshot/freeze.json').read_text())
        assert report==failure.value.application_capture
        assert report['accepted'] is False
        assert report['max_box_error_px']==40
        assert report['browser_executions']==2
        assert report['browser_screenshots']>=5
        assert report['host_browser_executions']==1
        assert report['host_browser_screenshots']==1
        assert host.browser.is_connected()

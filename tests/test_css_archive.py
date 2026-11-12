import json
from pathlib import Path

from bs4 import BeautifulSoup
import pytest

from framediff.css_archive import CSSArchive, ORIGIN


def fixtures(resources, calls):
    def fetch(url, timeout, limit):
        calls.append(url)
        value = resources[url]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, tuple):
            text, final = value
        else:
            text, final = value, url
        return text.encode(), final, 'utf-8'
    return fetch


def local_text(archive, url):
    return (archive.out / 'assets' / url.rsplit('/', 1)[-1]).read_text()


def test_imports_resolve_redirect_base_keep_media_and_block_new_assets(tmp_path):
    calls = []
    fetch = fixtures({
        'https://example.org/site.css': ('@import "nested.css" screen and (min-width: 500px);'
            '.a{background:url("pic.png")}@font-face{font-family:Test;src:url(data:font/woff;base64,AAAA)}',
            'https://cdn.example.org/styles/site.css'),
        'https://cdn.example.org/styles/nested.css': '.a{display:flex}',
    }, calls)
    archive = CSSArchive(tmp_path, fetcher=fetch)
    text = local_text(archive, archive.bundle('/site.css', 'https://example.org/page'))
    assert 'https://cdn.example.org/styles/nested.css' in calls
    assert 'screen and (min-width: 500px)' in text
    assert ORIGIN in text
    assert 'pic.png' not in text and 'data:font' not in text
    assert len(archive.blocked_assets) == 2
    assert len([e for e in archive.events if e['status'] == 'ok']) == 2


def test_existing_images_styles_scripts_preserved_and_active_link_order(tmp_path):
    calls = []
    archive = CSSArchive(tmp_path, fetcher=fixtures({'https://example.org/base/a.css': '.x{width:100px}'}, calls))
    source = '''<!doctype html><html><head><base href="/base/">
    <link rel="stylesheet" href="a.css" media="screen" integrity="old">
    <link rel="alternate stylesheet" href="not-requested.css">
    <style>.x{background:url(data:image/png;base64,AAAA)}</style></head>
    <body><img src="rick1.jpg" data-tuide-natural-width="20" data-tuide-natural-height="10">
    <script>document.body.remove()</script></body></html>'''
    blocked, restored = archive.restore_html(source, 'https://example.org/page')
    a, b = (BeautifulSoup(v, 'html.parser') for v in (blocked, restored))
    assert str(a.body) == str(b.body)
    assert str(a.style) == str(b.style)
    assert b.link['media'] == 'screen' and not b.link.has_attr('integrity')
    assert calls == ['https://example.org/base/a.css']


def test_download_cache_checks_hash_and_preserves_failure(tmp_path):
    calls = []
    fetch = fixtures({'https://example.org/a.css': '.a{color:red}',
                      'https://example.org/missing.css': OSError('404')}, calls)
    for _ in range(2):
        archive = CSSArchive(tmp_path, fetcher=fetch)
        archive.bundle('https://example.org/a.css')
        archive.bundle('https://example.org/missing.css')
    assert len(calls) == 2
    assert any(e['status'] == 'failed' for e in archive.events)
    meta = next(p for p in (tmp_path / 'downloads').glob('*.json') if json.loads(p.read_text()).get('ok'))
    meta.with_suffix('.css').write_text('changed')
    CSSArchive(tmp_path, fetcher=fetch).bundle('https://example.org/a.css')
    assert len(calls) == 3


def test_cycle_and_limits_are_explicit(tmp_path):
    calls = []
    fetch = fixtures({'https://example.org/a.css': '@import "a.css";.a{color:red}'}, calls)
    archive = CSSArchive(tmp_path, max_files=1, fetcher=fetch)
    archive.bundle('https://example.org/a.css')
    archive.bundle('https://example.org/b.css')
    assert len(calls) == 1
    assert {e['status'] for e in archive.events} == {'ok', 'cycle', 'failed'}


def test_embedded_stylesheet_keeps_existing_images_but_restores_import(tmp_path):
    calls = []
    archive = CSSArchive(tmp_path, fetcher=fixtures({'https://example.org/a.css': '.a{display:flex}'}, calls))
    url = archive.bundle('data:text/css,@import "a.css";.a{background:url(data:image/png;base64,AAAA)}', 'https://example.org/page')
    assert 'data:image/png' in local_text(archive, url)
    assert calls == ['https://example.org/a.css']


def test_namespace_urls_preserved_and_image_set_assets_suppressed(tmp_path):
    archive = CSSArchive(tmp_path, fetcher=fixtures({'https://example.org/a.css':
        '@namespace svg url(http://www.w3.org/2000/svg);svg|rect{fill:red}'
        '.a{background-image:image-set("data:image/png;base64,AAAA" 1x,url(pic.png) 2x)}'}, []))
    text = local_text(archive, archive.bundle('https://example.org/a.css'))
    assert 'http://www.w3.org/2000/svg' in text
    assert 'data:image' not in text and 'pic.png' not in text
    assert len(archive.blocked_assets) == 2


@pytest.mark.browser
def test_offline_browser_css_effect_preserves_images_and_disables_scripts(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    archive = CSSArchive(tmp_path, fetcher=fixtures({'https://example.org/a.css':
        'body{margin:0}.row{display:flex;gap:10px}.card{width:80px;height:40px}'}, []))
    source = '<html><head><link rel="stylesheet" href="https://example.org/a.css"></head><body><div class="row"><div class="card">A</div><div class="card">B</div></div><img src="rick1.jpg" data-tuide-natural-width="20" data-tuide-natural-height="10"><script>document.body.remove()</script></body></html>'
    blocked, restored = archive.restore_html(source, '')
    with HtmlBrowser() as browser:
        browser.page.route(ORIGIN + '/**', archive.route)
        browser.load(blocked, [300, 200]); browser.stable_capture()
        before = browser.page.locator('.card').evaluate_all('els=>els.map(e=>[e.offsetLeft,e.offsetTop,e.offsetWidth])')
        before_image = browser.page.locator('img').evaluate('e=>[e.src,e.naturalWidth,e.naturalHeight]')
        browser.load(restored, [300, 200]); browser.stable_capture()
        after = browser.page.locator('.card').evaluate_all('els=>els.map(e=>[e.offsetLeft,e.offsetTop,e.offsetWidth])')
        assert before[1][1] > before[0][1]
        assert after == [[0, 0, 80], [90, 0, 80]]
        assert browser.page.locator('img').evaluate('e=>[e.src,e.naturalWidth,e.naturalHeight]') == before_image
        assert browser.page.locator('body').count() == 1


@pytest.mark.browser
def test_saved_page_replay_resume_and_stale_inputs(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    from PIL import Image
    script = Path(__file__).parents[1] / 'scripts/compare_webui_css.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location('css_replay_script', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls = []
    fetch = fixtures({'https://example.org/a.css': 'body{margin:0}.row{display:flex}.card{width:80px;height:40px}'}, calls)
    real_archive = module.CSSArchive
    monkeypatch.setattr(module, 'CSSArchive', lambda *a, **k: real_archive(*a, **k, fetcher=fetch))
    source = tmp_path / 'view-html.html'
    source.write_text('<html><head><link rel="stylesheet" href="https://example.org/a.css"></head><body><div class="row"><div class="card">A</div><div class="card">B</div></div></body></html>')
    png = tmp_path / 'original.png'; Image.new('RGB', (300, 200), 'white').save(png)
    html_info = dict(path=str(source), available=True, sha256=module.sha256(source))
    image_info = dict(path=str(png), available=True, sha256=module.sha256(png))
    target = dict(id='page/view', viewport=[300, 200], html={'original': html_info, 'color': html_info},
                  images={k: image_info for k in ('original', 'color', 'evaluation')})
    args = SimpleNamespace(timeout=15, max_css=100, retry_failed=False)
    folder, cache = tmp_path / 'page', tmp_path / 'cache'
    first = module.compare_target(target, folder, cache, args)
    assert first['css_complete'] and first['outcome'] == 'css_changes_render'
    assert first['geometry_effect']['changed_elements'] > 0
    assert first['comparisons']['serialization_control']['exact_rgb_pixels']
    second = module.compare_target(target, folder, cache, args)
    assert second['reused'] and len(calls) == 1
    (folder / 'css-restored.png').write_bytes(b'incomplete')
    assert not module.compare_target(target, folder, cache, args)['reused']
    assert len(calls) == 1  # Re-render uses archived CSS, not a fresh download.
    Image.new('RGB', (300, 200), 'black').save(png)
    with pytest.raises(ValueError, match='image changed'):
        module.compare_target(target, folder, cache, args)

import base64
import json
import re
from urllib.parse import quote

import pytest

from framediff.local_assets import (
    ORIGIN, asset_root, externalize_data_uris, read_asset, store_asset,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('TUIDE_LOCAL_ASSET_CACHE', str(tmp_path/'assets'))


def urls(html):
    return re.findall(re.escape(ORIGIN) + r'/[0-9a-f]{64}\.[a-z0-9]+(?:#[\w-]+)?', html)


def test_large_binary_is_deduplicated_and_absent_from_html():
    body = b'font-data' * 500_000
    uri = 'data:font/woff2;base64,' + base64.b64encode(body).decode()
    html = '<style>@font-face{font-family:a;src:url("'+uri+'")}@font-face{font-family:b;src:url('+uri+')}</style>'
    short = externalize_data_uris(html)
    assert len(short) < 500
    first, second = urls(short)
    assert first == second
    assert read_asset(first)['body'] == body
    assert len(list(asset_root().glob('*.woff2'))) == 1
    assert externalize_data_uris(short) == short


def test_svg_charset_fragment_and_javascript_literals_preserved():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><text>caf\xe9</text></svg>'
    uri = 'data:image/svg+xml;charset=iso-8859-1,' + quote(svg) + '#icon'
    source = '<script>const image = `'+uri+'`;</script>'
    result = externalize_data_uris(source)
    url, = urls(result)
    assert result == '<script>const image = `'+url+'`;</script>'
    assert url.endswith('#icon')
    assert read_asset(url) == {'body':svg, 'content_type':'image/svg+xml;charset=iso-8859-1'}


def test_srcset_candidates_preserve_descriptors():
    one = 'data:image/png;base64,YQ=='
    two = 'data:image/png;base64,Yg=='
    for descriptors in [('1x', '2x'), ('100w', '200w')]:
        source = f'<source srcset="{one} {descriptors[0]}, {two} {descriptors[1]}"><img src="{one}">'
        result = externalize_data_uris(source)
        first, second, last = urls(result)
        assert first == last and first != second
        assert f'{first} {descriptors[0]}, {second} {descriptors[1]}' in result
        assert read_asset(first)['body'] == b'a'
        assert read_asset(second)['body'] == b'b'
    result = externalize_data_uris(f'<img srcset="{one}, {two} 2x">')
    first, second = urls(result)
    assert result == f'<img srcset="{first}, {second} 2x">'


def test_raw_svg_in_quoted_css_url_and_unquoted_base64():
    source = '''<style>.a{background:url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"/>')}</style><img src=data:image/png;base64,YQ==>'''
    result = externalize_data_uris(source)
    first, second = urls(result)
    assert read_asset(first)['body'] == b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    assert read_asset(second)['body'] == b'a'


def test_missing_corrupt_or_escaped_files_fail(tmp_path):
    url = store_asset(b'a', 'image/png')
    filename = url.rsplit('/', 1)[1]
    path = asset_root()/filename
    path.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='digest mismatch'):
        read_asset(url)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        read_asset(url)
    outside = tmp_path/'outside.png'
    outside.write_bytes(b'a')
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='escaped cache'):
        read_asset(url)
    for bad in [ORIGIN+'/../outside.png', url+'?file=outside', 'https://example.org/'+filename]:
        with pytest.raises(ValueError, match='Invalid local asset'):
            read_asset(bad)


def test_metadata_cannot_substitute_another_file():
    url = store_asset(b'abc', 'image/png')
    digest = url.rsplit('/', 1)[1].split('.')[0]
    metadata = asset_root()/(digest+'.json')
    data = json.loads(metadata.read_text())
    data['file'] = 'unrelated.png'
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='metadata mismatch'):
        read_asset(url)


def test_javascript_identifiers_and_strings_are_not_resource_uris():
    source = '''<script>
      new MutationObserver(()=>{}).observe(document.body,
        {subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['style','id']});
      const result = {data:[],metadata:true,Data:null};
      const instruction = "data:"; const label = 'data:true,attributes:true';
    </script>'''
    assert externalize_data_uris(source) == source
    assert list(asset_root().iterdir()) == []

"""Content-addressed local assets served offline under a reserved browser origin.

The short URL survives HTML copying to evaluation temporary directories. It is
not a network endpoint: only these registered, digest-checked files are served.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote_to_bytes, urlsplit

ORIGIN = 'https://framediff-assets.invalid'
CONTRACT = 'offline-file-assets-v1'
_URI_HEADER = r'''data:(?:[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+)?(?:;[^\s"'`<>(),]*)?,'''
_QUOTED_DATA = re.compile(r'''(["'`])(''' + _URI_HEADER + r'''[^\r\n]*?)\1''', re.I)
# Do not confuse JavaScript keys such as characterData:true, or data:[], with
# a URI. Rewriting an identifier here would corrupt the preserved application.
_DATA = re.compile(r'''(?<![\w-])''' + _URI_HEADER + r'''[^\s"'`<>)]*''', re.I)
_EXT = {'image/jpeg':'jpg', 'image/png':'png', 'image/svg+xml':'svg',
        'image/gif':'gif', 'image/webp':'webp', 'font/woff2':'woff2',
        'font/woff':'woff', 'font/ttf':'ttf', 'font/otf':'otf'}


def asset_root():
    root = Path(os.environ.get('TUIDE_LOCAL_ASSET_CACHE', '.cache/application-assets')).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic(path, data):
    import tempfile
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        name = Path(handle.name)
        handle.write(data)
    try:
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def store_asset(body, content_type):
    kind, sep, parameters = content_type.strip().partition(';')
    kind = kind.lower() or 'application/octet-stream'
    mime = kind + (sep+parameters if sep else '')
    if '\r' in mime or '\n' in mime:
        raise ValueError('Invalid asset content type')
    digest = hashlib.sha256(mime.encode()+b'\0'+body).hexdigest()
    filename = digest+'.'+_EXT.get(kind, 'bin')
    root = asset_root()
    path = root/filename
    meta = root/(digest+'.json')
    if not path.exists():
        _atomic(path, body)
    if not meta.exists():
        _atomic(meta, json.dumps({'file':filename, 'content_type':mime}).encode())
    return ORIGIN+'/'+filename


def is_local_asset(url):
    parts = urlsplit(url)
    return parts.scheme == 'https' and parts.netloc == 'framediff-assets.invalid'


def read_asset(url):
    parts = urlsplit(url)
    if not is_local_asset(url) or parts.query or not re.fullmatch(r'/[0-9a-f]{64}\.[a-z0-9]+', parts.path):
        raise ValueError('Invalid local asset URL')
    filename = parts.path[1:]
    digest = filename.split('.')[0]
    root = asset_root()
    path, meta = root/filename, root/(digest+'.json')
    if not path.resolve().is_relative_to(root) or not meta.resolve().is_relative_to(root):
        raise ValueError('Local asset escaped cache directory')
    data = json.loads(meta.read_text())
    if data['file'] != filename:
        raise ValueError('Local asset metadata mismatch')
    body = path.read_bytes()
    mime = data['content_type']
    if hashlib.sha256(mime.encode()+b'\0'+body).hexdigest() != digest:
        raise ValueError('Local asset digest mismatch')
    return {'body':body, 'content_type':mime}


def externalize_data_uris(html):
    """Replace encoded URI tokens without reserializing HTML or JavaScript.

    Quoted URIs can contain spaces/markup; srcset descriptors remain outside
    each candidate URI. Browser CSSOM normally percent-encodes text payloads.
    """
    def replace_value(value):
        header, payload = value[5:].split(',', 1)
        parts = header.split(';')
        mime = (parts[0] or 'text/plain') + ''.join(';'+p for p in parts[1:] if p.lower() != 'base64')
        payload, mark, fragment = payload.partition('#')
        raw = unquote_to_bytes(payload)
        body = base64.b64decode(raw, validate=True) if any(p.lower() == 'base64' for p in parts[1:]) else raw
        url = store_asset(body, mime)
        return url + ('#'+fragment if mark else '')
    def quoted(match):
        value = match[2]
        if re.search(r'\s+[\d.]+[wx](?:\s*,|\s*$)|,\s+data:', value):
            # srcset="data:... 1x, data:... 2x" is a list, not one payload.
            def candidate(match):
                uri = match[0].rstrip(',')
                return replace_value(uri) + match[0][len(uri):]
            value = _DATA.sub(candidate, value)
        else:
            value = replace_value(value)
        return match[1]+value+match[1]
    html = _QUOTED_DATA.sub(quoted, html)
    return _DATA.sub(lambda m: replace_value(m[0]), html)


def fulfill_local_asset(route, errors=None):
    request = route.request
    if not is_local_asset(request.url):
        return False
    try:
        if request.method != 'GET' or request.is_navigation_request():
            raise ValueError('Unsupported local asset request')
        resource = read_asset(request.url)
        route.fulfill(body=resource['body'], content_type=resource['content_type'],
                      headers={'access-control-allow-origin':'*'})
    except Exception as error:
        if errors is not None:
            errors.append(str(error))
        route.abort()
    return True

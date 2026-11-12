"""Bounded stylesheet archival for a CSS-only, offline browser comparison."""
import base64
import hashlib
import ipaddress
import json
from pathlib import Path
import socket
import time
from urllib.parse import urljoin, urlsplit, urldefrag, unquote_to_bytes
from urllib.request import Request, HTTPRedirectHandler, build_opener

import tinycss2

ORIGIN = 'https://tuide-css-audit.invalid'


def public_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise ValueError('Expected a public HTTP(S) stylesheet URL')
    addresses = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == 'https' else 80))
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('Non-public stylesheet destination blocked')


class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, timeout, max_bytes):
    public_url(url)
    started = time.monotonic()
    with build_opener(PublicRedirects()).open(
            Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'text/css,*/*;q=0.1'}),
            timeout=timeout) as response:
        mime = response.headers.get_content_type()
        if mime in ('text/html', 'application/xhtml+xml'):
            raise ValueError('Server returned HTML instead of CSS')
        parts, size = [], 0
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes or time.monotonic() - started > timeout:
                raise ValueError('Stylesheet exceeds byte/time limit')
            parts.append(chunk)
        return b''.join(parts), response.geturl(), response.headers.get_content_charset()


def url_value(token):
    if token.type in ('string', 'url'):
        return token.value
    if token.type == 'function' and token.lower_name == 'url':
        args = [x for x in token.arguments if x.type not in ('whitespace', 'comment')]
        if len(args) == 1 and args[0].type == 'string':
            return args[0].value
    return None


class CSSArchive:
    def __init__(self, out, timeout=15, max_files=100, max_bytes=8 * 1024 * 1024,
                 retry_failed=False, fetcher=download, progress=False):
        self.out = Path(out)
        (self.out / 'downloads').mkdir(parents=True, exist_ok=True)
        (self.out / 'assets').mkdir(exist_ok=True)
        self.timeout, self.max_files, self.max_bytes = timeout, max_files, max_bytes
        self.retry_failed, self.fetcher = retry_failed, fetcher
        self.progress = progress
        self.events, self.blocked_assets, self.memo = [], [], {}

    def store(self, text):
        data = text.encode('utf-8')
        name = hashlib.sha256(data).hexdigest() + '.css'
        path = self.out / 'assets' / name
        if not path.exists():
            path.write_bytes(data)
        return ORIGIN + '/' + name

    def fetch(self, url):
        key = hashlib.sha256(url.encode()).hexdigest()
        meta, raw = (self.out / 'downloads' / (key + suffix) for suffix in ('.json', '.css'))
        if meta.exists():
            saved = json.loads(meta.read_text())
            if saved.get('ok') and raw.exists() and hashlib.sha256(raw.read_bytes()).hexdigest() == saved['sha256']:
                if self.progress: print(f'[css cache] {url[:200]}', flush=True)
                return raw.read_bytes(), saved, True
            if not saved.get('ok') and not self.retry_failed:
                raise ValueError('Cached download failure: ' + saved.get('error', 'unknown'))
        try:
            if self.progress: print(f'[css download] {url[:200]}', flush=True)
            if url.startswith('data:'):
                header, payload = url.split(',', 1)
                if not header.lower().startswith('data:text/css'):
                    raise ValueError('Embedded stylesheet is not text/css')
                data = unquote_to_bytes(payload)
                if ';base64' in header.lower():
                    data = base64.b64decode(data, validate=True)
                if len(data) > self.max_bytes:
                    raise ValueError('Embedded stylesheet exceeds byte limit')
                final, charset = url, None
            else:
                data, final, charset = self.fetcher(url, self.timeout, self.max_bytes)
            saved = dict(ok=True, url=url, final_url=final, charset=charset, bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest(), fetched_unix=time.time())
            raw.write_bytes(data)
            meta.write_text(json.dumps(saved, indent=2))
            return data, saved, False
        except Exception as error:
            meta.write_text(json.dumps(dict(ok=False, url=url, error=str(error)), indent=2))
            raise

    def bundle(self, href, base='', stack=()):
        url = urldefrag(urljoin(base, href))[0]
        # Embedded CSS may contain relative imports, resolved in its document context.
        key = (url, base if url.startswith('data:') else '')
        if url in stack:
            self.events.append(dict(url=url, status='cycle', error='Cyclic CSS import omitted'))
            return self.store('/* cyclic import omitted */')
        if key in self.memo:
            return self.memo[key]
        if len(self.memo) >= self.max_files or len(stack) >= 16:
            self.events.append(dict(url=url[:500], status='failed', error='CSS count/depth limit'))
            return self.store('/* stylesheet limit */')
        self.memo[key] = None
        try:
            data, meta, cached = self.fetch(url)
            rules, encoding = tinycss2.parse_stylesheet_bytes(data, protocol_encoding=meta['charset'])
            effective_base = base if url.startswith('data:') else meta['final_url']
            text = self.rewrite_rules(rules, effective_base, (*stack, url),
                                      block_urls=not url.startswith('data:'))
            local = self.store('@charset "UTF-8";\n' + text)
            self.events.append(dict(url=url[:500], status='ok', cache_hit=cached,
                final_url=meta['final_url'][:500], sha256=meta['sha256'], bytes=meta['bytes'],
                encoding=encoding.name, local_url=local))
        except Exception as error:
            self.events.append(dict(url=url[:500], status='failed', error=f'{type(error).__name__}: {error}'))
            local = self.store('/* unavailable stylesheet: see report */')
        self.memo[key] = local
        return local

    def block_asset_urls(self, tokens):
        result = []
        for token in tokens:
            value = url_value(token) if token.type != 'string' else None
            if value is not None and not value.startswith('#'):
                self.blocked_assets.append(dict(url=value[:300], sha256=hashlib.sha256(value.encode()).hexdigest()))
                token = tinycss2.parse_component_value_list('url("about:blank#css-audit-blocked")')[0]
            else:
                if token.type == 'function' and token.lower_name in ('image-set', '-webkit-image-set', 'src'):
                    for i, argument in enumerate(token.arguments):
                        if argument.type == 'string':
                            value = argument.value
                            self.blocked_assets.append(dict(url=value[:300], sha256=hashlib.sha256(value.encode()).hexdigest()))
                            token.arguments[i] = tinycss2.parse_component_value_list('"about:blank#css-audit-blocked"')[0]
                for attr in ('content', 'arguments'):
                    children = getattr(token, attr, None)
                    if children is not None:
                        setattr(token, attr, self.block_asset_urls(children))
            result.append(token)
        return result

    def rewrite_rules(self, rules, base, stack=(), block_urls=False):
        output = []
        for rule in rules:
            if rule.type == 'at-rule' and rule.lower_at_keyword == 'charset':
                continue
            if rule.type == 'at-rule' and rule.lower_at_keyword == 'import':
                first = next((i for i, t in enumerate(rule.prelude) if t.type not in ('whitespace', 'comment')), None)
                href = url_value(rule.prelude[first]) if first is not None else None
                if href is not None:
                    local = self.bundle(href, base, stack)
                    rule.prelude[first:first+1] = tinycss2.parse_component_value_list('url("' + local + '")')
            elif block_urls:
                if getattr(rule, 'content', None) is not None:
                    rule.content = self.block_asset_urls(rule.content)
                # Namespace/document URLs identify selectors/conditions, not
                # downloadable assets. Rewriting them would change CSS meaning.
            if rule.type == 'error':
                self.events.append(dict(url=base[:500], status='parse_error', error=rule.message))
                continue
            output.append(rule)
        return tinycss2.serialize(output)

    def restore_html(self, source, base):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(source, 'html.parser')
        # The control goes through the identical HTML parser before CSS changes.
        control = str(soup)
        base_tag = soup.find('base', href=True)
        if base_tag:
            base = urljoin(base, base_tag['href'])
        for link in soup.find_all('link', href=True):
            rel = [str(r).lower() for r in link.get('rel', [])]
            if 'stylesheet' not in rel or 'alternate' in rel or link.has_attr('disabled'):
                continue
            # Existing embedded/local CSS already works offline. Only restore its
            # imports when embedded; do not suppress its existing image content.
            if link['href'].startswith('https://framediff-assets.invalid/'):
                continue
            link['href'] = self.bundle(link['href'], base)
            link.attrs.pop('integrity', None)  # Original SRI no longer describes rewritten bytes.
        for style in soup.find_all('style'):
            text = style.get_text()
            if '@import' in text.lower():
                style.string = self.rewrite_rules(tinycss2.parse_stylesheet(text), base)
        return control, str(soup)

    def route(self, route):
        name = urlsplit(route.request.url).path.lstrip('/')
        if len(name) != 68 or not name.endswith('.css') or any(c not in '0123456789abcdef' for c in name[:-4]):
            route.abort(); return
        path = self.out / 'assets' / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != name[:-4]:
            route.abort(); return
        route.fulfill(body=path.read_bytes(), content_type='text/css; charset=utf-8')

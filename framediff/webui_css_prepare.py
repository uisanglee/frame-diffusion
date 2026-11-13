"""CSS-first WebUI preparation using an immutable, previously published page split."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin, urldefrag, unquote_to_bytes, urlsplit

import tinycss2
from bs4 import BeautifulSoup

from .css_archive import CSSArchive, url_value
from .ir import read_jsonl
from .preparation_workers import page_map, write_json, write_jsonl
from .preparation_progress import PreparationProgress
from .web_experiment import digest, guard_run
from .webui_placeholders import convert_html, CONTRACT as IMAGE_CONTRACT

CONTRACT = 'webui-css-first-color-v1'


class Stylesheets(CSSArchive):
    """Inline CSS before image source selection; preserve import ordering/conditions."""
    def __init__(self, out, reuse=(), **kwargs):
        super().__init__(out, **kwargs)
        self.reuse = [Path(p) for p in reuse]
        self.image_refs = {}

    def fetch(self, url):
        import fcntl
        key = hashlib.sha256(url.encode()).hexdigest()
        folder = self.out/'downloads'
        with (folder/f'{key}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            meta, raw = folder/f'{key}.json', folder/f'{key}.css'
            if not meta.exists():
                for old in self.reuse:
                    previous, css = old/f'{key}.json', old/f'{key}.css'
                    if not previous.is_file() or not css.is_file(): continue
                    saved = json.loads(previous.read_text())
                    if (saved.get('ok') and saved.get('url') == url and
                            hashlib.sha256(css.read_bytes()).hexdigest() == saved.get('sha256')):
                        raw.write_bytes(css.read_bytes())
                        write_json(meta, saved)
                        break
            return super().fetch(url)

    def urls(self, tokens, base):
        result = []
        for token in tokens:
            value = url_value(token) if token.type != 'string' else None
            if value is not None:
                absolute = value if value.startswith(('data:', '#')) else urljoin(base, value)
                if urlsplit(absolute).scheme in ('http', 'https'):
                    self.image_refs[absolute] = absolute
                token = tinycss2.parse_component_value_list('url('+json.dumps(absolute)+')')[0]
            else:
                # image-set("foo.png" 1x) must also be visible to the image embedder.
                if token.type == 'function' and token.lower_name in ('image-set', '-webkit-image-set'):
                    token.arguments = [tinycss2.parse_component_value_list('url('+json.dumps(t.value)+')')[0]
                                       if t.type == 'string' else t for t in token.arguments]
                for attr in ('content', 'arguments'):
                    children = getattr(token, attr, None)
                    if children is not None: setattr(token, attr, self.urls(children, base))
            result.append(token)
        return result

    def sheet(self, href, base, stack=()):
        url = urldefrag(urljoin(base, href))[0]
        if url in stack:
            self.events.append(dict(url=url, status='cycle'))
            return ''
        key = (url, base if url.startswith('data:') else '')
        if key in self.memo: return self.memo[key]
        if len(self.memo) >= self.max_files or len(stack) >= 16:
            raise ValueError('CSS import count/depth limit exceeded')
        self.memo[key] = ''
        try:
            data, meta, cached = self.fetch(url)
        except Exception as error:
            self.events.append(dict(url=url, status='failed', error=str(error)))
            raise
        rules, _ = tinycss2.parse_stylesheet_bytes(data, protocol_encoding=meta.get('charset'))
        if stack and any(r.type == 'at-rule' and r.lower_at_keyword == 'namespace' for r in rules):
            raise ValueError('Imported @namespace cannot be safely flattened')
        text = self.rules(rules, base if url.startswith('data:') else meta['final_url'], (*stack, url))
        self.memo[key] = text
        self.events.append(dict(url=url, status='ok', cache_hit=cached, sha256=meta['sha256']))
        return text

    def rules(self, rules, base, stack=()):
        output = []
        for rule in rules:
            if rule.type == 'error':
                self.events.append(dict(status='parse_warning', error=rule.message))
                continue  # Browser also ignores invalid top-level rules.
            if rule.type == 'at-rule' and rule.lower_at_keyword == 'charset': continue
            if rule.type == 'at-rule' and rule.lower_at_keyword == 'import':
                tokens = [t for t in rule.prelude if t.type not in ('whitespace', 'comment')]
                href = url_value(tokens.pop(0)) if tokens else None
                if href is None: raise ValueError('Unsupported CSS @import syntax')
                text = self.sheet(href, base, stack)
                wrappers = []
                if tokens and tokens[0].type == 'ident' and tokens[0].value.lower() == 'layer':
                    tokens.pop(0); wrappers.append('@layer')
                elif tokens and tokens[0].type == 'function' and tokens[0].lower_name == 'layer':
                    wrappers.append('@layer '+tinycss2.serialize(tokens.pop(0).arguments))
                if tokens and tokens[0].type == 'function' and tokens[0].lower_name == 'supports':
                    condition = tinycss2.serialize(tokens.pop(0).arguments)
                    wrappers.append('@supports ('+condition+')')
                # Serialize the original remaining tokens with separators: bare
                # identifiers such as "screen and" must not become "screenand".
                if tokens: wrappers.append('@media '+' '.join(tinycss2.serialize([t]) for t in tokens))
                for wrapper in reversed(wrappers): text = wrapper+'{'+text+'}'
                output.append(text)
                continue
            if getattr(rule, 'content', None) is not None:
                rule.content = self.urls(rule.content, base)
            output.append(tinycss2.serialize([rule]))
        return '\n'.join(output)

    def apply(self, text, base):
        soup = BeautifulSoup(text, 'html.parser')
        base_tag = soup.find('base', href=True)
        if base_tag: base = urljoin(base, base_tag['href'])
        # Snapshot tags so newly inlined styles are not processed a second time.
        for tag in list(soup.find_all(['link', 'style'])):
            if tag.name == 'link':
                rel = [str(r).lower() for r in tag.get('rel', [])]
                if 'stylesheet' not in rel or 'alternate' in rel or tag.has_attr('disabled'): continue
                if not tag.get('href'): raise ValueError('Stylesheet link has no href')
                css = self.sheet(tag['href'], base)
                style = soup.new_tag('style')
                if tag.has_attr('media'): style['media'] = tag['media']
                style.string = css.replace('</style', '<\\/style')
                tag.replace_with(style)
            else:
                tag.string = self.rules(tinycss2.parse_stylesheet(tag.get_text()), base).replace('</style', '<\\/style')
        for tag in soup.find_all(style=True):
            tag['style'] = tinycss2.serialize(self.urls(tinycss2.parse_component_value_list(tag['style']), base))
        if not soup.html: raise ValueError('Expected complete raw HTML')
        soup.html['data-tuide-css-contract'] = CONTRACT
        return str(soup)


class Images:
    def __init__(self, reuse): self.reuse = [Path(p) for p in reuse]

    def __call__(self, url, cache, timeout):
        import fcntl
        from .image_asset_pilot import _fetch_image_locked
        cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(url.encode()).hexdigest()
        with (cache/f'{key}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            dest = cache/f'{key}.json'
            for path in [dest, *(p/f'{key}.json' for p in self.reuse)]:
                if not path.is_file(): continue
                saved = json.loads(path.read_text())
                if saved.get('ok') and saved.get('url') == url:
                    header, body = saved['uri'].split(',', 1)
                    data = base64.b64decode(body) if ';base64' in header else unquote_to_bytes(body)
                    if hashlib.sha256(data).hexdigest() == saved.get('sha256'):
                        if path != dest: write_json(dest, saved)
                        return saved
                elif path == dest and not saved.get('ok'):
                    return saved
            # Corrupt successes are never reused; failures from old runs are retried.
            dest.unlink(missing_ok=True)
            return _fetch_image_locked(url, cache, timeout)


def prepare_view(task):
    row, out, image_caches, css_caches = task
    from .html_bridge import HtmlBrowser
    folder = out/'pages'/hashlib.sha256(row['id'].encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    record = folder/'record.json'
    raw = Path(row['original_html']).resolve()
    sha = digest(raw)
    url_file = Path(str(raw).replace('-html.html', '-url.txt'))
    base = url_file.read_text().strip() if url_file != raw and url_file.is_file() else row.get('url', '')
    if record.exists():
        saved = json.loads(record.read_text())
        if (saved['raw_input_sha'] != sha or saved.get('source_url') != base or
                digest(saved['html']) != saved['source_sha'] or
                digest(saved['screenshot']) != saved['screenshot_sha']):
            raise ValueError('Cached source/artifact changed; use a new output folder')
        return saved, None, True
    try:
        text = raw.read_text()
        if 'data-tuide-safe-sizes' in text or 'data-tuide-image-contract' in text:
            raise ValueError('original_html must be raw, before normalization/image replacement')
        css = Stylesheets(out/'css-cache', reuse=css_caches)
        try:
            styled = css.apply(text, base)
        finally:
            write_json(folder/'css-report.json', css.events)
        (folder/'css-applied.html').write_text(styled)
        with HtmlBrowser() as browser:
            target, mapping = convert_html(styled, base, out/'downloads', browser, row['viewport'],
                extra_refs=css.image_refs, image_fetcher=Images(image_caches))
            html = folder/'target.html'; html.write_text(target)
            screenshot = folder/'target.png'
            browser.page.screenshot(path=str(screenshot), animations='disabled')
        saved = {**row, 'html':str(html), 'screenshot':str(screenshot), 'source_sha':digest(html),
                 'raw_input_sha':sha, 'original_html':str(raw), 'source_url':base, 'screenshot_sha':digest(screenshot),
                 'css_contract':CONTRACT, 'image_asset_contract':IMAGE_CONTRACT,
                 'css_report':str(folder/'css-report.json'), 'css_applied_html':str(folder/'css-applied.html')}
        saved.pop('source_files', None)
        saved.pop('initial_html', None)
        write_json(folder/'images.json', mapping); write_json(record, saved)
        return saved, None, False
    except Exception as error:
        return None, dict(id=row['id'], page_id=row['page_id'], split=row['split'], error=str(error)), False


def prepare(args):
    from .webui_final_split import preserved_assignment
    source = Path(args.manifest).resolve(); out = Path(args.out).resolve()
    if source == out or out in source.parents or source.parent in out.parents:
        raise ValueError('Use a separate output folder, not the previous published data')
    rows = list(read_jsonl(source))
    preserved_assignment(rows)
    for row in rows:
        if not row.get('original_html') or not Path(row['original_html']).is_file():
            raise ValueError(f'{row["id"]}: missing original_html; raw source is required')
        row['page_id'] = str(row.get('page_id', row['id'].split('/')[0]))
        row['source_group'] = row.get('source_group', row.get('group'))
        row['group'] = 'webui-page:'+row['page_id']
    guard_run(out, dict(contract=CONTRACT, source=digest(source),
        code=digest(__file__), conversion=digest(Path(__file__).with_name('webui_placeholders.py')),
        archive=digest(Path(__file__).with_name('css_archive.py')),
        image_download=digest(Path(__file__).with_name('image_asset_pilot.py')),
        font_policy='external-fonts-blocked', css_failure_policy='reject-view'), True)
    write_jsonl(out/'preserved-inputs.jsonl', rows)
    progress = PreparationProgress('css-first-images', len(rows), out/'progress.json')
    kept, errors, hits = [], [], 0
    progress.update(0, kept=0, failures=0, cache_hits=0, workers=args.workers)
    for index, (saved, error, hit) in enumerate(page_map(prepare_view,
            ((r,out,args.image_cache,args.css_cache) for r in rows), args.workers), 1):
        if error: errors.append(error); print(json.dumps(error), flush=True)
        else: kept.append(saved); hits += int(hit)
        progress.update(index, kept=len(kept), failures=len(errors), cache_hits=hits, workers=args.workers)
    from .visual_data import validate_splits
    write_jsonl(out/'rejections.jsonl', errors)
    validate_splits(kept)
    if not kept: raise ValueError('No CSS/image-complete views survived; see rejections.jsonl')
    write_jsonl(out/'manifest.jsonl', kept)
    for split in ('train','val','test'):
        write_jsonl(out/f'manifest-{split}.jsonl', [r for r in kept if r['split']==split])
    write_json(out/'report.json',dict(contract=CONTRACT, kept=len(kept), failed=len(errors), cache_hits=hits,
        policy='All required stylesheet downloads must succeed; external fonts and JavaScript remain disabled'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--image-cache', action='append', default=[])
    p.add_argument('--css-cache', action='append', default=[])
    args=p.parse_args()
    if args.workers < 1: p.error('workers must be positive')
    prepare(args)

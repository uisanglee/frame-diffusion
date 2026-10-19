"""Hardlink screenshot-policy inputs into a self-contained, resumable dataset.

No source files are removed. Immutable assets are hardlinked; JSON is rewritten
into separate files. Never modify linked assets in place. Cross-device links
fail explicitly, without silently copying large files.
Run again to verify or resume; READY is only written after all inputs validate.
"""
import argparse
import errno
import hashlib
from html import unescape
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(out):
    summary = json.loads((out/'summary.json').read_text())
    def check(value):
        if isinstance(value, dict):
            for item in value.values():check(item)
        elif isinstance(value, list):
            for item in value:check(item)
        elif isinstance(value, str) and value.startswith('/'):
            path = Path(value)
            if path.is_symlink() or not path.resolve().is_relative_to(out) or not path.is_file():
                raise ValueError(f'Non-independent or missing reference: {value}')
    for name, count in summary['rows'].items():
        rows = [json.loads(line) for line in (out/name).read_text().splitlines() if line.strip()]
        if len(rows) != count:raise ValueError(f'Row count changed: {name}')
        for row in rows:check(row)
    with (out/'asset-checksums.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line);path = out/record['path']
            check(str(path))
            if digest(path) != record['sha256']:raise ValueError(f'Asset changed: {path}')
            if path.suffix == '.json':check(json.loads(path.read_text()))
    print(f'Verified independent dataset: {out}', flush=True)


class Exporter:
    def __init__(self, out):
        self.out = out
        self.done = {}
        self.url_warnings = 0

    def asset(self, path):
        path = Path(os.path.abspath(path))
        if path in self.done:
            return self.done[path]
        if not path.is_file():
            raise FileNotFoundError(f'Referenced input is missing or not a file: {path}')
        target = self.out/'assets'/str(path).lstrip('/')
        if path.is_relative_to(self.out):
            raise ValueError(f'Input references output folder: {path}')
        self.done[path] = str(target)
        # Preserve HTML bytes (including source_sha) and relative asset layout.
        if path.suffix.lower() in ('.html', '.htm', '.css'):
            text = path.read_text()
            references = re.findall(r'''(?:src|href|poster)\s*=\s*["']([^"']+)["']|url\(\s*["']?([^\s)'";]+)''', text, re.I)
            for pair in references:
                ref = unescape(next(value for value in pair if value)).strip()
                try:
                    parsed = urlsplit(ref)
                except ValueError as exc:
                    # Broken remote links are part of the original HTML, not
                    # dependencies to copy. Do not change the HTML itself.
                    scheme = re.match(r'^([a-zA-Z][a-zA-Z0-9+.-]*):', ref)
                    remote = ref.startswith('//') or (scheme and scheme[1].lower() != 'file')
                    if not remote:
                        raise ValueError(f'Cannot safely resolve local resource in {path}: {ref}') from exc
                    self.out.mkdir(parents=True, exist_ok=True)
                    with (self.out/'url-warnings.jsonl').open('a') as log:
                        log.write(json.dumps({'html':str(path), 'url':ref, 'error':str(exc),
                                              'action':'ignored_invalid_external_url'}, ensure_ascii=False)+'\n')
                    self.url_warnings += 1
                    continue
                if parsed.scheme not in ('', 'file') or ref.startswith(('#', '//')):
                    continue
                # Decode only the path, after parsing; %23 and %3F can be
                # literal filename characters rather than URL delimiters.
                local = Path(unquote(parsed.path))
                if not parsed.path:
                    continue
                candidate = local if local.is_absolute() else path.parent/local
                if candidate.is_file():
                    if local.is_absolute() or parsed.scheme == 'file':
                        raise ValueError(f'Absolute local URL in {path}: {ref}; embed this resource before exporting')
                    self.asset(candidate)
            placeholder = path.parent/'rick.jpg'
            if path.suffix.lower() in ('.html', '.htm') and placeholder.is_file():
                self.asset(placeholder)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == '.json':
            content = json.dumps(self.rewrite(json.loads(path.read_text())), ensure_ascii=False)
            self.write(target, content)
        elif target.is_symlink() or not target.is_file() or not os.path.samefile(path, target):
            # An atomic replacement also converts outputs from the copy-based
            # exporter, without modifying either the old asset or the source.
            with tempfile.TemporaryDirectory(prefix='.link-', dir=target.parent) as staging:
                temporary = Path(staging)/'asset'
                try:
                    os.link(path.resolve(), temporary)
                except OSError as exc:
                    if exc.errno == errno.EXDEV:
                        raise ValueError(f'Hardlink requires the same filesystem: {path} -> {target}; no copy performed') from exc
                    raise
                os.replace(temporary, target)
        return str(target)

    def rewrite(self, value):
        if isinstance(value, dict):
            return {key: self.rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.rewrite(item) for item in value]
        if isinstance(value, str) and value.startswith('/'):
            return self.asset(value)
        return value

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(text)
        os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data')
    parser.add_argument('--out', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[7000])
    parser.add_argument('--verify', action='store_true', help='Verify exported assets without reading original folders')
    args = parser.parse_args()
    if args.verify:
        verify(Path(args.out).resolve())
        return
    if not args.data:parser.error('--data is required for export')
    source, out = Path(args.data).resolve(), Path(args.out).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        parser.error('Source and output must be separate, non-nested folders')
    manifests = [source/str(size)/'css'/f'{kind}-{split}.jsonl'
                 for size in args.sizes for split in ('train', 'val', 'test')
                 for kind in ('policy', 'pages')]
    config = {'version': 1, 'inputs': {str(p): digest(p) for p in manifests}}
    out.mkdir(parents=True, exist_ok=True)
    config_path = out/'export-config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Input manifests changed; use a new output folder')
    Exporter.write(config_path, json.dumps(config, indent=2))
    (out/'READY').unlink(missing_ok=True)
    exporter = Exporter(out)
    (out/'url-warnings.jsonl').write_text('')
    counts = {}
    for manifest in manifests:
        target = out/manifest.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        count = 0
        with manifest.open() as inp, temporary.open('w') as dest:
            for line in inp:
                if not line.strip():
                    continue
                dest.write(json.dumps(exporter.rewrite(json.loads(line)), ensure_ascii=False) + '\n')
                count += 1
                if count % 1000 == 0:
                    print(f'{manifest.name}: {count} rows, {len(exporter.done)} assets', flush=True)
        os.replace(temporary, target)
        counts[str(target.relative_to(out))] = count
        print(f'Exported {target}: {count} rows', flush=True)
    Exporter.write(out/'summary.json', json.dumps({'rows': counts, 'assets': len(exporter.done),
        'invalid_external_urls': exporter.url_warnings,
        'source_preserved': True, 'asset_mode': 'hardlink',
        'warning': 'Do not modify linked HTML/images in place; JSON/manifests are independent.'}, indent=2))
    with (out/'asset-checksums.jsonl').open('w') as stream:
        for target in exporter.done.values():
            path = Path(target)
            stream.write(json.dumps({'path':str(path.relative_to(out)), 'sha256':digest(path)})+'\n')
    verify(out)
    (out/'READY').write_text('All manifests exported; immutable assets hardlinked and verified.\n')
    print(f'Complete: {out}. Source folders have NOT been deleted.', flush=True)


if __name__ == '__main__':
    main()

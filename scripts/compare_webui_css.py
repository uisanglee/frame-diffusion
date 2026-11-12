"""Replay the saved WebUI pilot with only external CSS restored locally."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_target_provenance import compare_images, sha256
from framediff.css_archive import CSSArchive, ORIGIN


GEOMETRY = '''()=>({viewport:[innerWidth,innerHeight],
  scroll:[document.documentElement.scrollWidth,document.documentElement.scrollHeight],
  elements:[...document.querySelectorAll('body,body *')]
    .filter(e=>!['SCRIPT','STYLE','LINK','META','NOSCRIPT'].includes(e.tagName)).map((e,index)=>{
      const r=e.getBoundingClientRect(),s=getComputedStyle(e);
      return {index,tag:e.tagName,box:[r.x,r.y,r.width,r.height],display:s.display,
        font:s.fontFamily,visible:!!e.getClientRects().length&&s.visibility!=='hidden'&&s.display!=='none'
          &&r.right>0&&r.bottom>0&&r.left<innerWidth&&r.top<innerHeight};})})'''


def geometry_delta(a, b):
    if len(a['elements']) != len(b['elements']) or any(
            x['tag'] != y['tag'] for x, y in zip(a['elements'], b['elements'])):
        return {'comparable': False, 'reason': 'DOM order/structure changed'}
    changed = []
    for x, y in zip(a['elements'], b['elements']):
        error = max(abs(p-q) for p, q in zip(x['box'], y['box']))
        if error > .25 or x['visible'] != y['visible'] or x['display'] != y['display']:
            changed.append(dict(index=x['index'], tag=x['tag'], before=x['box'], after=y['box'],
                                max_error_px=error, visible_before=x['visible'], visible_after=y['visible']))
    return {'comparable': True, 'changed_elements': len(changed), 'changes': changed,
            'visible_before': sum(e['visible'] for e in a['elements']),
            'visible_after': sum(e['visible'] for e in b['elements']),
            'scroll_before': a['scroll'], 'scroll_after': b['scroll']}


def source_url(target):
    original = Path(target['html']['original']['path'])
    url_file = original.with_name(original.name.replace('-html.html', '-url.txt'))
    if url_file != original and url_file.is_file():
        return url_file.read_text().strip()
    for candidate in target.get('color_source_candidates', []):
        if candidate['row'].get('url'):
            return candidate['row']['url']
    return ''  # Absolute stylesheet links still work; unresolved relative links fail explicitly.


def check_inputs(target):
    for name in ('original', 'color'):
        item = target['html'][name]
        if not item.get('available') or not Path(item['path']).is_file():
            raise ValueError(f'Missing {name} HTML; rerun target provenance audit after locating it')
        if sha256(item['path']) != item.get('sha256'):
            raise ValueError(f'{name} HTML changed since provenance audit; rerun that audit')
    for check in target.get('source_hash_checks', {}).values():
        if check.get('matches') is False:
            raise ValueError('Source hash mismatch in provenance report; do not mix preprocessing versions')


def capture(browser, text, viewport, folder, name):
    browser.load(text, viewport)
    png, stability = browser.stable_capture(timeout_ms=15000)
    (folder / (name + '.png')).write_bytes(png)
    geometry = browser.page.evaluate(GEOMETRY)
    (folder / (name + '-dom.json')).write_text(json.dumps(geometry, indent=2))
    return geometry, stability


def compare_target(target, folder, shared, args):
    check_inputs(target)
    source = Path(target['html']['color']['path']).read_text()
    base = source_url(target)
    viewport = target['viewport']
    identity = {'id': target['id'], 'viewport': viewport, 'base_url': base,
                'source_sha': sha256(target['html']['color']['path']),
                'raw_sha': sha256(target['html']['original']['path']),
                'script_sha': sha256(__file__), 'archive_sha': sha256(Path(__file__).parents[1] / 'framediff/css_archive.py'),
                'timeout': args.timeout, 'max_files': args.max_css,
                'images': {name: {'path': info.get('path'),
                    'sha256': sha256(info['path']) if info.get('path') and Path(info['path']).is_file() else None}
                    for name, info in target['images'].items()}}
    for name, info in target['images'].items():
        if identity['images'][name]['sha256'] is not None and identity['images'][name]['sha256'] != info.get('sha256'):
            raise ValueError(f'{name} image changed since provenance audit; rerun that audit')
    saved = folder / 'result.json'
    if saved.is_file() and not args.retry_failed:
        result = json.loads(saved.read_text())
        artifacts = result.get('artifacts', {})
        if result.get('identity') == identity and artifacts and all(
                Path(p).is_file() and sha256(p) == h for p, h in artifacts.items()):
            return {**result, 'reused': True}
    archive = CSSArchive(shared, timeout=args.timeout, max_files=args.max_css,
                         retry_failed=args.retry_failed, progress=True)
    control, restored = archive.restore_html(source, base)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'blocked.html').write_text(control)
    (folder / 'css-restored.html').write_text(restored)
    from framediff.html_bridge import HtmlBrowser
    with HtmlBrowser() as browser:
        browser.page.route(ORIGIN + '/**', archive.route)
        # No browser network is enabled. The route serves digest-checked local CSS only.
        original_geo, _ = capture(browser, source, viewport, folder, 'source-replay')
        blocked_geo, _ = capture(browser, control, viewport, folder, 'blocked')
        failed = []
        browser.page.on('requestfailed', lambda request: failed.append(
            {'url': request.url[:500], 'kind': request.resource_type}))
        restored_geo, _ = capture(browser, restored, viewport, folder, 'css-restored')
    comparisons = {
        'serialization_control': compare_images(folder / 'source-replay.png', folder / 'blocked.png'),
        'css_effect': compare_images(folder / 'blocked.png', folder / 'css-restored.png'),
    }
    for name, field in [('original', 'original'), ('saved-color', 'color'), ('evaluation', 'evaluation')]:
        info = target['images'][field]
        path = Path(info['path']) if info.get('path') else None
        if path and path.is_file():
            if info.get('sha256') != sha256(path):
                raise ValueError(f'{name} image changed since provenance audit')
            shutil.copyfile(path, folder / (name + path.suffix))
        comparisons[name + '_to_blocked'] = compare_images(path, folder / 'blocked.png')
        comparisons[name + '_to_restored'] = compare_images(path, folder / 'css-restored.png')
    errors = [e for e in archive.events if e['status'] != 'ok']
    controls_match = comparisons['serialization_control'].get('exact_rgb_pixels') is True
    geometry_control = geometry_delta(original_geo, blocked_geo)
    controlled = controls_match and geometry_control.get('comparable') and not geometry_control['changed_elements']
    effect = comparisons['css_effect'].get('exact_rgb_pixels') is False
    outcome = ('confounded_by_html_serialization' if not controlled else
               'css_changes_render' if effect else 'no_observed_css_effect')
    if errors:
        outcome += '_partial_css'
    result = dict(id=target['id'], identity=identity, folder=str(folder), outcome=outcome,
                  css_complete=not errors, css_events=archive.events, blocked_new_css_assets=archive.blocked_assets,
                  blocked_browser_requests=failed, comparisons=comparisons, geometry_control=geometry_control,
                  geometry_effect=geometry_delta(blocked_geo, restored_geo), reused=False)
    paths = [p for p in folder.iterdir() if p.name != 'result.json' and p.is_file()]
    paths.extend(shared / 'assets' / e['local_url'].rsplit('/', 1)[-1]
                 for e in archive.events if e.get('local_url'))
    result['artifacts'] = {str(p): sha256(p) for p in paths}
    saved.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def write_summary(out, results):
    lines = ['# WebUI CSS-only replay', '',
        'Same saved color-image HTML and viewport. No training, VLM calls or normalization.',
        'JavaScript and browser networking stay disabled. Archived CSS is served locally.',
        'New image/font URLs inside downloaded CSS are suppressed; existing placeholders are unchanged.',
        'CSS downloads reflect current servers, not necessarily the historical WebUI capture.',
        'An effect proves CSS changes this rendering, not that the historical design is fully restored.', '',
        '| Page | CSS loaded / errors | Changed elements | CSS effect MAE | Outcome |',
        '|---|---:|---:|---:|---|']
    sections = []
    for result in results:
        title = html.escape(result['id'])
        if result.get('error'):
            lines.append(f'| {result["id"]} | -- | -- | -- | ERROR: {result["error"]} |')
            sections.append(f'<h2>{title}</h2><pre>{html.escape(result["error"])}</pre>')
            continue
        events = result['css_events']
        loaded, errors = sum(e['status'] == 'ok' for e in events), sum(e['status'] != 'ok' for e in events)
        mae = result['comparisons']['css_effect'].get('rgb_mae')
        changed = result['geometry_effect'].get('changed_elements', '--')
        lines.append(f'| {result["id"]} | {loaded} / {errors} | {changed} | {mae} | {result["outcome"]} |')
        folder = Path(result['folder'])
        cards = []
        for prefix, label in [('original', 'Original dataset capture'), ('saved-color', 'Saved color target'),
                              ('blocked', 'CSS blocked control'), ('css-restored', 'CSS restored'),
                              ('evaluation', 'Previous evaluation target')]:
            candidates = sorted(p for p in folder.glob(prefix + '.*') if p.suffix in ('.png', '.webp', '.jpg', '.jpeg'))
            body = 'Missing saved image'
            if candidates:
                relative = candidates[0].relative_to(out).as_posix()
                body = f'<a href="{relative}"><img loading="lazy" src="{relative}"></a>'
            cards.append(f'<section><h3>{label}</h3>{body}</section>')
        css_errors = [e for e in events if e['status'] != 'ok']
        sections.append(f'<h2>{title}</h2><p>{html.escape(result["outcome"])}</p><div class="row">{"".join(cards)}</div>'
                        f'<details><summary>CSS failures</summary><pre>{html.escape(json.dumps(css_errors, indent=2))}</pre></details>')
    lines.extend(['', 'Inspect index.html for visual recovery. Original-vs-color RGB errors also include intentional placeholder changes.',
                  'A missing stylesheet or missing effect cannot exclude a CSS cause. Download failures remain partial results.', ''])
    for result in results:
        for event in result.get('css_events', []):
            if event['status'] != 'ok':
                lines.append(f'- {result["id"]}: {event["url"]} — {event.get("error", event["status"])}')
    (out / 'README.md').write_text('\n'.join(lines))
    (out / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CSS replay</title>'
        '<style>body{font-family:system-ui;margin:24px}.row{display:flex;gap:16px;overflow:auto}section{flex:0 0 360px}img{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>External CSS replay on saved WebUI samples</h1><p>Images remain colored. JavaScript and new font/image downloads are disabled. CSS may differ from the historical capture.</p>' + ''.join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='runs/target-provenance-audit/report.json')
    parser.add_argument('--out', default='runs/webui-css-replay10-v1')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--max-css', type=int, default=100)
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed CSS downloads and rerender comparisons')
    args = parser.parse_args()
    if args.count < 1 or args.timeout <= 0 or args.max_css < 1:
        parser.error('count, timeout, max-css must be positive')
    report = json.loads(Path(args.audit).read_text())
    targets = report['targets'][:args.count]
    if not targets:
        parser.error('Provenance report has no targets')
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    results = []
    for index, target in enumerate(targets, 1):
        print(f'[{index}/{len(targets)}] {target["id"]}: archiving CSS and comparing', flush=True)
        key = hashlib.sha256(json.dumps([target['id'], target['viewport'], target['html']['color']['path']]).encode()).hexdigest()[:20]
        try:
            result = compare_target(target, out / 'pages' / key, out / 'css-cache', args)
        except Exception as error:
            result = dict(id=target['id'], error=f'{type(error).__name__}: {error}')
        results.append(result)
        write_summary(out, results)
        print(json.dumps({k: result.get(k) for k in ('id', 'outcome', 'css_complete', 'reused', 'error')}), flush=True)
    print(f'Report: {out / "README.md"}\nGallery: {out / "index.html"}', flush=True)
    if any(r.get('error') for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

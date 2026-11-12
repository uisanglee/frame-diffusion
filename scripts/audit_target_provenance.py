"""Audit saved WebUI target stages without rendering, downloading, or loading models.

Run from the repository root. Missing server files remain missing evidence; this
script never substitutes a normalized screenshot for the original capture.
"""
import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil


STAGES = (
    ('original', 'Original dataset capture'),
    ('color', 'Color-image source'),
    ('tagged', 'Tagged normalization input'),
    ('normalized', 'Normalized target'),
    ('evaluation', 'Experiment target'),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value, parent=None):
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.exists() and parent is not None:
        candidate = Path(parent) / path
        if candidate.exists():
            path = candidate
    return path.resolve()


def file_info(path):
    if path is None:
        return {'path': None, 'available': False}
    path = Path(path)
    info = {'path': str(path), 'available': path.is_file()}
    if info['available']:
        try:
            info.update(bytes=path.stat().st_size, sha256=sha256(path))
        except OSError as error:
            info.update(available=False, error=str(error))
    return info


def read_json(path, issues):
    if path is None or not Path(path).is_file():
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        issues.append(f'Cannot read JSON {path}: {error}')
        return {}


def jsonl(path, issues):
    try:
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError as error:
                    issues.append(f'{path}:{number}: {error}')
    except OSError as error:
        issues.append(f'Cannot read {path}: {error}')


class CSSInventory(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.styles, self.inline = [], [], []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'link' and 'stylesheet' in (attrs.get('rel') or '').lower().split():
            self.links.append({key: attrs.get(key) for key in ('href', 'media', 'disabled')
                               if key in attrs})
        if 'style' in attrs:
            self.inline.append(attrs['style'] or '')
        if tag == 'style':
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == 'style' and self.current is not None:
            self.styles.append(''.join(self.current))
            self.current = None


def css_inventory(path):
    info = file_info(path)
    if not info['available']:
        return info
    try:
        parser = CSSInventory()
        parser.feed(Path(path).read_text(errors='replace'))
        for link in parser.links:
            href = link.get('href') or ''
            link['resource_kind'] = ('embedded_data_uri' if href.startswith('data:') else
                'bundled_local_asset' if href.startswith('https://framediff-assets.invalid/') else
                'external_or_relative' if href else 'missing_href')
        if parser.current is not None:
            parser.styles.append(''.join(parser.current))
        try:
            import tinycss2
        except ImportError:
            tinycss2 = None
        imports, blocks = [], []
        for css in parser.styles:
            block = {'characters': len(css)}
            if tinycss2 is not None:
                rules = tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True)
                block['rules'] = sum(rule.type in ('qualified-rule', 'at-rule') for rule in rules)
                block['parse_errors'] = sum(rule.type == 'error' for rule in rules)
                imports.extend(tinycss2.serialize(rule.prelude).strip() for rule in rules
                               if rule.type == 'at-rule' and rule.lower_at_keyword == 'import')
            else:
                imports.extend(re.findall(r'@import\s+([^;]+)', css, re.I))
            blocks.append(block)
        info.update(stylesheet_links=parser.links, embedded_styles=blocks,
                    embedded_css_characters=sum(map(len, parser.styles)),
                    inline_style_count=len(parser.inline),
                    inline_style_characters=sum(map(len, parser.inline)),
                    css_imports=imports,
                    css_parser='tinycss2' if tinycss2 is not None else 'regex-import-fallback')
    except Exception as error:
        info['inventory_error'] = f'{type(error).__name__}: {error}'
    return info


def compare_images(left, right):
    """Compare saved pixels, never resize or compare substituted screenshots."""
    result = {'left': str(left) if left else None, 'right': str(right) if right else None}
    if not left or not right or not Path(left).is_file() or not Path(right).is_file():
        return {**result, 'status': 'unavailable', 'reason': 'missing_image'}
    try:
        from PIL import Image, ImageChops, ImageStat
        with Image.open(left) as a, Image.open(right) as b:
            result.update(left_size=list(a.size), right_size=list(b.size))
            if a.size != b.size:
                return {**result, 'status': 'unavailable', 'reason': 'dimension_mismatch'}
            diff = ImageChops.difference(a.convert('RGB'), b.convert('RGB'))
            channels = diff.split()
            maximum = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
            changed = a.width * a.height - maximum.histogram()[0]
            return {**result, 'status': 'compared', 'exact_rgb_pixels': changed == 0,
                    'rgb_mae': sum(ImageStat.Stat(diff).mean) / (3 * 255),
                    'changed_pixel_fraction': changed / (a.width * a.height)}
    except Exception as error:
        return {**result, 'status': 'unavailable', 'reason': f'{type(error).__name__}: {error}'}


def page_view(row):
    parts = str(row.get('id', '')).split('/')
    return (str(row.get('page_id') or parts[0]),
            str(row.get('view') or (parts[1] if len(parts) > 1 else '')))


def select_targets(run, count, issues):
    """Count page/views, retaining every saved target variant for those views."""
    selected, variants = set(), {}
    for manifest in sorted(Path(run).rglob('prepare/prepared.jsonl')):
        for row in jsonl(manifest, issues):
            if not ('webui' in manifest.parts or str(row.get('group', '')).startswith('webui')
                    or str(row.get('dataset', '')).lower() == 'webui'):
                continue
            view = (*page_view(row), tuple(row.get('viewport', [])))
            if view not in selected:
                if len(selected) >= count:
                    continue
                selected.add(view)
            target = file_info(resolve(row.get('screenshot'), manifest.parent))
            normalized = resolve(row.get('html'), manifest.parent)
            key = (view, str(normalized), target.get('sha256') or target['path'])
            origin = {'manifest': str(manifest.resolve()), 'id': row.get('id'),
                      'model': row.get('vlm_model'), 'target_path': target['path']}
            if key not in variants:
                variants[key] = {'record': row, 'manifest': manifest, 'origins': []}
            variants[key]['origins'].append(origin)
    return list(variants.values()), len(selected)


def source_index(data, wanted, issues):
    root = Path(data) / 'master/source'
    manifest = root / 'manifest.jsonl'
    paths = [manifest] if manifest.is_file() else sorted(root.glob('manifest-*.jsonl'))
    index = {}
    for path in paths:
        for row in jsonl(path, issues):
            key = page_view(row)
            if key in wanted:
                index.setdefault(key, []).append((row, path))
    for key in wanted - index.keys():
        record = root / 'pages' / hashlib.sha256('/'.join(key).encode()).hexdigest() / 'record.json'
        row = read_json(record, issues)
        if row:
            index[key] = [(row, record)]
    return index


def hash_check(path, expected):
    actual = file_info(path)
    return {**actual, 'expected_sha256': expected,
            'matches': actual.get('sha256') == expected if expected and actual['available'] else None}


def inspect_target(entry, sources):
    row, manifest = entry['record'], entry['manifest']
    issues = []
    normalized = resolve(row.get('html'), manifest.parent)
    folder = normalized.parent if normalized else manifest.parent
    record_path, stamp_path, report_path = (folder / name for name in
                                            ('record.json', 'normalization.json', 'report.json'))
    record = read_json(record_path, issues)
    stamp = read_json(stamp_path, issues)
    qa = read_json(report_path, issues)
    tagged = resolve(record.get('page', {}).get('original_html') or row.get('original_html'), folder)
    tagged_record_path = tagged.parent / 'records.json' if tagged else None
    if tagged and not tagged_record_path.is_file():
        tagged_record_path = tagged.parent / 'record.json'  # Legacy outputs.
    tagged_record = read_json(tagged_record_path, issues)
    candidates = sources.get(page_view(row), [])
    distinct = {}
    for candidate, location in candidates:
        key = (candidate.get('html'), candidate.get('source_sha'), candidate.get('original_html'))
        distinct[key] = (candidate, location)
    source, source_location = {}, None
    if len(distinct) == 1:
        source, source_location = next(iter(distinct.values()))
    elif len(distinct) > 1:
        issues.append('Conflicting color-source records; no source was silently selected.')
    else:
        issues.append('Color-source manifest/record could not resolve this page/view.')
    color = resolve(source.get('html'), source_location.parent if source_location else None)
    color_record = read_json(color.parent / 'record.json', issues) if color else {}
    original = resolve(source.get('original_html') or color_record.get('original_html'),
                       color.parent if color else None)
    original_candidates = []
    if original and original.name.endswith('-html.html'):
        stem = original.name[:-len('-html.html')]
        original_candidates = [original.with_name(stem + '-screenshot.' + suffix)
                               for suffix in ('webp', 'png', 'jpg', 'jpeg')]
    available_originals = [p for p in original_candidates if p.is_file()]
    if len(available_originals) > 1:
        issues.append('Multiple original screenshot formats exist; gallery uses first candidate; all paths reported.')
    images = {
        'original': available_originals[0] if available_originals else None,
        'color': resolve(source.get('screenshot') or color_record.get('screenshot'),
                         color.parent if color else None),
        'tagged': resolve(tagged_record.get('page', {}).get('screenshot'), tagged.parent) if tagged else None,
        'normalized': folder / 'after.png' if normalized else None,
        'evaluation': resolve(row.get('screenshot'), manifest.parent),
    }
    if tagged and images['tagged'] is None:
        images['tagged'] = tagged.parent / 'target.png'
    inventories = {name: css_inventory(path) for name, path in
                   [('original', original), ('color', color), ('tagged', tagged), ('normalized', normalized)]}
    comparisons = {f'{a}_to_{b}': compare_images(images[a], images[b]) for a, b in
                   [('original', 'color'), ('color', 'tagged'), ('tagged', 'normalized'),
                    ('normalized', 'evaluation'), ('original', 'evaluation')]}
    checks = {'normalization_input': hash_check(tagged, stamp.get('source_sha')),
              'normalized_html': hash_check(normalized, stamp.get('sha')),
              'record_input': hash_check(tagged, record.get('source_sha')),
              'assets': [hash_check(resolve(path, folder), expected)
                         for path, expected in record.get('assets', {}).items()]}
    source_checks = {
        'raw_input': hash_check(original, source.get('raw_input_sha') or color_record.get('raw_input_sha')),
        'color_source': hash_check(color, source.get('source_sha') or color_record.get('source_sha')),
        'color_to_tagged_source': hash_check(color, tagged_record.get('page', {}).get('source_sha')),
    }
    findings = []
    original_css = inventories['original']
    if any(link.get('resource_kind') == 'external_or_relative'
           for link in original_css.get('stylesheet_links', [])) or original_css.get('css_imports'):
        findings.append('Original HTML references external/relative stylesheets or CSS imports. The current static pipeline blocks unbundled external resources; embedded data CSS and bundled local assets are exceptions. Their visual contribution is not proven by this file-only audit.')
    if qa.get('accepted') is True:
        findings.append('Saved normalization QA reports preservation of its immediate input; it does not establish fidelity to the original dataset screenshot.')
    elif qa.get('accepted') is False:
        findings.append('Saved normalization QA rejected preservation; inspect the full report and hashes.')
    for name in ('color_to_tagged', 'tagged_to_normalized', 'normalized_to_evaluation'):
        comparison = comparisons[name]
        if comparison.get('status') == 'compared':
            findings.append(f'{name}: saved RGB pixels are ' + ('identical.' if comparison['exact_rgb_pixels'] else 'different; inspect this stage, without assigning a cause from pixels alone.'))
    if comparisons['original_to_evaluation'].get('exact_rgb_pixels') is False:
        findings.append('Original capture differs from evaluation target. Color placeholders are intentional; CSS/resource omission, fonts, browser differences, and normalization are not separated by this difference alone. No CSS ablation was performed.')
    for name, info in inventories.items():
        if not info['available']:
            issues.append(f'Missing {name} HTML evidence: {info["path"]}')
    for name, path in images.items():
        if not path or not path.is_file():
            issues.append(f'Missing {name} image evidence: {path}')
    for path in (record_path, stamp_path, report_path):
        if not path.is_file():
            issues.append(f'Missing normalization evidence: {path}')
    if any(check.get('matches') is False for key, check in checks.items() if key != 'assets') or any(
            check.get('matches') is False for check in checks['assets']):
        findings.append('A saved hash disagrees with the current file. Normalization QA cannot be trusted for that changed artifact without resolving provenance.')
    if any(check['matches'] is False for check in source_checks.values()):
        findings.append('A raw/color/tagged source hash disagrees. Page/view IDs match, but these stages may belong to different preprocessing versions; do not interpret them as a verified chain.')
    if source_checks['color_to_tagged_source']['matches'] is None:
        issues.append('Color-to-tagged source hash linkage unavailable; the color-source join uses page/view only.')
    return {'id': row.get('id'), 'page_view': list(page_view(row)), 'viewport': row.get('viewport'),
            'origins': entry['origins'], 'images': {name: file_info(path) for name, path in images.items()},
            'html': inventories, 'comparisons': comparisons, 'findings': findings, 'issues': issues,
            'original_screenshot_candidates': [file_info(p) for p in original_candidates],
            'color_source_candidates': [{'manifest': str(path), 'row': value} for value, path in candidates],
            'tagged_record_path': str(tagged_record_path) if tagged_record_path else None,
            'source_hash_checks': source_checks,
            'normalization': {'record_path': str(record_path), 'record': record,
                              'stamp_path': str(stamp_path), 'stamp': stamp,
                              'report_path': str(report_path), 'report': qa, 'hash_checks': checks},
            'provided_screenshot': file_info(resolve(row.get('provided_screenshot'), manifest.parent))}


def write_reports(out, report):
    out = Path(out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    sections = []
    lines = ['# Saved target provenance audit', '',
             'File-only audit: no rendering, downloads, preprocessing, or model calls.',
             'Original captures may intentionally differ because images were replaced with colors.',
             'Image URL replacement can intentionally change CSS text; CSS changes alone are not damage.',
             'Pixel differences alone do not prove a CSS or normalization cause.', '',
             f'Page/views: {report["selected_page_views"]}; target variants: {len(report["targets"])}.', '']
    lines.extend(['| Target | Original CSS links / imports | Normalization QA MAE / box error (px) | Tagged → normalized | Normalized → evaluation |',
                  '|---|---:|---:|---|---|'])
    def comparison_summary(comparison):
        if comparison['status'] != 'compared':
            return 'unavailable: ' + comparison['reason']
        return 'identical' if comparison['exact_rgb_pixels'] else f'different (MAE {comparison["rgb_mae"]:.6g})'
    for index, target in enumerate(report['targets'], 1):
        css = target['html']['original']
        counts = f'{len(css.get("stylesheet_links", []))} / {len(css.get("css_imports", []))}' if css['available'] else 'unavailable'
        qa = target['normalization']['report']
        values = f'{qa.get("pixel_mae", "--")} / {qa.get("max_box_error_px", "--")}'
        comparisons = target['comparisons']
        title = str(target['id']).replace('|', '&#124;')
        lines.append(f'| {index}. {title} | {counts} | {values} | {comparison_summary(comparisons["tagged_to_normalized"])} | {comparison_summary(comparisons["normalized_to_evaluation"])} |')
    lines.extend(['', 'CSS link counts include embedded/local stylesheets; resource kinds are distinguished in report.json. QA values describe the saved immediate-input preservation test.', ''])
    for index, target in enumerate(report['targets'], 1):
        cards = []
        for stage, label in STAGES:
            image = target['images'][stage]
            if image['available']:
                source = Path(image['path'])
                relative = f'images/{index:03d}-{stage}{source.suffix.lower()}'
                shutil.copyfile(source, out / relative)
                image['gallery_file'] = relative
                body = f'<a href="{relative}"><img loading="lazy" src="{relative}" alt="{html.escape(label)}"></a>'
            else:
                body = '<p>Missing saved evidence</p>'
            cards.append(f'<section><h3>{html.escape(label)}</h3>{body}</section>')
        title = f'{index}. {target["id"]}'
        notes = ''.join(f'<li>{html.escape(text)}</li>' for text in target['findings'] + target['issues'])
        sections.append(f'<article><h2>{html.escape(title)}</h2><div class="stages">{"".join(cards)}</div><ul>{notes}</ul><details><summary>Image comparisons</summary><pre>{html.escape(json.dumps(target["comparisons"], indent=2))}</pre></details></article>')
        lines.extend([f'## {title}', '', *('- ' + text for text in target['findings'] + target['issues']), ''])
    if report['issues']:
        lines.extend(['## Collection issues', '', *('- ' + text for text in report['issues']), ''])
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Target provenance</title>'
        '<style>body{font:16px system-ui;margin:24px;color:#18202a}article{border-top:2px solid #aaa;margin-top:32px}.stages{display:flex;gap:12px;overflow-x:auto}section{flex:0 0 360px}img{width:100%;border:1px solid #bbb}pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:6px 0}</style>'
        '<h1>Saved target provenance</h1><p>No rendering or downloads. Click a saved image for its original dimensions. Color replacement is intentional; a visual difference alone does not identify its cause.</p>'
        '<p><a href="report.json">Full evidence and original paths</a> · <a href="README.md">Summary</a></p>' + ''.join(sections))


def audit(run, data, out, count=10):
    issues = []
    entries, page_count = select_targets(run, count, issues)
    if not entries:
        raise ValueError(f'No WebUI prepare/prepared.jsonl records found under {run}')
    sources = source_index(data, {page_view(entry['record']) for entry in entries}, issues)
    report = {'protocol': 'saved-target-provenance-v1', 'run': str(Path(run).resolve()),
              'data': str(Path(data).resolve()), 'requested_page_views': count,
              'selected_page_views': page_count, 'issues': issues,
              'targets': [inspect_target(entry, sources) for entry in entries]}
    groups = {}
    for index, target in enumerate(report['targets']):
        key = json.dumps([target['page_view'], target['viewport']])
        groups.setdefault(key, []).append(index)
    report['multiple_target_variants'] = [indices for indices in groups.values() if len(indices) > 1]
    write_reports(out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/paper-color-pilot10-seed42-n2-v1')
    parser.add_argument('--data', default='data/new')
    parser.add_argument('--out', default='runs/target-provenance-audit')
    parser.add_argument('--count', type=int, default=10, help='Page/views; retain all target variants within them')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    try:
        report = audit(args.run, args.data, args.out, args.count)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({'page_views': report['selected_page_views'], 'target_variants': len(report['targets']),
                      'gallery': str((Path(args.out) / 'index.html').resolve()),
                      'report': str((Path(args.out) / 'report.json').resolve())}))


if __name__ == '__main__':
    main()

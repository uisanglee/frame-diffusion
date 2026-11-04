"""Export two original WebUI screenshots and their evaluated local renders."""
import argparse
import html
import json
import shutil
from pathlib import Path


def original_index(data_root, raw_root=None):
    """Resolve original screenshots from published source manifests, not after.png."""
    originals = {}
    def add(path):
        path = Path(path)
        if not path.is_file() or '-screenshot.' not in path.name:
            return
        key = (path.parent.name, path.name.rsplit('-screenshot.', 1)[0])
        values = originals.setdefault(key, [])
        if path not in values:
            values.append(path)
    if raw_root is not None:
        for pattern in ('*-screenshot.webp', '*-screenshot.png'):
            for path in raw_root.rglob(pattern):
                add(path)
    else:
        manifests = sorted(set(data_root.glob('*/7000/source/manifest-test.jsonl')) |
                           set(data_root.glob('*/master/source/manifest-test.jsonl')) |
                           set(data_root.glob('*/source/manifest-test.jsonl')))
        # Source manifests preserve the paths to raw WebUI screenshots even when
        # normalized test manifests point to after.png instead.
        for manifest in manifests:
            with manifest.open() as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        if row.get('screenshot'):
                            add(row['screenshot'])
        print(f'Original source manifests: {len(manifests)}; available page/views: {len(originals)}')
    return originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs', help='Experiment root containing prepare/prepared.jsonl')
    parser.add_argument('--raw-root', help='Optional override; otherwise read original paths from source manifests')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--out', default='runs/webui-image-comparison')
    parser.add_argument('--count', type=int, default=2)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    raw_root = Path(args.raw_root) if args.raw_root else None
    if raw_root is not None and not raw_root.is_dir():
        parser.error(f'Original WebUI folder not found: {raw_root}')
    manifests = sorted(Path(args.run).rglob('prepare/prepared.jsonl'),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        parser.error('No prepare/prepared.jsonl found under --run')
    originals = original_index(Path(args.data_root), raw_root)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = []
    seen = set()
    for manifest in manifests:
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row.get('target_image_source') != 'reference-render':
                continue
            parts = row['id'].split('/')
            page = row.get('page_id') or parts[0]
            view = row.get('view') or (parts[1] if len(parts) > 1 else '')
            candidates = originals.get((page, view), [])
            rendered = Path(row['screenshot'])
            if page in seen or not candidates or not rendered.is_file():
                continue
            # Ambiguous raw copies must not be silently selected if contents differ.
            import hashlib
            hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}
            if len(hashes) != 1:
                print(f'Skipping conflicting original copies: {page}/{view}')
                continue
            original = sorted(candidates)[0]
            number = len(selected) + 1
            left = f'{number}-original{original.suffix}'
            right = f'{number}-rendered.png'
            shutil.copy2(original, out / left)
            shutil.copy2(rendered, out / right)
            selected.append(dict(id=row['id'], original=str(original), rendered=str(rendered),
                                 html=row.get('html'), left=left, right=right))
            seen.add(page)
            if len(selected) >= args.count:
                break
        if len(selected) >= args.count:
            break
    if not selected:
        parser.error('Source manifests did not resolve available original/rendered pairs. Original files may be missing or their stored paths may belong to another server; no normalized image was substituted.')
    sections = []
    for row in selected:
        sections.append(f'<h2>{html.escape(row["id"])}</h2><div class="pair">'
                        f'<section><h3>Original WebUI screenshot</h3><a href="{row["left"]}"><img src="{row["left"]}"></a></section>'
                        f'<section><h3>Local reference render</h3><a href="{row["right"]}"><img src="{row["right"]}"></a></section></div>')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>WebUI comparison</title>'
        '<style>body{font-family:sans-serif;margin:24px}.pair{display:flex;gap:16px}'
        'section{width:50%;min-width:0}img{width:100%;border:1px solid #aaa}</style>'
        '<h1>Original capture vs evaluated reference render</h1>'
        '<p>Original capture is from the dataset, not the current live website. Click images for full resolution.</p>'
        + ''.join(sections))
    (out / 'sources.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    print(f'Exported {len(selected)} distinct pages: {out / "index.html"}')
    if len(selected) < args.count:
        print(f'Only {len(selected)} of {args.count} requested pages were available.')


if __name__ == '__main__':
    main()

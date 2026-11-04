"""Export two original WebUI screenshots and their evaluated local renders."""
import argparse
import html
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Experiment root containing prepare/prepared.jsonl')
    parser.add_argument('--raw-root', required=True, help='Extracted original WebUI dataset root')
    parser.add_argument('--out', default='runs/webui-image-comparison')
    parser.add_argument('--count', type=int, default=2)
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        parser.error(f'Original WebUI folder not found: {raw_root}')
    manifests = sorted(Path(args.run).rglob('prepare/prepared.jsonl'))
    if not manifests:
        parser.error('No prepare/prepared.jsonl found under --run')
    # Index original files once, matching both unique page and exact viewport name.
    originals = {}
    for pattern in ('*-screenshot.webp', '*-screenshot.png'):
        for path in raw_root.rglob(pattern):
            key = (path.parent.name, path.name.rsplit('-screenshot.', 1)[0])
            originals.setdefault(key, []).append(path)
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
        parser.error('No matching original/rendered pairs. Check --raw-root contains original page-ID folders.')
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

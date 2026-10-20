"""Read-only audit of rejected policy renders; outputs only to a new directory."""
import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageChops, ImageDraw


def pixel_metrics(saved, rendered):
    if saved.size != rendered.size:
        return {'size_equal': False, 'saved_size': saved.size, 'rendered_size': rendered.size}
    delta = np.abs(np.asarray(saved.convert('RGB'), dtype=np.int16) -
                   np.asarray(rendered.convert('RGB'), dtype=np.int16))
    maximum = delta.max(axis=2)
    return {'size_equal': True, 'saved_size': saved.size, 'rendered_size': rendered.size,
            'pixels_equal': bool(not maximum.any()), 'mae_0_255': float(delta.mean()),
            'max_channel_difference': int(delta.max()),
            'changed_pixel_fraction': float((maximum > 0).mean()),
            'over_8_pixel_fraction': float((maximum > 8).mean())}


def overlay(image, elements):
    image = image.convert('RGB').copy()
    draw = ImageDraw.Draw(image)
    for item in elements:
        box = item.get('box', item.get('xyxy'))
        if box and len(box) == 4:
            draw.rectangle(box, outline='red', width=2)
    return image


def audit(row, out):
    from framediff.html_bridge import HtmlBrowser
    from framediff.visual import detector_labels
    report = {'id': row['id'], 'viewport': row['viewport'], 'renders': {}}
    with HtmlBrowser() as browser:
        report['chromium_version'] = browser.browser.version
        for prefix in ('target', 'current'):
            detail = {'html': row.get(prefix+'_html'), 'image': row.get(prefix+'_image')}
            report['renders'][prefix] = detail
            try:
                html = Path(row[prefix+'_html']).read_text()
                detail['html_sha256'] = hashlib.sha256(html.encode()).hexdigest()
                saved = Image.open(row[prefix+'_image']).convert('RGB')
                saved.save(out/f'{prefix}-saved.png')
                browser.load(html, row['viewport'])
                png, _ = browser.stable_capture()
                rendered = Image.open(io.BytesIO(png)).convert('RGB')
                rendered.save(out/f'{prefix}-rerender.png')
                detail.update(pixel_metrics(saved, rendered))
                labels = detector_labels(browser)['elements']
                detail['new_elements'] = labels
                detail['environment'] = browser.page.evaluate("""() => ({
                  userAgent:navigator.userAgent,devicePixelRatio:devicePixelRatio,
                  fonts:[...document.fonts].map(f=>({family:f.family,status:f.status})),
                  images:[...document.images].map(i=>({src:i.getAttribute('src'),
                    complete:i.complete,naturalWidth:i.naturalWidth,naturalHeight:i.naturalHeight})),
                  text:[...document.querySelectorAll('body *')].filter(e=>
                    [...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())).slice(0,100)
                    .map(e=>({id:e.getAttribute('data-fd-id'),tag:e.tagName,
                      text:e.textContent.slice(0,100),font:getComputedStyle(e).font}))
                })""")
                old = row.get(prefix+'_elements')
                detail['stored_elements'] = old
                detail['stored_elements_available'] = old is not None
                # Never draw current-state boxes as if they were target ground truth.
                old_boxes = row.get(prefix+'_boxes')
                measured = browser.tagged_boxes([n['id'] for n in row['current']['nodes'][1:]])
                detail['measured_owner_boxes'] = measured
                detail['stored_owner_boxes_available'] = old_boxes is not None
                if old_boxes is not None:
                    detail['owner_box_deltas'] = [
                        {'id': key, 'saved': box, 'measured': measured[key],
                         'max_abs_delta': max(abs(a-b) for a,b in zip(box, measured[key]))}
                        for key, box in old_boxes.items() if key in measured
                    ]
                old_view = overlay(saved, old or [])
                new_view = overlay(rendered, labels)
                old_view.save(out/f'{prefix}-saved-boxes.png')
                new_view.save(out/f'{prefix}-rerender-boxes.png')
                width, height = max(saved.width, rendered.width), max(saved.height, rendered.height)
                a = Image.new('RGB', (width, height), 'white'); a.paste(saved)
                b = Image.new('RGB', (width, height), 'white'); b.paste(rendered)
                diff = ImageChops.difference(a, b)
                diff.save(out/f'{prefix}-diff.png')
                diff.point(lambda v: min(255, v*8)).save(out/f'{prefix}-diff-x8.png')
                panel = Image.new('RGB', (width*3, height+32), 'white')
                for index, (image, title) in enumerate((
                    (old_view, 'Saved / stored boxes' if old is not None else 'Saved / NO stored element boxes'),
                    (new_view, 'Guava rerender / new DOM boxes'), (diff, 'Absolute RGB difference'))):
                    panel.paste(image, (width*index, 32))
                    ImageDraw.Draw(panel).text((width*index+5, 8), title, fill='black')
                panel.save(out/f'{prefix}-comparison.png')
            except Exception as exc:
                detail['error'] = f'{type(exc).__name__}: {exc}'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Original policy-val.jsonl')
    parser.add_argument('--rejections', required=True)
    parser.add_argument('--out', required=True, help='New audit directory')
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    wanted = []
    with open(args.rejections) as source:
        for line in source:
            item = json.loads(line)
            if item.get('reason') in ('screenshot_mismatch','geometry_mismatch','image_size_mismatch') and item.get('sample_id') not in wanted:
                wanted.append(item['sample_id'])
            if len(wanted) >= args.limit:
                break
    rows = {}
    with open(args.data) as source:
        for line in source:
            row = json.loads(line)
            if row['id'] in wanted:
                rows[row['id']] = row
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    summary = {'platform': platform.platform(), 'python': sys.version,
               'source': args.data, 'rejections': args.rejections,
               'note': 'Exact pixel differences do not establish the cause. Font CSS is not proof of the actual fallback font.',
               'samples': []}
    for index, sample in enumerate(wanted):
        folder = out/f'{index:02d}';folder.mkdir()
        try:
            report = audit(rows[sample], folder)
        except Exception as exc:
            report = {'id': sample, 'error': f'{type(exc).__name__}: {exc}'}
        (folder/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        summary['samples'].append({'id': sample, 'report': str(folder/'report.json'),
                                   'target': {k:v for k,v in report.get('renders', {}).get('target', {}).items()
                                              if k in ('size_equal','pixels_equal','mae_0_255','changed_pixel_fraction','over_8_pixel_fraction','error')},
                                   'error': report.get('error')})
        (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary['samples'][-1], ensure_ascii=False), flush=True)
    if not wanted:
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        raise SystemExit('No screenshot/geometry/image-size mismatch samples found')


if __name__ == '__main__':
    main()

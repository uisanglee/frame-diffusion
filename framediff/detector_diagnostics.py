"""Class-aware detector diagnostics; error categories are geometric heuristics, not causes."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw

from .visual import CLASSES
from .ir import write_json


def compare(pred, target):
    def boxes(items):
        return np.asarray([e['box'] for e in items], dtype=float).reshape(-1, 4)
    p, t = boxes(pred), boxes(target)
    lo = np.maximum(p[:, None, :2], t[None, :, :2])
    hi = np.minimum(p[:, None, 2:], t[None, :, 2:])
    intersection = np.maximum(hi-lo, 0).prod(axis=2)
    pa = np.maximum(p[:, 2:]-p[:, :2], 0).prod(axis=1)
    ta = np.maximum(t[:, 2:]-t[:, :2], 0).prod(axis=1)
    iou = intersection / np.maximum(pa[:, None]+ta[None, :]-intersection, 1e-12)
    same = np.asarray([e['label'] for e in pred])[:, None] == np.asarray([e['label'] for e in target])[None, :]
    valid = same & (iou >= .5)
    # Cardinality dominates the sum of all possible IoU tie breakers.
    ii, jj = linear_sum_assignment(-(valid*(min(len(pred), len(target))+1)+iou*valid))
    matches = {int(i): int(j) for i, j in zip(ii, jj) if valid[i, j]}
    hits = set(matches.values())
    errors = []
    for i in range(len(pred)):
        if i in matches:
            kind = 'true_positive'
        elif valid[i].any():
            kind = 'duplicate'
        elif ((~same[i]) & (iou[i] >= .5)).any():
            kind = 'class_confusion'
        elif (same[i] & (iou[i] >= .1) & (iou[i] < .5)).any():
            kind = 'localization'
        else:
            kind = 'unmatched'
        j = int(iou[i].argmax()) if len(target) else None
        errors.append({'index': i, 'kind': kind, 'matched_target': matches.get(i),
                       'nearest_target': j, 'nearest_iou': float(iou[i, j]) if j is not None else 0.})
    return {'predictions': errors, 'missed_targets': [j for j in range(len(target)) if j not in hits],
            'matches': matches}


def metrics(tp, predictions, targets, small_tp=0, small_n=0, n=0):
    return {'n': n, 'matched_iou50': tp, 'predictions': predictions, 'targets': targets,
            'false_positives': predictions-tp, 'false_negatives': targets-tp,
            'precision_iou50': tp/predictions if predictions else 0.,
            'recall_iou50': tp/targets if targets else None,
            'f1_iou50': 2*tp/(predictions+targets) if predictions+targets else None,
            'small_recall_iou50': small_tp/small_n if small_n else None, 'small_n': small_n}


class Aggregate:
    def __init__(self):
        self.total = np.zeros(6, dtype=int)
        self.by_class = defaultdict(lambda: np.zeros(6, dtype=int))
        self.by_view = defaultdict(lambda: np.zeros(6, dtype=int))
        self.by_device = defaultdict(lambda: np.zeros(6, dtype=int))
        self.errors = Counter()
        self.records = []

    def add(self, row, pred, target, detail):
        hits = set(detail['matches'].values())
        small = {i for i, e in enumerate(target) if (e['box'][2]-e['box'][0])*(e['box'][3]-e['box'][1]) < 32**2}
        counts = np.array([len(hits), len(pred), len(target), len(hits & small), len(small), 1])
        view = row.get('view') or row['id'].rsplit('/', 1)[-1].removesuffix('-screenshot.webp')
        device = 'desktop' if view.startswith('default_') else ('iPhone' if view.startswith('iPhone') else ('iPad' if view.startswith('iPad') else 'other'))
        self.total += counts
        self.by_view[view] += counts
        self.by_device[device] += counts
        for label, name in enumerate(CLASSES[1:], 1):
            ids = {j for j, e in enumerate(target) if e['label'] == label}
            self.by_class[name] += [len(hits & ids), sum(e['label'] == label for e in pred), len(ids),
                                    len(hits & ids & small), len(ids & small), 1]
        self.errors.update(e['kind'] for e in detail['predictions'] if e['kind'] != 'true_positive')
        self.records.append({'id': row['id'], 'page_id': row.get('page_id', row['id'].rsplit('/', 1)[0]),
                             'view': view, **metrics(*map(int, counts))})

    def report(self):
        pages = defaultdict(lambda: np.zeros(3, dtype=int))
        for row in self.records:
            pages[row['page_id']] += [row['matched_iou50'], row['predictions'], row['targets']]
        return {**metrics(*map(int, self.total)),
                'by_class': {k: metrics(*map(int, v)) for k, v in sorted(self.by_class.items())},
                'by_view': {k: metrics(*map(int, v)) for k, v in sorted(self.by_view.items())},
                'by_device': {k: metrics(*map(int, v)) for k, v in sorted(self.by_device.items())},
                'prediction_errors': dict(self.errors), 'unique_pages': len(pages),
                'macro_page': {'precision_iou50': float(np.mean([m/p if p else 0. for m,p,t in pages.values()])),
                               'recall_iou50': float(np.mean([m/t if t else 0. for m,p,t in pages.values()]))}}


def visualize(path, image_path, pred, target, detail):
    colors = {'true_positive': '#008000', 'duplicate': '#9933cc', 'class_confusion': '#e07000',
              'localization': '#0077cc', 'unmatched': '#dd2222'}
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    canvas = Image.new('RGB', (image.width*2, image.height+80), 'white')
    canvas.paste(image, (0, 80)); canvas.paste(image, (image.width, 80))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), 'Predictions: green=TP purple=duplicate orange=class blue=location red=unmatched', fill='black')
    draw.text((image.width+8, 25), 'Targets: green=matched red=missed', fill='black')
    for item in detail['predictions']:
        e = pred[item['index']]; x1,y1,x2,y2 = e['box']; color = colors[item['kind']]
        draw.rectangle((x1,y1+80,x2,y2+80), outline=color, width=2)
        draw.text((x1,y1+80), f"{item['index']} {CLASSES[e['label']]} {e['score']:.2f}", fill=color)
    for j, e in enumerate(target):
        x1,y1,x2,y2 = e['box']; color = '#dd2222' if j in detail['missed_targets'] else '#008000'
        draw.rectangle((x1+image.width,y1+80,x2+image.width,y2+80), outline=color, width=2)
        draw.text((x1+image.width,y1+80), f"{j} {CLASSES[e['label']]}", fill=color)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    write_json(path.with_suffix('.json'), {'predictions': pred, 'targets': target, 'diagnostics': detail})

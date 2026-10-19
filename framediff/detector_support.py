"""Detector target cache, CPU prefetch and evaluation diagnostics."""
import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
import multiprocessing as mp
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.util import Finalize
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
from .visual import CLASSES
from .ir import write_json


class TargetPredictionCache:
    def __init__(self,folder,checkpoint_sha,threshold,capacity=256):
        from importlib.metadata import version
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.identity={'contract':'frozen-target-prediction-v1','checkpoint':checkpoint_sha,
                       'threshold':threshold,'torchvision':version('torchvision')}
        self.threshold=threshold;self.capacity=capacity;self.memory=OrderedDict()
        self.hits=0;self.misses=0

    def remember(self,key,items):
        self.memory[key]=items;self.memory.move_to_end(key)
        while len(self.memory)>self.capacity:self.memory.popitem(last=False)

    def read(self,key):
        from .visual import CLASSES
        if key in self.memory:
            self.memory.move_to_end(key);return self.memory[key]
        try:
            saved=json.loads((self.folder/f'{key}.json').read_text())
            if saved['key']!=key:return None
            items=saved['items']
            if not isinstance(items,list):return None
            for item in items:
                if (len(item['box'])!=4 or not all(math.isfinite(v) for v in item['box'])
                    or item['label'] not in range(1,len(CLASSES)) or not math.isfinite(item['score'])):return None
            self.remember(key,items);return items
        except (OSError,ValueError,KeyError,TypeError):return None

    def predict(self,net,images):
        from .visual_train import detector_input,detect_batch
        keys=[];missing={};results={}
        for image in images:
            data=image if isinstance(image,bytes) else Path(image).read_bytes()
            key=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()+data).hexdigest()
            keys.append(key)
            if key in results or key in missing:continue
            items=self.read(key)
            if items is None:missing[key]=data
            else:results[key]=items;self.hits+=1
        if missing:
            predictions=detect_batch(net,[detector_input(data) for data in missing.values()],self.threshold)
            for key,items in zip(missing,predictions):
                self.misses+=1;results[key]=items;self.remember(key,items)
                temporary=None
                try:
                    with tempfile.NamedTemporaryFile(mode='w',dir=self.folder,delete=False) as file:
                        temporary=Path(file.name);json.dump({'key':key,'items':items},file,allow_nan=False)
                    os.replace(temporary,self.folder/f'{key}.json')
                finally:
                    if temporary is not None:temporary.unlink(missing_ok=True)
        return [results[key] for key in keys]


_cache=None
_pools=None


def initialize(pools,cache_dir):
    global _cache,_pools
    import torch
    from .browser import enable_worker_browser,close_worker_browser
    from .lazy_webui import TrainViewCache
    torch.set_num_threads(1)
    enable_worker_browser()
    _cache=TrainViewCache(cache_dir=cache_dir);_pools=pools
    Finalize(None,_cache.close,exitpriority=10)
    Finalize(None,close_worker_browser,exitpriority=9)


def materialize(task):
    rows,seeds=task;result=[];errors=[]
    from .lazy_webui import portable_images
    from .visual_train import sample_page_views
    for row,seed in zip(rows,seeds):
        rng=random.Random(seed)
        for _ in range(32):
            try:
                result.append(portable_images(_cache.get(row),('image',)) if row.get('lazy_contract') else row)
                break
            except ValueError as exc:
                errors.append({'id':row['id'],'error':str(exc)})
                row=sample_page_views(_pools,1,rng)[0]
        else:raise RuntimeError(f'32 lazy detector preparation failures: {errors}')
    return result,errors


class DetectorStream:
    def __init__(self,pools,rng,batch_size,cache_dir,log,workers=2,prefetch=4,timeout=900):
        if min(workers,prefetch,batch_size,timeout)<=0:raise ValueError('Invalid detector prefetch settings')
        self.pools=pools;self.rng=rng;self.planner=random.Random();self.planner.setstate(rng.getstate())
        self.batch_size=batch_size;self.cache_dir=str(cache_dir);self.log=Path(log)
        self.workers=workers;self.prefetch=prefetch;self.timeout=timeout;self.pending=deque();self.pool=None

    def __enter__(self):
        self.pool=ProcessPoolExecutor(self.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(self.pools,self.cache_dir))
        self.fill();return self

    def fill(self):
        from .visual_train import sample_page_views
        while len(self.pending)<self.prefetch:
            rows=sample_page_views(self.pools,self.batch_size,self.planner)
            seeds=[self.planner.getrandbits(64) for _ in rows]
            self.pending.append((self.pool.submit(materialize,(rows,seeds)),self.planner.getstate()))

    def take(self):
        future,state=self.pending[0]
        rows,errors=future.result(timeout=self.timeout)
        self.pending.popleft();self.rng.setstate(state)
        if errors:
            with self.log.open('a') as file:
                for error in errors:file.write(json.dumps(error)+'\n')
        self.fill();return rows

    def __exit__(self,*args):
        if self.pool is not None:
            from .preparation_workers import stop_executor
            stop_executor(self.pool,list((getattr(self.pool,'_processes',None) or {}).values()))
            self.pool=None


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


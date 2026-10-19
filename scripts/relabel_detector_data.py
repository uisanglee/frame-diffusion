"""Rebuild detector-only manifests from cached HTML, without policy normalization."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framediff.ir import read_jsonl
from framediff.visual import CONTRACT, VISUAL_GEOMETRY_CONTRACT, DETECTOR_LABEL_CONTRACT


def initialize():
    import torch
    from multiprocessing.util import Finalize
    from framediff.browser import enable_worker_browser, close_worker_browser
    torch.set_num_threads(1)
    enable_worker_browser()
    Finalize(None, close_worker_browser, exitpriority=9)


def process(task):
    row, folder = task
    from framediff.html_bridge import HtmlBrowser
    from framediff.lazy_webui import TrainViewCache
    from framediff.visual import detector_labels
    import PIL.Image
    key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    work = Path(folder) / 'assets' / key
    record = work / 'record.json'
    if record.exists():
        saved = json.loads(record.read_text())
        if Path(saved['image']).is_file() and Path(saved['audit_html']).is_file():
            return saved
    work.mkdir(parents=True, exist_ok=True)
    if row.get('lazy_contract'):
        source = dict(row)
        source['lazy_settings'] = {**row['lazy_settings'],
            'visual_geometry_contract': VISUAL_GEOMETRY_CONTRACT,
            'detector_label_contract': DETECTOR_LABEL_CONTRACT}
        cache = TrainViewCache()
        try:
            materialized = cache.get(source)
            image = Path(materialized['image'])
            html = image.with_suffix('.html')
            (work / 'image.png').write_bytes(image.read_bytes())
            (work / 'page.html').write_bytes(html.read_bytes())
            labels = materialized['elements']
            audit = materialized['label_audit']
            viewport = materialized['viewport']
        finally:
            cache.close()
        reused_pixels = False
    else:
        image = Path(row['image'])
        html = image.with_suffix('.html')
        if not html.is_file():
            raise FileNotFoundError(f'Cached sample HTML missing: {html}')
        with PIL.Image.open(image) as old:
            old = old.convert('RGB')
            viewport = row.get('viewport', list(old.size))
            with HtmlBrowser() as browser:
                browser.load(html.read_text(), viewport)
                png, _ = browser.stable_capture()
                generated = detector_labels(browser)
            import io
            with PIL.Image.open(io.BytesIO(png)) as new:
                reused_pixels = old.size == new.size and old.tobytes() == new.convert('RGB').tobytes()
        # Never attach fresh boxes to a different historical screenshot.
        (work / 'image.png').write_bytes(image.read_bytes() if reused_pixels else png)
        (work / 'page.html').write_bytes(html.read_bytes())
        labels, audit = generated['elements'], generated['label_audit']
    result = {k: row[k] for k in ('id', 'group', 'split', 'source_sha', 'page_id', 'view') if k in row}
    result.update(contract=CONTRACT, visual_geometry_contract=VISUAL_GEOMETRY_CONTRACT,
                  detector_label_contract=DETECTOR_LABEL_CONTRACT, viewport=viewport,
                  image=str(work / 'image.png'), audit_html=str(work / 'page.html'),
                  elements=labels, label_audit=audit, reused_pixels=reused_pixels)
    temp = record.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False))
    os.replace(temp, record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[7000, 56169])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    source, out = Path(args.data).resolve(), Path(args.out).resolve()
    if source == out:
        parser.error('Use a fresh output root, not the source dataset')
    selections, unique = {}, {}
    digests = {}
    for size in args.sizes:
        for split in ('train', 'val', 'test'):
            path = source / str(size) / 'rendered' / f'detector-{split}.jsonl'
            rows = list(read_jsonl(path))
            digests[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            keys = []
            for row in rows:
                key = (split, row['id'])
                if key in unique and unique[key] != row:
                    raise ValueError(f'Nested scale row differs: {key}')
                unique[key] = row
                keys.append(key)
            selections[size, split] = keys
    out.mkdir(parents=True, exist_ok=True)
    config = {'inputs': digests, 'label_contract': DETECTOR_LABEL_CONTRACT,
              'code': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [
                  ('script', Path(__file__)),
                  ('visual', Path(__file__).resolve().parents[1] / 'framediff/visual.py')]}}
    config_path = out / 'relabel-config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Source/code changed; choose a new output root')
    config_path.write_text(json.dumps(config, indent=2))
    (out / 'READY').unlink(missing_ok=True)
    completed = {}
    keys = list(unique)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        try:
            # Bound pending jobs; do not queue hundreds of thousands of futures.
            from collections import deque
            pending = deque()
            cursor = 0
            while cursor < len(keys) or pending:
                while cursor < len(keys) and len(pending) < args.workers * 2:
                    key = keys[cursor]; cursor += 1
                    pending.append((key, pool.submit(process, (unique[key], str(out)))))
                key, future = pending.popleft()
                completed[key] = future.result()
                if len(completed) % 50 == 0 or len(completed) == len(keys):
                    print(f'Relabel [{len(completed)}/{len(keys)}]', flush=True)
        except BaseException:
            from framediff.preparation_workers import stop_executor
            stop_executor(pool, list((getattr(pool, '_processes', None) or {}).values()))
            raise
    for (size, split), selected in selections.items():
        folder = out / str(size) / 'rendered'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'detector-{split}.jsonl'
        temp = path.with_suffix('.tmp')
        with temp.open('w') as file:
            for key in selected:
                file.write(json.dumps(completed[key], ensure_ascii=False) + '\n')
        os.replace(temp, path)
    (out / 'READY').write_text('Detector relabeling complete\n')
    print(f'Complete: {out}. Original policy/raw assets have NOT been deleted.', flush=True)


if __name__ == '__main__':
    main()

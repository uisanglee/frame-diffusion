"""Paired executor ablation, with both outputs judged in Chromium."""
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .browser import Browser
from .ir import execute, read_jsonl, write_json, write_jsonl
from .metrics import box_metrics, hungarian_box_metrics
from .model import load_model
from .search import Executor, repair
from .train import select_device


class RenderBackend:
    def __init__(self, browser, raster):
        self.browser = browser
        self.raster = raster
        self.screenshots = 0

    def render(self, tree, viewport):
        boxes = self.browser.render(tree, viewport)
        if self.raster:
            self.browser.page.screenshot()  # Include rasterization/PNG encoding; no disk IO.
            self.screenshots += 1
        return boxes


def truth_for(record):
    if record.get('target_kind') == 'predicted_frames' and not any(
        key in record for key in ('evaluation_observations', 'reference_observations')
    ):
        raise ValueError('Predicted frames require independent evaluation truth')
    # Older WebUI files stored real observations without connecting them to evaluation.
    if 'evaluation_observations' not in record and 'original_target' in record:
        if len(record['observations']) != 1:
            raise ValueError('Legacy original_target requires one viewport')
        return [{'viewport': record['observations'][0]['viewport'], 'target': record['original_target']}]
    return record.get('evaluation_observations', record.get('reference_observations', record['observations']))


def score(record, tree, frames, truth):
    root = tree['nodes'][0]['id']
    matching = record.get('evaluation_matching', 'identity')
    if matching not in ('identity', 'hungarian_geometry'):
        raise ValueError(f'Unknown matching: {matching}')
    metrics = []
    for boxes, obs in zip(frames, truth):
        if matching == 'identity':
            metrics.append(box_metrics(boxes, obs['target'], obs['viewport'], (root,)))
        else:
            metrics.append(hungarian_box_metrics(boxes, obs['target'], obs['viewport'],
                                                (root,), record.get('reference_root_ids', [])))
    return statistics.mean(m['box_iou'] for m in metrics)


def summarize(rows, seed):
    pairs = []
    for rid in sorted({r['id'] for r in rows}):
        page = [r for r in rows if r['id'] == rid]
        methods = {m: [r for r in page if r['backend'] == m] for m in {r['backend'] for r in page}}
        proxy = methods['proxy']
        for backend, measured in methods.items():
            if backend == 'proxy':
                continue
            def mean(rs, key):
                return statistics.mean(r[key] for r in rs)
            pairs.append({'id': rid, 'group': page[0]['group'], 'backend': backend,
                          'proxy_repair_seconds': mean(proxy, 'repair_seconds'),
                          'browser_repair_seconds': mean(measured, 'repair_seconds'),
                          'proxy_pipeline_seconds': mean(proxy, 'pipeline_seconds'),
                          'browser_pipeline_seconds': mean(measured, 'pipeline_seconds'),
                          'proxy_browser_iou': mean(proxy, 'browser_iou'),
                          'browser_browser_iou': mean(measured, 'browser_iou'),
                          'iou_loss_pp': 100*(mean(measured, 'browser_iou')-mean(proxy, 'browser_iou')),
                          'browser_calls_saved': mean(measured, 'search_browser_calls')-mean(proxy, 'search_browser_calls')})
    summaries = []
    rng = np.random.default_rng(seed)
    for backend in sorted({p['backend'] for p in pairs}):
        ps = [p for p in pairs if p['backend'] == backend]
        def mean(key):
            return statistics.mean(p[key] for p in ps)
        groups = sorted({p['group'] for p in ps})
        clusters = [[p for p in ps if p['group'] == g] for g in groups]
        boot = []
        for _ in range(1000):
            sampled = [p for i in rng.integers(0, len(groups), len(groups)) for p in clusters[i]]
            boot.append(statistics.mean(p['iou_loss_pp'] for p in sampled))
        summaries.append({'backend': backend, 'pages': len(ps),
                          'repair_speedup': mean('browser_repair_seconds')/max(mean('proxy_repair_seconds'), 1e-12),
                          'pipeline_speedup': mean('browser_pipeline_seconds')/max(mean('proxy_pipeline_seconds'), 1e-12),
                          'proxy_seconds': mean('proxy_repair_seconds'), 'browser_seconds': mean('browser_repair_seconds'),
                          'proxy_browser_iou': mean('proxy_browser_iou'), 'browser_browser_iou': mean('browser_browser_iou'),
                          'iou_loss_pp': mean('iou_loss_pp'),
                          'iou_loss_pp_ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
                          'browser_calls_saved': mean('browser_calls_saved')})
    return pairs, summaries


def compare_renderers(args):
    records = list(read_jsonl(args.data))
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Dataset must be nonempty with unique IDs')
    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.limit:
        records = records[:args.limit]
    device = select_device(args.device)
    if device == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint['objective'] != 'edits':
        raise ValueError('An edit checkpoint is required')
    if set(checkpoint.get('training_groups', [])) & {r['group'] for r in records}:
        raise ValueError('Evaluation groups overlap checkpoint training groups')
    for record in records:
        if [o['viewport'] for o in record['observations']] != [o['viewport'] for o in truth_for(record)]:
            raise ValueError('Input/truth viewport mismatch')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backends = ['proxy', 'browser'] + (['browser-raster'] if args.raster else [])
    rows = []

    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    with Browser() as browser:
        def run(record, backend, steps):
            renderer = RenderBackend(browser, backend == 'browser-raster')
            executor = Executor(None if backend == 'proxy' else renderer)
            calls = browser.executions
            sync(); start = time.perf_counter()
            result, trace = repair(record['current'], record['observations'], model=model,
                                   steps=steps, beam=args.beam, proposals_per_state=args.topk,
                                   budget=args.budget, executor=executor, seed=args.seed)
            sync(); elapsed = time.perf_counter()-start
            return result, trace, elapsed, browser.executions-calls, renderer.screenshots

        for _ in range(args.warmup):
            for backend in backends:
                run(records[0], backend, min(2, args.steps))
        for index, record in enumerate(records):
            truth = truth_for(record)
            for repeat in range(args.repeats):
                order = backends.copy(); rng.shuffle(order)
                for backend in order:
                    result, trace, elapsed, calls, screenshots = run(record, backend, args.steps)
                    start = time.perf_counter()
                    actual = [browser.render(result, o['viewport']) for o in truth]
                    verify_seconds = time.perf_counter()-start
                    proxy = [execute(result, o['viewport']) for o in truth]
                    error = max(abs(b[k][j]-a[k][j]) for a,b in zip(actual,proxy) for k in a for j in range(4))
                    row = {'id': record['id'], 'group': record['group'], 'repeat': repeat, 'backend': backend,
                           'repair_seconds': elapsed, 'verify_seconds': verify_seconds,
                           'pipeline_seconds': elapsed+verify_seconds+record.get('upstream_seconds', 0),
                           'search_browser_calls': calls, 'verification_browser_calls': len(truth),
                           'search_screenshots': screenshots, 'executor_calls': trace['executions'],
                           'candidates': trace['candidates'], 'browser_iou': score(record, result, actual, truth),
                           'proxy_iou': score(record, result, proxy, truth), 'parity_max_error_px': error}
                    rows.append(row)
                    write_json(out/'predictions'/f'{index}-{repeat}-{backend}.json', {'result': result, 'trace': trace})
                    write_jsonl(out/'results.jsonl', rows)
                    print(f'[{index+1}/{len(records)}] repeat={repeat+1} {backend}: '
                          f'{elapsed:.3f}s browser-IoU={row["browser_iou"]:.4f} calls={calls}', flush=True)
        version = browser.browser.version
    pairs, summaries = summarize(rows, args.seed)
    write_jsonl(out/'paired.jsonl', pairs)
    write_json(out/'summary.json', summaries)
    write_json(out/'manifest.json', {**vars(args), 'device_actual': device, 'chromium': version,
                                   'torch': str(torch.__version__), 'sample_ids': [r['id'] for r in records],
                                   'protocol': 'Equal candidate budget; fresh cache per run; randomized order; both outputs judged in Chromium.'})
    lines = ['# Paired renderer comparison', '',
             '| Browser backend | Pages | Proxy s | Browser s | Repair speedup | Pipeline speedup | Proxy IoU | Browser IoU | Loss pp [95% CI] | Search calls saved |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for s in summaries:
        lo,hi = s['iou_loss_pp_ci95']
        lines.append(f'| {s["backend"]} | {s["pages"]} | {s["proxy_seconds"]:.3f} | {s["browser_seconds"]:.3f} | '
                     f'{s["repair_speedup"]:.2f}x | {s["pipeline_speedup"]:.2f}x | {s["proxy_browser_iou"]:.4f} | '
                     f'{s["browser_browser_iou"]:.4f} | {s["iou_loss_pp"]:.3f} [{lo:.3f}, {hi:.3f}] | {s["browser_calls_saved"]:.1f} |')
    lines += ['', 'Positive loss pp means proxy loses accuracy; negative means proxy improves it.',
              'CI resamples page/domain groups after averaging repeats. Warmup and startup are excluded.',
              'Raster includes PNG generation per uncached geometry evaluation; no VLM feedback or disk IO.',
              'Both backends render the same supported LayoutIR. This does not measure arbitrary CSS approximation.',
              'Pipeline time includes final verification and recorded upstream VLM time, when available.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')

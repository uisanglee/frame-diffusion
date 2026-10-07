"""Page-weighted final repair evaluation. No model selection or oracle feedback."""
import argparse
from collections import defaultdict
import csv
import math
import re
from pathlib import Path

import numpy as np

from .ir import read_jsonl, write_json


QUALITY = {
    'dom_box_iou': 1, 'webui_box_iou': 1,
    'dom_center_error': -1, 'webui_center_error': -1,
    'dom_size_error': -1, 'webui_size_error': -1,
    'pixel_mae': -1,
    'official_block': 1, 'official_position': 1, 'official_clip': 1,
}
COST = ('pipeline_seconds', 'repair_seconds', 'repair_browser_executions',
        'repair_browser_screenshots', 'repair_actions', 'repair_vlm_calls',
        'repair_input_tokens', 'repair_output_tokens')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def estimate(values, seed=42):
    """Bootstrap pages, after averaging repeats within each page."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n_pages': 0, 'mean': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    # Bounded temporary memory for large corpora.
    means = []
    for _ in range(20):
        means.extend(values[rng.integers(len(values), size=(100, len(values)))].mean(1))
    return {'n_pages': len(values), 'mean': float(values.mean()),
            'ci95': np.percentile(means, [2.5, 97.5]).tolist()}


def report(rows, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        # Older evaluators only wrote page_id for repaired visual methods;
        # the shared initial row still has an explicit /repeat-N suffix.
        page_id = row.get('page_id', re.sub(r'/repeat-\d+$', '', row['id']))
        grouped[(page_id, row['method'])].append(row)
    pages = {}
    for key, trials in grouped.items():
        normalized = []
        for row in trials:
            item = dict(row)
            timing = row.get('visual_timing', {})
            for source, target in [('seconds', 'repair_seconds'), ('browser_executions', 'repair_browser_executions'),
                                   ('browser_screenshots', 'repair_browser_screenshots'), ('actions', 'repair_actions')]:
                if finite(timing.get(source)): item[target] = timing[source]
            item['pipeline_failure_rate'] = float(bool(row.get('failed')))
            item['evaluation_failure_rate'] = float(bool(row.get('evaluation_error')))
            normalized.append(item)
        pages[key] = {}
        for metric in (*QUALITY, *COST, 'pipeline_failure_rate', 'evaluation_failure_rate'):
            values = [r[metric] for r in normalized if finite(r.get(metric))]
            # Require complete repeats; otherwise a method could benefit from
            # excluding its failed measurements. Missing never becomes zero.
            if len(values) == len(trials): pages[key][metric] = float(np.mean(values))
    methods = sorted({key[1] for key in pages})
    summaries = []
    for method in methods:
        subset = [v for (page, m), v in pages.items() if m == method]
        metrics = {k: estimate([p[k] for p in subset if k in p])
                   for k in (*QUALITY, *COST, 'pipeline_failure_rate', 'evaluation_failure_rate')}
        summaries.append({'method': method, 'n_pages': len(subset),
                          'n_trials': sum(len(v) for (p, m), v in grouped.items() if m == method), 'metrics': metrics})
    paired = []
    # Include the primary comparison as well as improvement over initial.
    for baseline in ('initial', 'screenshot-policy'):
        for method in methods:
            if method == baseline or method == 'initial': continue
            ids = sorted(p for p, m in pages if m == method and (p, baseline) in pages)
            for metric, direction in {**QUALITY, **{k: -1 for k in COST}}.items():
                gains = [direction * (pages[p, method][metric] - pages[p, baseline][metric])
                         for p in ids if metric in pages[p, method] and metric in pages[p, baseline]]
                if not gains: continue
                paired.append({'method': method, 'baseline': baseline, 'metric': metric,
                               'positive_is_better': True, **estimate(gains),
                               'improved_page_rate': sum(g > 1e-8 for g in gains) / len(gains),
                               'worsened_page_rate': sum(g < -1e-8 for g in gains) / len(gains),
                               'unchanged_page_rate': sum(abs(g) <= 1e-8 for g in gains) / len(gains)})
    result = {'protocol': 'page-weighted-repair-v1', 'methods': summaries, 'paired': paired,
              'notes': ['Repeats averaged within page; 2000 paired page bootstrap resamples, seed 42.',
                        'Failed pipelines retain fallback HTML. Missing evaluations excluded with metric-specific n.',
                        'Geometry uses diagnostic Hungarian matching, not official Design2Code scoring.',
                        'Repair time includes target preprocessing; excludes model loading and final evaluation.',
                        'Page bootstrap does not account for correlation between different pages of the same domain.']}
    write_json(out / 'paper-summary.json', result)
    with (out / 'paper-summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['method', 'metric', 'n_pages', 'mean', 'ci95_low', 'ci95_high'])
        for method in summaries:
            for metric, value in method['metrics'].items():
                if value['n_pages']:
                    writer.writerow([method['method'], metric, value['n_pages'], value['mean'], *value['ci95']])
    def cell(metric):
        if metric['mean'] is None: return 'N/A'
        lo, hi = metric['ci95']
        return f"{metric['mean']:.4f} [{lo:.4f}, {hi:.4f}] (n={metric['n_pages']})"
    lines = ['# Final repair evaluation', '', 'Mean [95% page-bootstrap CI]. n counts pages, not repeats.', '']
    for title, keys in [('Geometry and appearance', list(QUALITY)), ('Cost and failures', [*COST, 'pipeline_failure_rate', 'evaluation_failure_rate'])]:
        keys = [k for k in keys if any(s['metrics'][k]['n_pages'] for s in summaries)]
        lines += [f'## {title}', '', '| Method | ' + ' | '.join(keys) + ' |', '|---|' + '---|' * len(keys)]
        for item in summaries:
            lines.append('| ' + item['method'] + ' | ' + ' | '.join(cell(item['metrics'][k]) for k in keys) + ' |')
        lines.append('')
    lines += ['## Paired gains', '', 'Positive gain means better; rates compare page averages (tolerance 1e-8).', '',
              '| Method | Baseline | Metric | Gain [95% CI] | Improved | Worsened |', '|---|---|---|---|---:|---:|']
    for item in paired:
        lines.append(f"| {item['method']} | {item['baseline']} | {item['metric']} | {cell(item)} | {item['improved_page_rate']:.1%} | {item['worsened_page_rate']:.1%} |")
    lines += ['', *result['notes']]
    (out / 'paper-report.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild final paper tables from existing metrics.jsonl; no rendering')
    parser.add_argument('--metrics', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report(list(read_jsonl(args.metrics)), args.out)

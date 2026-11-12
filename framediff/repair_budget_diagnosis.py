"""Paired target-source/edit-budget diagnosis, using saved initials only."""
import argparse
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from .ir import read_jsonl, write_json, write_jsonl


def reference_elements(record, browser, cache):
    from PIL import Image
    import numpy as np
    from .html_bridge import embed_placeholder
    from .visual import detector_labels
    from .web_experiment import digest
    if record.get('target_image_source') != 'reference-render':
        raise ValueError('Oracle requires the exact locally rendered reference target')
    source = Path(record['html'])
    placeholder = source.parent / 'rick.jpg'
    html = embed_placeholder(source.read_text(), placeholder if placeholder.is_file() else None)
    identity = dict(html=html, image=digest(record['screenshot']), viewport=record['viewport'],
                    collector=digest(Path(__file__).with_name('visual.py')),
                    browser=digest(Path(__file__).with_name('html_bridge.py')))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache) / (key + '.json')
    if path.exists():
        return json.loads(path.read_text())['elements']
    browser.load(html, record['viewport'])
    actual = Image.open(io.BytesIO(browser.page.screenshot(animations='disabled'))).convert('RGB')
    expected = Image.open(record['screenshot']).convert('RGB')
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError('Oracle reference rerender differs from saved target; refusing mismatched labels')
    labels = detector_labels(browser)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, labels)
    return labels['elements']


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['model'], row['seed'], row['budget'], row['condition']), []).append(row)
    result = []
    for key, values in sorted(groups.items()):
        eligible = [v for v in values if not v['initial_failed']]
        valid = [v for v in eligible if all(isinstance(v.get(k), (int, float)) and math.isfinite(v[k])
                     for k in ('position', 'clip', 'iou', 'initial_iou'))]
        summary = dict(zip(('dataset','model','seed','budget','condition'), key))
        summary.update(attempted=len(values), eligible=len(eligible), scored=len(valid),
                       missing=len(eligible)-len(valid), failed=sum(v['failed'] for v in eligible))
        # Never silently average a different cohort when a metric or oracle render fails.
        complete = len(valid) == len(eligible) and bool(valid)
        for field in ('position','clip','iou','delta_iou','actions'):
            summary[field] = statistics.mean(v[field] for v in valid) if complete else None
        summary['improved_percent'] = 100*sum(v['delta_iou']>0 for v in valid)/len(valid) if complete else None
        result.append(summary)
    return result


def saved_baseline(seed_root, prepared):
    """Reuse original evaluated initials only when their HTML and target still agree."""
    from .web_experiment import digest
    from .evaluation_recovery import reusable_metric
    results = seed_root/'repair/results.jsonl'
    metrics = seed_root/'evaluation/metrics.jsonl'
    if not results.is_file() or not metrics.is_file():
        return None
    records = {r['id']:r for r in read_jsonl(results)}
    scores = {(r['id'],r['method']):r for r in read_jsonl(metrics)}
    reused=[]
    for source in prepared:
        identity=source['id']+'/repeat-1'
        previous=records.get(identity)
        score=scores.get((identity,'initial'))
        if previous is None or score is None:
            return None
        if previous['viewport']!=source['viewport']:
            return None
        for old,new in ((previous['screenshot'],source['screenshot']),
                        (previous['methods']['initial']['html'],source['methods']['initial']['html'])):
            if digest(old)!=digest(new):return None
        if not reusable_metric(score,initial_failed=bool(source['methods']['initial'].get('failed')),official=True):
            return None
        reused.append(dict(score,id=source['id'],diagnostic_reused_from=str(metrics)))
    return reused


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--abstract-checkpoint', required=True)
    p.add_argument('--detector-checkpoint', required=True)
    p.add_argument('--official-repo', required=True)
    p.add_argument('--datasets', nargs='+', default=['webui','design2code','design2code-hard'])
    p.add_argument('--models', nargs='+', default=['llava-next-7b','glm4v-9b-thinking','websight-8b','ui2coden'])
    p.add_argument('--seed', default='42')
    p.add_argument('--device', default='cuda')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if root == out or root in out.parents:
        raise ValueError('Use a separate sibling output directory to preserve the original experiment')
    all_rows = []
    def run(*command):
        subprocess.run([sys.executable, '-m', 'framediff', *map(str, command)], check=True)
    for dataset in args.datasets:
        for model in args.models:
            source = root/dataset/model/'seeds'/args.seed/'prepare/prepared.jsonl'
            if not source.is_file():
                raise FileNotFoundError(source)
            work = out/dataset/model/'seeds'/args.seed
            work.mkdir(parents=True, exist_ok=True)
            # Same exact pages, viewport, initial HTML and policy conditioning; no generation.
            prepared = list(read_jsonl(source))
            manifest = work/'prepared.jsonl'
            rows = []
            for row in prepared:
                row = dict(row, methods={'initial':row['methods']['initial']})
                rows.append(row)
            if not args.report_only:
                write_jsonl(manifest, rows)
            baseline = work/'budget-0/evaluation'
            if not args.report_only:
                reused=saved_baseline(source.parent.parent,rows)
                if reused is not None:
                    baseline.mkdir(parents=True,exist_ok=True)
                    write_jsonl(baseline/'metrics.jsonl',reused)
                    print(f'Reused {len(reused)} existing Initial evaluations: {source.parent.parent}',flush=True)
                else:
                    run('web-evaluate','--data',manifest,'--out',baseline,'--official-repo',args.official_repo,'--resume')
            initial_metrics = {v['id']:v for v in read_jsonl(baseline/'metrics.jsonl') if v['method']=='initial'}
            for budget in (0,1,5,10,20):
                print(f'=== {dataset} | {model} | budget={budget} | detector vs reference DOM ===', flush=True)
                trial = work/f'budget-{budget}'
                if budget and not args.report_only:
                    run('visual-evaluate','--data',manifest,'--out',trial/'repair',
                        '--abstract-checkpoint',args.abstract_checkpoint,'--detector-checkpoint',args.detector_checkpoint,
                        '--abstract-only','--reference-dom-ablation','--steps',budget,'--repeats',1,
                        '--seed',args.seed,'--pixel-goal-fraction',-1,'--device',args.device,'--resume')
                    run('web-evaluate','--data',trial/'repair/results.jsonl','--out',trial/'evaluation',
                        '--official-repo',args.official_repo,'--resume')
                metrics = {(v['id'],v['method']):v for v in read_jsonl(trial/'evaluation/metrics.jsonl')}
                timing = {(v['id'],v['method']):v for v in read_jsonl(trial/'repair/timings.jsonl')} if budget else {}
                for row in rows:
                    initial = initial_metrics.get(row['id'], {})
                    for condition, method in [('detector','abstract-policy'),('reference-dom','abstract-oracle')]:
                        metric = metrics.get((row['id']+'/repeat-1',method),{}) if budget else initial
                        t = timing.get((row['id'],method),{})
                        base_iou = initial.get('dom_box_iou',initial.get('webui_box_iou'))
                        iou = metric.get('dom_box_iou',metric.get('webui_box_iou'))
                        all_rows.append(dict(dataset=dataset,model=model,seed=args.seed,id=row['id'],budget=budget,
                            condition=condition,initial_failed=bool(row['methods']['initial'].get('failed')),
                            failed=bool(t.get('failed',False)),error=t.get('error') or metric.get('error'),
                            position=metric.get('official_position'),clip=metric.get('official_clip'),iou=iou,
                            initial_iou=base_iou,delta_iou=iou-base_iou if iou is not None and base_iou is not None else None,
                            actions=t.get('actions',0),stop_reason=t.get('stop_reason'),
                            oracle_error=condition=='reference-dom' and bool(t.get('failed'))))
    # Oracle failures must not masquerade as valid oracle comparisons via fallback scores.
    for row in all_rows:
        if row['oracle_error']:
            row['position']=None
    write_jsonl(out/'per-page.jsonl',all_rows)
    summary=summarize(all_rows)
    write_json(out/'summary.json',summary)
    lines=['# Target abstraction and edit-budget diagnosis','',
           'Same Stage2 weights and saved initials. Budgets count policy attempts; Actions counts accepted edits. No pixel-goal early stop. Missing scores invalidate the cell; initial failures are excluded equally.','',
           '| Dataset | Model | Seed | Budget | Target | N / Eligible | Position | CLIP | IoU | Delta IoU | Improved % | Actions | Failed |',
           '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        f=lambda k: '--' if r[k] is None else f'{r[k]:.4f}'
        lines.append(f"| {r['dataset']} | {r['model']} | {r['seed']} | {r['budget']} | {r['condition']} | {r['scored']} / {r['eligible']} | "+' | '.join(f(k) for k in ('position','clip','iou','delta_iou','improved_percent','actions'))+f" | {r['failed']} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(out/'README.md')

if __name__ == '__main__':
    main()

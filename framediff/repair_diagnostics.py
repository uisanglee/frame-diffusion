"""Post-evaluation DOM/detector disagreement, never used to guide repair."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .ir import read_jsonl, write_json, write_jsonl

CONTRACT = 'repair-disagreement-iou50-v1'


def comparison(pred, target):
    from .detector_support import compare
    detail = compare(pred, target)
    tp = len(detail['matches'])
    return dict(predictions=len(pred), targets=len(target), matched=tp,
                unmatched_predictions=len(pred)-tp, unmatched_targets=len(target)-tp,
                precision=tp/len(pred) if pred else None,
                recall=tp/len(target) if target else None, detail=detail)


def compare_three(reference, detector, initial):
    pairs = {'detector_vs_reference': comparison(detector, reference)}
    if initial is not None:
        pairs.update(initial_vs_reference=comparison(initial, reference),
                     initial_vs_detector=comparison(initial, detector))
    return pairs


def outcomes(metrics):
    """Keep repetitions and methods separate; absent scores are not failures."""
    groups = defaultdict(list)
    for row in metrics:
        groups[row['id']].append(row)
    result = []
    for identity, rows in groups.items():
        initial = next((r.get('dom_box_iou') for r in rows if r['method']=='initial'), None)
        for row in rows:
            final = row.get('dom_box_iou')
            delta = final-initial if initial is not None and final is not None else None
            result.append(dict(id=identity, method=row['method'], initial_box_iou=initial,
                               final_box_iou=final, delta_box_iou=delta,
                               improved=delta>0 if delta is not None else None,
                               worsened=delta<0 if delta is not None else None,
                               failed=row.get('failed'), error=row.get('error')))
    return result


def summarize(records):
    totals = defaultdict(lambda: dict(pages=0, predictions=0, targets=0, matched=0))
    seen = set()
    for row in records:
        for pair, value in row.get('comparisons', {}).items():
            # One target observation, regardless of number of initial VLMs/seeds.
            key = (pair, row['target_key'])
            if pair=='detector_vs_reference' and key in seen:
                continue
            seen.add(key)
            total = totals[pair];total['pages'] += 1
            for field in ('predictions','targets','matched'):total[field] += value[field]
    for total in totals.values():
        total['precision'] = total['matched']/total['predictions'] if total['predictions'] else None
        total['recall'] = total['matched']/total['targets'] if total['targets'] else None
        total['unmatched_predictions'] = total['predictions']-total['matched']
        total['unmatched_targets'] = total['targets']-total['matched']
    return dict(totals)


def associations(records):
    """Stratified descriptive results; one observation per method/repetition."""
    groups=defaultdict(list)
    for row in records:
        if row['initial_failed'] or row['errors']:
            continue
        for pair,value in row['comparisons'].items():
            category=('both' if value['unmatched_predictions'] and value['unmatched_targets'] else
                      'unmatched_predictions_only' if value['unmatched_predictions'] else
                      'unmatched_targets_only' if value['unmatched_targets'] else 'fully_matched')
            for outcome in row['repair_outcomes']:
                if outcome['method']=='initial' or outcome['delta_box_iou'] is None:
                    continue
                groups[(row['model'],outcome['method'],pair,category)].append(outcome)
    return [dict(model=key[0],method=key[1],comparison=key[2],disagreement=key[3],
                 n=len(values),mean_delta_box_iou=sum(v['delta_box_iou'] for v in values)/len(values),
                 improved_percent=100*sum(v['improved'] for v in values)/len(values),
                 worsened_percent=100*sum(v['worsened'] for v in values)/len(values))
            for key,values in sorted(groups.items())]


def run(args):
    import numpy as np
    from PIL import Image
    from .html_bridge import HtmlBrowser, embed_placeholder
    from .visual import detector_labels, VISUAL_GEOMETRY_CONTRACT
    from .visual_train import load_detector, detect, detector_input
    from .detector_support import Aggregate, compare
    from .web_experiment import digest
    root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    checkpoint_sha=digest(args.detector_checkpoint)
    net,ck=load_detector(args.detector_checkpoint,args.device)
    if ck.get('config',{}).get('visual_geometry_contract') != VISUAL_GEOMETRY_CONTRACT:
        raise ValueError('Diagnostics require the full visible DOM detector contract')
    records=[];bucket=Aggregate();seen_targets=set()
    manifests=sorted(root.glob('*/seeds/*/prepare/prepared.jsonl'))
    if not manifests:raise ValueError(f'No prepared manifests under {root}')
    with HtmlBrowser() as browser:
        for manifest in manifests:
            seed_root=manifest.parent.parent
            metric_rows=[]
            for path in (seed_root/'evaluation/metrics.jsonl',seed_root/'ui2coden/evaluation/metrics.jsonl'):
                if path.is_file():metric_rows.extend(read_jsonl(path))
            metric_index=defaultdict(list)
            for metric in metric_rows:
                metric_index[metric['id'].split('/repeat-')[0]].append(metric)
            for source in read_jsonl(manifest):
                row=dict(id=source['id'],dataset=args.dataset,model=seed_root.parent.parent.name,
                         seed=seed_root.name,initial_failed=bool(source['methods']['initial'].get('failed') or source.get('errors',{}).get('initial')),
                         comparisons={},errors={})
                try:
                    if source.get('group') in ck.get('training_groups',[]) or source.get('source_sha') in ck.get('training_hashes',[]):
                        raise ValueError('Detector evaluation overlaps training corpus')
                    if source.get('target_image_source')!='reference-render':
                        raise ValueError('Diagnostics require locally rendered reference targets')
                    viewport=source['viewport'];html_path=Path(source['html'])
                    placeholder=html_path.parent/'rick.jpg'
                    placeholder=placeholder if placeholder.is_file() else None
                    ref_html=embed_placeholder(html_path.read_text(),placeholder)
                    identity=dict(contract=CONTRACT,html=ref_html,image=digest(source['screenshot']),
                                  viewport=viewport,checkpoint=checkpoint_sha,threshold=args.threshold,
                                  collector=digest(Path(__file__).with_name('visual.py')),
                                  browser=digest(Path(__file__).with_name('html_bridge.py')))
                    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                    row['target_key']=key;path=cache/f'{key}.json'
                    if path.exists():target=json.loads(path.read_text())
                    else:
                        browser.load(ref_html,viewport)
                        # Legacy reference caches do not contain the detector label collector.
                        # Verify the reloaded reference still matches the exact evaluated PNG.
                        import io
                        png=browser.page.screenshot(animations='disabled')
                        with Image.open(io.BytesIO(png)) as actual, Image.open(source['screenshot']) as expected:
                            if not np.array_equal(np.asarray(actual.convert('RGB')),np.asarray(expected.convert('RGB'))):
                                raise ValueError('Reference rerender differs from target PNG; diagnostics not scored')
                        target=detector_labels(browser)
                        target['detector']=detect(net,detector_input(source['screenshot']),args.threshold)
                        write_json(path,target)
                    if key not in seen_targets:
                        bucket.add(source,target['detector'],target['elements'],compare(target['detector'],target['elements']))
                        seen_targets.add(key)
                    initial=None
                    if not row['initial_failed']:
                        initial_html=embed_placeholder(Path(source['methods']['initial']['html']).read_text(),placeholder)
                        initial_key=hashlib.sha256(json.dumps([CONTRACT,initial_html,viewport,identity['collector'],identity['browser']],sort_keys=True).encode()).hexdigest()
                        initial_path=cache/f'initial-{initial_key}.json'
                        if initial_path.exists():observed=json.loads(initial_path.read_text())
                        else:
                            browser.load(initial_html,viewport);observed=detector_labels(browser)
                            write_json(initial_path,observed)
                        initial=observed['elements'];row['initial_label_audit']=observed['label_audit']
                    row['comparisons']=compare_three(target['elements'],target['detector'],initial)
                    row['reference_label_audit']=target['label_audit']
                    row['elements']=dict(reference=target['elements'],detector=target['detector'],initial=initial)
                except Exception as exc:
                    from .runtime_errors import raise_if_fatal
                    raise_if_fatal(exc)
                    row['errors']['diagnostics']=str(exc)
                row['repair_outcomes']=outcomes(metric_index[source['id']])
                records.append(row)
        write_jsonl(out/'pages.jsonl',records)
    detector=bucket.report() if seen_targets else None
    report=dict(contract=CONTRACT,dataset=args.dataset,threshold=args.threshold,checkpoint_sha=checkpoint_sha,
                attempted_records=len(records),diagnostic_errors=sum(bool(r['errors']) for r in records),
                initial_failed=sum(r['initial_failed'] for r in records),detector=detector,pairs=summarize(records),
                repair_associations=associations(records),
                note='Class-aware one-to-one IoU>=0.5. Reference uses the same full visible DOM collector as detector training. '
                     'Initial unmatched boxes indicate geometric disagreement, NOT proven semantic absence. '
                     'Detector counts are deduplicated across models/seeds; diagnostics are excluded from repair latency. '
                     'Associations with repair outcomes are observational, not causal.')
    write_json(out/'summary.json',report)
    lines=['| Dataset | Target renderings | Precision (%) | Recall (%) | FP | FN |',
           '|---|---:|---:|---:|---:|---:|']
    if detector:
        recall=detector['recall_iou50']
        lines.append(f"| {args.dataset} | {detector['n']} | {100*detector['precision_iou50']:.2f} | {100*recall:.2f} | {detector['false_positives']} | {detector['false_negatives']} |" if recall is not None else f'| {args.dataset} | {detector["n"]} | -- | -- | -- | -- |')
    (out/'detector-table.md').write_text('\n'.join(lines)+'\n')
    lines=['| Initial model | Repair method | Comparison | Disagreement | N | Mean Box IoU change | Improved (%) | Worsened (%) |',
           '|---|---|---|---|---:|---:|---:|---:|']
    for item in report['repair_associations']:
        lines.append(f"| {item['model']} | {item['method']} | {item['comparison']} | {item['disagreement']} | {item['n']} | {item['mean_delta_box_iou']:.4f} | {item['improved_percent']:.2f} | {item['worsened_percent']:.2f} |")
    (out/'repair-associations.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(diagnostics=str(out),records=len(records),errors=report['diagnostic_errors'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='Dataset experiment folder containing model/seeds/seed/prepare')
    p.add_argument('--dataset',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--detector-checkpoint',required=True)
    p.add_argument('--threshold',type=float,default=.4)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if not 0<=args.threshold<=1:p.error('threshold must be in [0,1]')
    run(args)


if __name__=='__main__':main()

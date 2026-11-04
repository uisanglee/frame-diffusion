"""Detector-vs-oracle diagnosis on an already prepared paired image-asset pilot."""
import argparse
import html
import io
import json
import shutil
import statistics
from pathlib import Path

from .ir import read_jsonl, write_json, write_jsonl


def reference_labels(browser,row):
    """Same full-visible collector as detector GT; verify against evaluated PNG."""
    import numpy as np
    from PIL import Image
    from .visual import detector_labels
    browser.reset_context()
    browser.load(Path(row['target_html']).read_text(),row['viewport'])
    png,_=browser.stable_capture(timeout_ms=15000)
    with Image.open(io.BytesIO(png)) as actual, Image.open(row['target_image']) as expected:
        if not np.array_equal(np.asarray(actual.convert('RGB')),np.asarray(expected.convert('RGB'))):
            raise ValueError(f'Target render changed: {row["id"]}; oracle labels would not match the saved target image')
    labels=detector_labels(browser)
    if not labels['elements']:raise ValueError(f'No visible oracle elements: {row["id"]}')
    return labels


def paired_inputs(root):
    conditions={mode:list(read_jsonl(root/mode/'policy-test.jsonl')) for mode in ('blocked','images')}
    left,right=conditions.values()
    if not left or len({r['id'] for r in left})!=len(left):raise ValueError('Empty/duplicate pilot samples')
    if [r['id'] for r in left]!=[r['id'] for r in right]:raise ValueError('Image conditions must use exactly the same ordered sample IDs')
    for a,b in zip(left,right):
        if a['viewport']!=b['viewport']:raise ValueError('Paired viewports differ')
        if a['declaration_state']!=b['declaration_state'] or a['target_declaration_state']!=b['target_declaration_state']:
            raise ValueError(f'Paired CSS edits differ: {a["id"]}')
    return conditions


def overlay(image,items,viewport,color):
    rects=[]
    for item in items:
        x,y,xx,yy=item['box']
        rects.append(f'<rect x="{x}" y="{y}" width="{max(0,xx-x)}" height="{max(0,yy-y)}"><title>class {int(item["label"])}</title></rect>')
    return (f'<svg viewBox="0 0 {viewport[0]} {viewport[1]}" xmlns="http://www.w3.org/2000/svg">'
            f'<image href="{html.escape(image,quote=True)}" width="{viewport[0]}" height="{viewport[1]}"/>'
            f'<g fill="none" stroke="{color}" stroke-width="2">'+''.join(rects)+'</g></svg>')


def audit(args):
    from .html_bridge import HtmlBrowser
    from .visual_train import load_detector, detect, detector_input
    from .visual import VISUAL_GEOMETRY_CONTRACT
    from .detector_support import Aggregate, compare, metrics
    from .web_experiment import guard_run,digest,signature
    from .fixed_test import checkpoint_overlap
    root=Path(args.root).resolve();out=root/'oracle-analysis';out.mkdir(exist_ok=True)
    conditions=paired_inputs(root)
    net,checkpoint=load_detector(args.detector_checkpoint,args.device)
    if checkpoint.get('config',{}).get('visual_geometry_contract')!=VISUAL_GEOMETRY_CONTRACT:
        raise ValueError('Oracle diagnosis requires the full-visible-DOM detector')
    identity=dict(contract='paired-image-detector-oracle-v1',checkpoint=digest(args.detector_checkpoint),
                  threshold=args.threshold,inputs={mode:[dict(id=r['id'],html=digest(r['target_html']),
                       image=digest(r['target_image']),viewport=r['viewport']) for r in rows] for mode,rows in conditions.items()})
    guard_run(out,identity,True)
    records=[];summary={};sections=[]
    with HtmlBrowser() as browser:
        for mode,rows in conditions.items():
            bucket=Aggregate()
            for row in rows:
                if checkpoint_overlap(row,[checkpoint]):raise ValueError('Pilot overlaps detector training')
                folder=out/'detections'/mode/signature(row['id'])[:20];folder.mkdir(parents=True,exist_ok=True)
                cache=folder/'record.json'
                if cache.is_file():record=json.loads(cache.read_text())
                else:
                    labels=reference_labels(browser,row)
                    predicted=detect(net,detector_input(row['target_image']),args.threshold)
                    detail=compare(predicted,labels['elements'])
                    record=dict(id=row['id'],condition=mode,viewport=row['viewport'],
                                target=labels['elements'],predicted=predicted,label_audit=labels['label_audit'],
                                detail=detail,**metrics(len(detail['matches']),len(predicted),len(labels['elements']),n=1))
                    write_json(cache,record)
                # Recompute matching to avoid JSON integer-key conversion issues.
                bucket.add(row,record['predicted'],record['target'],compare(record['predicted'],record['target']))
                shutil.copyfile(row['target_image'],folder/'target.png')
                relative=(folder/'target.png').relative_to(out).as_posix()
                sections.append(f'<h2>{html.escape(mode+" / "+row["id"])}</h2><div><section><h3>Reference DOM (green)</h3>'+
                                overlay(relative,record['target'],row['viewport'],'green')+
                                '</section><section><h3>Detector (red)</h3>'+overlay(relative,record['predicted'],row['viewport'],'red')+'</section></div>')
                records.append(record)
                print(json.dumps({k:record[k] for k in ('id','condition','precision_iou50','recall_iou50','false_positives','false_negatives')}),flush=True)
            summary[mode]=bucket.report()
    write_jsonl(out/'detector-pages.jsonl',records);write_json(out/'detector-summary.json',summary)
    lines=['| Condition | N | Precision (%) | Recall (%) | FP | FN |','|---|---:|---:|---:|---:|---:|']
    for mode,r in summary.items():
        lines.append(f'| {mode} | {r["n"]} | {100*r["precision_iou50"]:.2f} | {100*r["recall_iou50"]:.2f} | {r["false_positives"]} | {r["false_negatives"]} |')
    (out/'detector-table.md').write_text('\n'.join(lines)+'\n')
    (out/'detector.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:sans-serif}div{display:flex;gap:12px}section{width:50%}svg{width:100%}</style>'+''.join(sections))


def report(args):
    root=Path(args.root).resolve();out=root/'oracle-analysis'
    conditions=paired_inputs(root)
    detections={(r['condition'],r['id']):r for r in read_jsonl(out/'detector-pages.jsonl')}
    paired=[];summary=[];sections=[]
    for mode,inputs in conditions.items():
        results=list(read_jsonl(out/mode/'repair/results.jsonl'))
        indexed={(r['id'],r['method']):r for r in results}
        expected={(r['id'],method) for r in inputs for method in ('abstract-policy','abstract-oracle')}
        if len(results)!=len(expected) or set(indexed)!=expected:raise ValueError('Missing or duplicate paired rollout results')
        for source in inputs:
            identity=source['id'];pred=indexed[identity,'abstract-policy'];oracle=indexed[identity,'abstract-oracle']
            if abs(pred['initial_box_iou']-oracle['initial_box_iou'])>1e-9:raise ValueError('Different initial states within ablation')
            det=detections[mode,identity]
            paired.append(dict(id=identity,condition=mode,detector_precision=det['precision_iou50'],
                               detector_recall=det['recall_iou50'],fp=det['false_positives'],fn=det['false_negatives'],
                               predicted=pred,oracle=oracle,oracle_minus_predicted_iou=oracle['final_box_iou']-pred['final_box_iou']))
            sections.append(f'<h2>{html.escape(mode+" / "+identity)}</h2><p>Detector P={det["precision_iou50"]:.3f}, R={det["recall_iou50"]:.3f}; '
                            f'Final IoU: predicted={pred["final_box_iou"]:.3f}, oracle={oracle["final_box_iou"]:.3f}</p><div>')
            work=Path(pred['png']).parent
            for label,path in [('Target',work/'target.png'),('Initial',work/'initial.png'),('Detector repair',Path(pred['png'])),('Oracle repair',Path(oracle['png']))]:
                sections.append(f'<figure><figcaption>{label}</figcaption><img src="{path.relative_to(out).as_posix()}"></figure>')
            sections.append('</div>')
        for method in ('abstract-policy','abstract-oracle'):
            group=[r for r in results if r['method']==method]
            summary.append(dict(condition=mode,method=method,n=len(group),
                initial_iou=statistics.mean(r['initial_box_iou'] for r in group),
                final_iou=statistics.mean(r['final_box_iou'] for r in group),
                improved_percent=100*statistics.mean(r['final_box_iou']>r['initial_box_iou'] for r in group),
                failures=sum(r['failed'] for r in group),scroll_rollbacks=sum(r['scroll_guard_rolled_back'] for r in group)))
    write_jsonl(out/'paired-results.jsonl',paired);write_json(out/'repair-summary.json',summary)
    lines=['| Images | Target abstraction | N | Initial IoU | Final IoU | Improved (%) | Failed | Scroll rollback |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["condition"]} | {r["method"]} | {r["n"]} | {r["initial_iou"]:.4f} | {r["final_iou"]:.4f} | {r["improved_percent"]:.1f} | {r["failures"]} | {r["scroll_rollbacks"]} |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:sans-serif}div{display:flex}figure{width:25%;margin:5px}img{width:100%}</style>'
                                '<h1>Detector vs oracle repair</h1><a href="detector.html">Detector box comparison</a>'+''.join(sections))
    print(out/'comparison.md');print(out/'index.html')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','report'])
    p.add_argument('--root',default='runs/webui-image-assets-pilot10')
    p.add_argument('--detector-checkpoint')
    p.add_argument('--threshold',type=float,default=.4)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if not 0<=args.threshold<=1:p.error('Invalid threshold')
    if args.stage=='audit' and not args.detector_checkpoint:p.error('--detector-checkpoint required')
    (audit if args.stage=='audit' else report)(args)


if __name__=='__main__':main()

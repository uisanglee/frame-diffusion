from __future__ import annotations
import csv
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from .ir import read_jsonl,write_json,write_jsonl,execute
from .model import load_model,encode,collate
from .metrics import box_metrics
from .search import repair,Executor
from .train import select_device

def evaluate(args):
    records=list(read_jsonl(args.data))
    if args.limit:records=records[:args.limit]
    if not records:raise ValueError('Empty evaluation dataset')
    device=select_device(args.device)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    model=ck=None
    if args.checkpoint:
        model,ck=load_model(args.checkpoint,device)
        overlap=set(ck.get('training_groups',[]))&{r['group'] for r in records}
        if overlap and not args.allow_train_overlap:raise ValueError(f'{len(overlap)} evaluation groups seen in training')
    methods=args.methods.split(',')
    allowed={'none','copy-boxes','coordinate','random','model','model-browser','one-shot'}
    if set(methods)-allowed:raise ValueError(f'Unknown methods: {set(methods)-allowed}')
    if any(m in methods for m in ('model','model-browser','one-shot')) and model is None:raise ValueError('Checkpoint required')
    if 'one-shot' in methods and ck['objective']!='boxes':raise ValueError('one-shot needs a checkpoint trained with --objective boxes')
    if any(m in methods for m in ('model','model-browser')) and ck['objective']!='edits':raise ValueError('Edit checkpoint required')
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    browser=None
    if args.browser_verify or 'model-browser' in methods:
        from .browser import Browser
        browser=Browser().__enter__()
    rows=[]
    try:
        for record in records:
            for method in methods:
                if device.startswith('cuda'):torch.cuda.synchronize()
                start=time.perf_counter();initial=record['current'];obs=record['observations']
                truth=record.get('evaluation_observations',obs)
                if record.get('target_kind')=='predicted_frames' and 'evaluation_observations' not in record:
                    raise ValueError('Predicted-frame evaluation requires independent evaluation_observations')
                if [o['viewport'] for o in obs]!=[o['viewport'] for o in truth]:raise ValueError('Evaluation and input viewports must match in order')
                root=initial['nodes'][0]['id'];stats={'executions':0,'candidates':0,'edits':[]}
                before_calls=browser.executions if browser else 0
                executor=Executor(browser if method=='model-browser' else None)
                result=initial;frames=None
                if method in ('coordinate','random','model','model-browser'):
                    result,stats=repair(initial,obs,'model' if method.startswith('model') else method,model,args.steps,args.beam,args.topk,args.budget,executor,args.seed)
                elif method=='copy-boxes':
                    # Explicit oracle baseline: shows why unconstrained box prediction is trivial.
                    frames=[o['target'] for o in obs]
                elif method=='one-shot':
                    if len(obs)!=1:raise ValueError('one-shot box baseline currently requires single-viewport records')
                    with torch.inference_mode():
                        out=model(collate([encode(initial,obs,model.cfg.max_nodes)],device))['boxes'][0].cpu().tolist()
                    w,h=obs[0]['viewport']
                    frames=[{n['id']:[b[0]*w,b[1]*h,max(0,b[2]*w),max(0,b[3]*h)] for n,b in zip(initial['nodes'],out)}]
                if device.startswith('cuda'):torch.cuda.synchronize()
                repair_seconds=time.perf_counter()-start
                if frames is None:frames=[execute(result,o['viewport']) for o in obs]
                per_view=[box_metrics(b,o['target'],o['viewport'],(root,)) for b,o in zip(frames,truth)]
                row={'id':record['id'],'group':record['group'],'method':method,'target_kind':record.get('target_kind','unknown'),
                     **{k:statistics.mean(m[k] for m in per_view) for k in per_view[0]},
                     'worst_view_iou':min(m['box_iou'] for m in per_view), 'repair_seconds':repair_seconds,
                     'proxy_executions':stats['executions'] if method!='model-browser' else 0,
                     'candidate_count':stats['candidates'],'edit_count':len(stats['edits']),
                     'executable_output':method not in ('copy-boxes','one-shot')}
                if args.browser_verify and row['executable_output']:
                    errors=[];browser_metrics=[]
                    for index,o in enumerate(obs):
                        screenshot=output/'screenshots'/f'{len(rows)}-{index}.png' if args.screenshots else None
                        actual=browser.render(result,o['viewport'],screenshot)
                        errors.extend(abs(actual[k][j]-frames[index][k][j]) for k in actual for j in range(4))
                        browser_metrics.append(box_metrics(actual,truth[index]['target'],o['viewport'],(root,)))
                    row['proxy_browser_max_error_px']=max(errors)
                    row['browser_box_iou']=statistics.mean(x['box_iou'] for x in browser_metrics)
                row['browser_executions']=(browser.executions-before_calls) if browser else 0
                row['total_seconds']=time.perf_counter()-start
                artifact={'record_id':record['id'],'method':method,'result':result if row['executable_output'] else None,
                          'frames':frames,'trace':stats}
                write_json(output/'predictions'/f'{len(rows)}.json',artifact)
                rows.append(row);print(f"{record['id']} {method}: IoU={row['box_iou']:.4f} time={repair_seconds:.3f}s",flush=True)
    finally:
        if browser:browser.__exit__()
    write_jsonl(output/'results.jsonl',rows)
    report_rows(rows,output,args.seed)
    config={k:v for k,v in vars(args).items() if k!='func'}
    config['device_actual']=device;config['torch']=str(torch.__version__)
    write_json(output/'manifest.json',config)
    return rows

def report_rows(rows,out,seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metrics=['box_iou','center_error','size_error','relation_accuracy','element_success_09','page_success_09','overflow_rate','worst_view_iou','repair_seconds','total_seconds','browser_executions','proxy_executions']
    summaries=[];rng=np.random.default_rng(seed)
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method];summary={'method':method,'n':len(group)}
        for key in metrics:summary[key]=statistics.mean(r[key] for r in group)
        for key in ('browser_box_iou','proxy_browser_max_error_px'):
            values=[r[key] for r in group if key in r]
            summary[key]=statistics.mean(values) if values else None
        summary['p95_seconds']=float(np.percentile([r['repair_seconds'] for r in group],95))
        # Cluster bootstrap by domain/group, not by viewport or corrupted duplicate.
        groups=sorted({r['group'] for r in group})
        means=[statistics.mean(r['box_iou'] for r in group if r['group']==g) for g in groups]
        values=np.asarray(means);boot=values[rng.integers(0,len(values),(1000,len(values)))].mean(1)
        summary['group_mean_iou']=float(values.mean())
        summary['group_iou_ci95']=[float(x) for x in np.percentile(boot,[2.5,97.5])]
        summaries.append(summary)
    write_json(out/'summary.json',summaries)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    lines=['# Evaluation report','', '| Method | n | IoU ↑ | center error ↓ | size error ↓ | seconds ↓ | browser executions |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['method']} | {s['n']} | {s['box_iou']:.4f} | {s['center_error']:.4f} | {s['size_error']:.4f} | {s['repair_seconds']:.3f} | {s['browser_executions']:.1f} |")
    lines+=['','`copy-boxes` copies input coordinates; oracle only when those are ground truth, never executable code repair.',
            'Latency includes proposal/selection; total_seconds additionally includes final verification and metrics.',
            'Published LayoutDM / LayoutFormer++ results are NOT reproduced or relabeled by this runner.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')

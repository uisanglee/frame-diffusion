"""Frozen held-out WebUI CSS-corruption rollouts with inspectable artifacts."""
import copy
import io
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from . import css_owners
from .html_bridge import HtmlBrowser
from .ir import read_json, read_jsonl, write_json, write_jsonl
from .metrics import box_metrics
from .train import select_device
from .visual import load_policy
from .visual_experiment import rollout, sync
from .visual_train import load_detector
from .web_experiment import digest, guard_run, signature


KIND = 'frozen-webui-css-rollout-v1'


def state_distance(a, b):
    if len(a) != len(b) or any(len(x) != len(y) for x,y in zip(a,b)):
        raise ValueError('CSS declaration state shape changed during rollout')
    return sum(x != y for row,target in zip(a,b) for x,y in zip(row,target))


def pixel_mae(a, b):
    with Image.open(a) as left, Image.open(b) as right:
        if left.size != right.size: raise ValueError('Target/final screenshot dimensions differ')
        x=np.asarray(left.convert('RGB'),dtype=np.float32)
        y=np.asarray(right.convert('RGB'),dtype=np.float32)
    return float(np.abs(x-y).mean()/255.)


def boxes(browser, tree, viewport):
    result=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
    result[tree['nodes'][0]['id']]=[0,0,*viewport]
    return result


def checkpoint_overlap(row, checkpoints):
    for checkpoint in checkpoints:
        groups=set(checkpoint.get('training_groups',[]))|set(checkpoint.get('parser_training_groups',[]))
        hashes=set(checkpoint.get('training_hashes',[]))|set(checkpoint.get('parser_training_hashes',[]))
        if row['group'] in groups or row.get('source_sha') in hashes:
            return True
    return False


def evaluate(args):
    if args.steps<1 or args.repeats<1 or args.limit<0 or args.time_budget<0 or not 0<=args.threshold<=1:
        raise ValueError('Invalid fixed-test rollout settings')
    if not args.raw_checkpoint and not args.abstract_checkpoint:
        raise ValueError('Supply --raw-checkpoint and/or --abstract-checkpoint')
    if args.abstract_checkpoint and not args.detector_checkpoint:
        raise ValueError('Abstract rollout requires --detector-checkpoint')
    for key in ('goal_threshold','abstract_goal_threshold'):
        value=getattr(args,key,-1)
        if value!=-1 and not 0<=value<=1:raise ValueError(key+' must be -1 or in [0,1]')
    device=select_device(args.device)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    methods={};checkpoints=[];parser=None
    if args.raw_checkpoint:
        raw,checkpoint=load_policy(args.raw_checkpoint,device)
        if raw.cfg.mode!='screenshot':raise ValueError('Raw checkpoint is not a screenshot policy')
        methods['screenshot-policy']=raw;checkpoints.append(checkpoint)
    if args.abstract_checkpoint:
        abstract,checkpoint=load_policy(args.abstract_checkpoint,device)
        if abstract.cfg.mode!='abstract':raise ValueError('Abstract checkpoint is not an abstract policy')
        methods['abstract-policy']=abstract;checkpoints.append(checkpoint)
        parser,checkpoint=load_detector(args.detector_checkpoint,device);checkpoints.append(checkpoint)
    rows=list(read_jsonl(args.data));rows=rows[:args.limit] if args.limit else rows
    page_rows={r['id']:r for r in read_jsonl(args.pages)}
    if not rows:raise ValueError('Empty frozen WebUI test set')
    if any(r.get('split')!='test' for r in rows):raise ValueError('visual-test-rollout accepts test rows only')
    for row in rows:
        page_id=row['id'].rsplit('/',1)[0]
        page=page_rows.get(page_id)
        if not page or not page.get('evaluation_target_boxes'):
            raise ValueError(f'Missing exact held-out target boxes for {page_id}')
        if checkpoint_overlap(row,checkpoints):raise ValueError('Held-out test overlaps policy/parser training')
    out=Path(args.out).resolve()
    settings={k:v for k,v in vars(args).items() if k not in ('out','resume')}
    paths=[args.data,args.pages,args.raw_checkpoint,args.abstract_checkpoint,args.detector_checkpoint]
    guard_run(out,{'kind':KIND,'inputs':{p:digest(p) for p in paths if p},'settings':settings},args.resume)
    results=[]
    with HtmlBrowser() as browser:
        for index,row in enumerate(rows):
            page_id=row['id'].rsplit('/',1)[0];page=page_rows[page_id]
            work=out/'pages'/signature(row['id'])[:20];work.mkdir(parents=True,exist_ok=True)
            cache=work/'result.json'
            if args.resume and cache.exists():
                saved=read_json(cache);results.extend(saved);continue
            target_boxes=page['evaluation_target_boxes'];viewport=row['viewport'];tree=row['current']
            root=tree['nodes'][0]['id'];initial_html=Path(row['current_html']).read_text()
            shutil.copyfile(row['target_image'],work/'target.png')
            shutil.copyfile(row['current_image'],work/'initial.png')
            (work/'target.html').write_text(Path(row['target_html']).read_text())
            (work/'initial.html').write_text(initial_html)
            initial_geometry=box_metrics(row['current_boxes'],target_boxes,viewport,(root,))
            local=[]
            for repeat in range(1,args.repeats+1):
                for method,policy in methods.items():
                    started=time.perf_counter();error=None;stats={};html=initial_html
                    try:
                        html,stats,_=rollout(browser,initial_html,copy.deepcopy(tree),viewport,row['target_image'],policy,
                            parser if policy.cfg.mode=='abstract' else None,args.threshold,args.steps,args.time_budget,
                            goal_threshold=(args.abstract_goal_threshold if policy.cfg.mode=='abstract' else args.goal_threshold))
                        if stats.get('failed'):error=stats.get('error') or 'Rollout failed'
                    except Exception as exc:error=str(exc)
                    html_path=work/f'{method}-r{repeat}.html';html_path.write_text(html)
                    browser.reset_context();browser.load(html,viewport)
                    final_boxes=boxes(browser,tree,viewport)
                    final_state=css_owners.read(browser,tree)['state']
                    png=work/f'{method}-r{repeat}.png'
                    browser.page.screenshot(path=str(png),animations='disabled')
                    final_geometry=box_metrics(final_boxes,target_boxes,viewport,(root,))
                    metrics={f'initial_{k}':v for k,v in initial_geometry.items()}
                    metrics.update({f'final_{k}':v for k,v in final_geometry.items()})
                    metrics.update(pixel_mae=pixel_mae(row['target_image'],png),
                        initial_symbolic_distance=state_distance(row['declaration_state'],row['target_declaration_state']),
                        final_symbolic_distance=state_distance(final_state,row['target_declaration_state']),
                        actions=stats.get('actions',0),seconds=stats.get('seconds',time.perf_counter()-started),
                        browser_executions=stats.get('browser_executions',0),
                        browser_screenshots=stats.get('browser_screenshots',0))
                    record={'id':row['id'],'page_id':page_id,'method':method,'repeat':repeat,
                        'failed':error is not None,'error':error,'html':str(html_path),'png':str(png),**metrics}
                    write_json(work/f'{method}-r{repeat}-trace.json',stats)
                    local.append(record)
            write_json(cache,local);results.extend(local);write_jsonl(out/'results.jsonl',results)
            print(f'[{index+1}/{len(rows)}] {row["id"]}: fixed WebUI test rollout complete',flush=True)
    write_jsonl(out/'results.jsonl',results)
    summary=[]
    for method in methods:
        selected=[r for r in results if r['method']==method]
        item={'method':method,'n':len(selected),'pages':len({r['page_id'] for r in selected}),
              'failure_rate':statistics.mean(r['failed'] for r in selected)}
        for key in ('initial_box_iou','final_box_iou','initial_center_error','final_center_error',
                    'initial_size_error','final_size_error','pixel_mae','initial_symbolic_distance',
                    'final_symbolic_distance','actions','seconds','browser_executions','browser_screenshots'):
            item[key]=statistics.mean(r[key] for r in selected)
        item['geometry_improvement_rate']=statistics.mean(r['final_box_iou']>r['initial_box_iou'] for r in selected)
        item['symbolic_improvement_rate']=statistics.mean(r['final_symbolic_distance']<r['initial_symbolic_distance'] for r in selected)
        summary.append(item)
    write_json(out/'summary.json',summary)
    lines=['# Frozen held-out WebUI CSS rollout','',
           'Fixed test corruptions only; no VLM generation and no target state is used for action selection.','',
           '| Method | n | failed | initial IoU | final IoU | pixel MAE | symbolic before | symbolic after | actions | seconds |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f'| {s["method"]} | {s["n"]} | {s["failure_rate"]:.3f} | {s["initial_box_iou"]:.4f} | '
                     f'{s["final_box_iou"]:.4f} | {s["pixel_mae"]:.4f} | {s["initial_symbolic_distance"]:.3f} | '
                     f'{s["final_symbolic_distance"]:.3f} | {s["actions"]:.2f} | {s["seconds"]:.3f} |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return results

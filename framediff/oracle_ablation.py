"""Controlled real-HTML ablation with exact identities and oracle/VLM target boxes."""
import json
import math
import random
import time
from pathlib import Path

from PIL import Image

from . import vlm
from .adapters import fit_observation
from .benchmarks import discover_design2code
from .html_bridge import HtmlBrowser, embed_placeholder
from .ir import read_json, write_json, write_jsonl
from .metrics import box_metrics, objective
from .web_experiment import digest, guard_run, signature, validate_target, validated_generation


def add_parser(sub):
    p = sub.add_parser('web-oracle-prepare',
        help='Build controlled same-DOM corruptions with VLM and oracle target boxes')
    p.add_argument('--root',required=True); p.add_argument('--out',required=True)
    p.add_argument('--backend',choices=['qwen','hf','openai-compatible'],default='qwen')
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct'); p.add_argument('--revision',default='main')
    p.add_argument('--endpoint',default='http://localhost:8000/v1/chat/completions')
    p.add_argument('--api-key-env',default='VLM_API_KEY'); p.add_argument('--four-bit',action='store_true')
    p.add_argument('--resume',action='store_true'); p.add_argument('--retry-failed',action='store_true')
    for key,default in [('max-nodes',64),('max-new-tokens',16384),('max-pixels',1048576),
                        ('limit',0),('seed',42),('corruptions',4),('vlm-retries',2)]:
        p.add_argument('--'+key,type=int,default=default)


def _corrupt(browser, html, tree, boxes, viewport, count, rng):
    candidates=[n for n in tree['nodes'][1:] if n['role'] not in ('body','html') and n['id'] in boxes]
    if not candidates: candidates=tree['nodes'][1:]
    if not candidates: raise ValueError('No editable reference elements')
    current=html; actions=[]; fields=('width','height','dx','dy')
    for index in range(count):
        node=candidates[index%len(candidates)] if index<len(candidates) else rng.choice(candidates)
        field=fields[index%len(fields)]; box=boxes[node['id']]
        scale={'width':max(16,min(box[2]*.25,viewport[0]*.12)),
               'height':max(12,min(box[3]*.25,viewport[1]*.12)),
               'dx':max(12,viewport[0]*.04),'dy':max(12,viewport[1]*.04)}[field]
        # Stable seed and alternating directions make the corruption reproducible.
        delta=scale*(1 if rng.random()>=.5 else -1)
        browser.edit_property(current,viewport,node['id'],field,delta)
        current=browser.page.content()
        measured=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
        actual=max(abs(a-b) for a,b in zip(measured[node['id']],boxes[node['id']]))
        actions.append({'node':node['id'],'role':node['role'],'field':field,
                        'requested_delta_px':delta,'target_box':boxes[node['id']],
                        'result_box':measured[node['id']],'changed_from_target_px':actual})
    browser.load(current,viewport)
    measured=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
    measured[tree['nodes'][0]['id']]=[0,0,*viewport]
    if not all(len(b)==4 and all(math.isfinite(x) for x in b) and min(b[2:])>0 for b in measured.values()):
        raise ValueError('Corruption produced invalid boxes')
    if objective(measured,boxes,viewport,tree['nodes'][0]['id'])<1e-4:
        raise ValueError('Corruption did not change measured layout')
    return current,measured,actions


def prepare_oracle(args):
    if (not 2<=args.max_nodes<=256 or args.max_new_tokens<1 or args.max_pixels<1 or
            args.limit<0 or args.corruptions<1 or args.vlm_retries<0):
        raise ValueError('Invalid oracle preparation settings')
    pairs=discover_design2code(args.root)
    if args.limit:pairs=pairs[:args.limit]
    if not pairs:raise ValueError('No Design2Code PNG/HTML pairs found')
    out=Path(args.out).resolve()
    sources=[{'id':i,'image_sha':digest(p),'html_sha':digest(h)} for i,p,h in pairs]
    settings={k:v for k,v in vars(args).items() if k not in ('out','resume','retry_failed')}
    guard_run(out,{'protocol':2,'settings':settings,'sources':sources},args.resume)
    runtime=None; rows=[]; report=[]
    with HtmlBrowser() as browser:
        for index,(sample_id,image_path,html_path) in enumerate(pairs):
            work=out/'pages'/signature(sample_id)[:20];work.mkdir(parents=True,exist_ok=True)
            cache=work/'record.json'
            if args.resume and cache.exists():
                cached=read_json(cache)
                if not (args.retry_failed and cached.get('errors')):
                    report.append({'id':sample_id,'errors':cached.get('errors',{})})
                    if 'initial' in cached.get('methods',{}): rows.append(cached)
                    write_jsonl(out/'prepared.jsonl',rows);continue
            with Image.open(image_path) as image:viewport=list(image.size)
            placeholder=html_path.parent/'rick.jpg';placeholder=placeholder if placeholder.exists() else None
            target_png=work/'reference-target.png'
            record={'id':sample_id,'group':sample_id,'screenshot':str(target_png),
                    'dataset_screenshot':str(image_path.resolve()),
                    'html':str(html_path.resolve()),'viewport':viewport,'methods':{},'errors':{},
                    'target_stats':{},'target_extraction_metrics':None,'controlled_oracle':True,
                    'vlm_model':args.model,'vlm_attempts':{}}
            started=time.perf_counter();before=browser.executions
            try:
                reference=browser.snapshot(embed_placeholder(html_path.read_text(errors='replace'),placeholder),
                                           viewport,target_png,args.max_nodes-1)
                tagged=reference.pop('html')
                reference_tree,reference_boxes,fit=fit_observation(reference)
                root=reference_tree['nodes'][0]['id'];reference_boxes[root]=[0,0,*viewport]
                corrupted,corrupted_boxes,actions=_corrupt(browser,tagged,reference_tree,reference_boxes,
                    viewport,args.corruptions,random.Random(args.seed+index))
                corrupted_obs={'viewport':viewport,'nodes':[
                    {**{k:n[k] for k in ('id','parent','role','name')},'box':corrupted_boxes[n['id']]}
                    for n in reference_tree['nodes'][1:]]}
                tree,original,corrupt_fit=fit_observation(corrupted_obs)
                if [n['id'] for n in tree['nodes']] != [n['id'] for n in reference_tree['nodes']]:
                    raise ValueError('Controlled corruption changed fitted identities')
                oracle={'viewport':viewport,'target':reference_boxes}
                corruption_metrics=box_metrics(corrupted_boxes,reference_boxes,viewport,exclude=(root,))
                initial_path=work/'initial-corrupted.html';initial_path.write_text(corrupted)
                tagged_path=work/'tagged-initial.html';tagged_path.write_text(corrupted)
                write_json(work/'initial-ir.json',tree);write_json(work/'oracle-target.json',oracle)
                write_json(work/'corruption.json',{'actions':actions,'boxes':corrupted_boxes})
                construction_seconds=time.perf_counter()-started
                record.update(current=tree,original_boxes=original,tagged_html=str(tagged_path),
                    observations=[oracle],observations_by_source={'oracle':[oracle]},fit=corrupt_fit,
                    reference_fit=fit,corruption_actions=actions,selected_nodes=len(tree['nodes'])-1)
                record['corruption_metrics']=corruption_metrics
                record['methods']['initial']={'html':str(initial_path),'seconds':construction_seconds,
                    'browser_executions':browser.executions-before,'vlm_calls':0,'failed':False,'error':None}
                record['target_stats']['oracle']={'seconds':0.,'vlm_calls':0,'browser_executions':0}
                from .html_feedback import frame_png
                named_started=time.perf_counter()
                named_frame=work/'current-named-frame.png';named_frame.write_bytes(frame_png(tree,original,viewport))
                named_seconds=time.perf_counter()-named_started
                record['named_frame']=str(named_frame)
                ids={n['id'] for n in tree['nodes']};base=vlm.prompt_for(SimpleArgs(args,'extract-frames',work/'initial-ir.json'))
                extra=('Image 1 is the TARGET webpage screenshot. Image 2 is the CURRENT HTML named-frame map, '
                       'generated from exact browser boxes. Use image 2 to identify IDs, roles, nesting, and current '
                       'geometry; estimate output coordinates only from image 1. Image-2 positions are not targets. '
                       f'The required output viewport is EXACTLY {json.dumps(viewport)}. The original screenshot '
                       f'is {viewport[0]} by {viewport[1]} pixels. Return boxes in original pixels. ')
                images=vlm.images_for(SimpleArgs(args,'extract-frames',work/'initial-ir.json',[target_png,named_frame]))
                call_args=SimpleArgs(args,'extract-frames',work/'initial-ir.json',[target_png,named_frame])
                def generate(attempt,feedback):
                    nonlocal runtime
                    if args.backend in ('qwen','hf') and runtime is None:
                        runtime=(vlm.load_qwen_runtime if args.backend=='qwen' else vlm.load_hf_runtime)(args)
                    label='vlm-target' if attempt==0 else f'vlm-target.retry-{attempt}'
                    prompt=extra+base+feedback;write_json(work/f'{label}.prompt.json',{'prompt':prompt})
                    record['vlm_attempts']['vlm-target']=record['vlm_attempts'].get('vlm-target',0)+1
                    call_started=time.perf_counter();answer,meta=vlm.generate(call_args,prompt,images,runtime)
                    elapsed=time.perf_counter()-call_started;(work/f'{label}.raw.txt').write_text(answer)
                    write_json(work/f'{label}.meta.json',{**meta,'wall_seconds':elapsed});return answer,elapsed
                target_started=time.perf_counter()
                try:
                    predicted,_=validated_generation(generate,lambda s:validate_target(s,viewport,ids),args.vlm_retries)
                    record['observations_by_source']['vlm']=[predicted]
                    extraction=box_metrics(predicted['target'],reference_boxes,viewport,exclude=(root,))
                    record['target_extraction_metrics']=extraction
                    write_json(work/'vlm-target.json',predicted)
                except Exception as error:
                    record['errors']['vlm_frames']=str(error)
                record['target_stats']['vlm']={'seconds':named_seconds+time.perf_counter()-target_started,
                    'vlm_calls':record['vlm_attempts'].get('vlm-target',0),'browser_executions':0}
                write_json(cache,record)
            except Exception as error:
                record['errors']['prepare']=str(error)
                # A construction failure cannot yield a usable record; retain diagnostics only.
                write_json(cache,record)
            report.append({'id':sample_id,'errors':record['errors'],
                'corruption_metrics':record.get('corruption_metrics'),
                'target_extraction_metrics':record.get('target_extraction_metrics')})
            if 'initial' not in record['methods']:
                print(f'[{index+1}/{len(pairs)}] {sample_id}: oracle rejected, errors={record["errors"]}',flush=True)
                continue
            rows.append(record)
            write_jsonl(out/'prepared.jsonl',rows)
            print(f'[{index+1}/{len(pairs)}] {sample_id}: oracle prepared, errors={record["errors"]}',flush=True)
    valid=[r for r in rows if not r['errors']]
    metric_keys={k for r in valid for k in (r.get('target_extraction_metrics') or {})
                 if isinstance((r.get('target_extraction_metrics') or {}).get(k),(int,float))}
    summary={k:sum(r['target_extraction_metrics'][k] for r in valid)/len(valid) for k in metric_keys} if valid else {}
    constructed=[r for r in rows if r.get('corruption_metrics')]
    corruption_keys={k for r in constructed for k,v in r['corruption_metrics'].items() if isinstance(v,(int,float))}
    corruption_summary={k:sum(r['corruption_metrics'][k] for r in constructed)/len(constructed)
                        for k in corruption_keys} if constructed else {}
    write_json(out/'prepare-report.json',{'prepared':len(valid),'total':len(rows),
        'corruption_summary':corruption_summary,'target_extraction_summary':summary,'records':report})
    return rows


class SimpleArgs:
    """Small view over CLI arguments for shared VLM prompt/image helpers."""
    def __init__(self,args,task,current,image=None):
        self.__dict__.update(vars(args));self.task=task;self.current=str(current);self.frames=None
        if isinstance(image,(list,tuple)): self.image=[str(path) for path in image]
        else: self.image=[str(image)] if image else []

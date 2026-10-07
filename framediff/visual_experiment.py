"""Matched screenshot vs one-shot abstraction rollouts. No oracle in decisions."""
import copy
import io
from pathlib import Path
import random
import statistics
import time

from PIL import Image
import torch

from .html_bridge import HtmlBrowser
from .html_feedback import refresh_geometry
from .ir import read_jsonl,write_json,write_jsonl
from .metrics import box_metrics
from .train import select_device
from .visual import (CONTRACT,ACTION_CONTRACT,abstract_image,elements,image_tensor,semantic_masks,observation_mae,
                     current_features,visual_batch,decode_action,load_policy)
from .visual_train import load_detector,detect,detector_input
from .web_experiment import digest,guard_run,signature


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


def rollout(browser,html,tree,viewport,target_path,policy,parser=None,threshold=.4,
            steps=10,time_budget=0,oracle_elements=None,diagnostic_target_boxes=None,goal_threshold=-1):
    """Only the explicitly enabled oracle ablation accepts target elements.

    Autoregressive policies have no learned STOP. No box/pixel oracle ranking, no hidden
    candidate search. Wall-clock budgets are soft and checked between operations.
    """
    device=next(policy.parameters()).device;cfg=policy.cfg
    before=browser.executions;shots=browser.screenshots;sync(device);start=time.perf_counter()
    stats={'target_abstraction_seconds':0.,'target_encoding_seconds':0.,'current_render_seconds':0.,
           'current_encoding_seconds':0.,'policy_seconds':0.,'layout_seconds':0.,
           'abstraction_calls':0,'current_image_encodings':0,'target_image_encodings':0,
           'actions':0,'mode':cfg.mode,'stop_reason':'steps','history':[],
           'observation_contract':cfg.observation_contract,'goal_threshold':goal_threshold,
           'pair_encoding_seconds':0.,'pair_image_encodings':0,'dom_queries':0}
    t=time.perf_counter()
    if cfg.mode=='abstract':
        if oracle_elements is None:
            if parser is None:raise ValueError('Predicted abstraction needs detector')
            items=detect(parser,detector_input(target_path),threshold);stats['abstraction_calls']=1
            if not items:raise ValueError('Target abstraction detector found no elements; inspect detector validation/threshold')
        else:items=oracle_elements
        target=abstract_image(items,viewport,cfg.size)
        sync(device);stats['target_abstraction_seconds']=time.perf_counter()-t
        stats['predicted_elements']=len(items)
    else:
        with Image.open(target_path) as image:target=image.convert('RGB')
    target_tensor=(semantic_masks(items,viewport,cfg.size) if cfg.semantic else image_tensor(target,cfg.size))[None].to(device)
    target_tokens=None
    if cfg.policy_head not in ('autoregressive','replacement'):
        t=time.perf_counter()
        with torch.inference_mode():target_tokens,_=policy.encode_image(target_tensor)
        sync(device);stats['target_encoding_seconds']=time.perf_counter()-t;stats['target_image_encodings']=1
    root=tree['nodes'][0]['id'];ids=[n['id'] for n in tree['nodes'][1:]]
    t=time.perf_counter();browser.load(html,viewport);stats['layout_seconds']+=time.perf_counter()-t
    current_html=html;current_tree=copy.deepcopy(tree)
    for step in range(steps):
        if time_budget and time.perf_counter()-start>=time_budget:stats['stop_reason']='time_budget';break
        t=time.perf_counter();boxes=browser.tagged_boxes(ids);boxes[root]=[0,0,*viewport]
        stats['dom_queries']+=1
        current_tree=refresh_geometry(current_tree,boxes);stats['layout_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        if cfg.mode=='screenshot':
            png=browser.page.screenshot(animations='disabled');browser.screenshots+=1
            with Image.open(io.BytesIO(png)) as image:current_image=image.convert('RGB')
        elif cfg.semantic:current_image=semantic_masks(elements(current_tree,boxes,viewport),viewport,cfg.size)
        else:current_image=abstract_image(elements(current_tree,boxes,viewport),viewport,cfg.size)
        stats['current_render_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        current_tensor=(current_image if cfg.semantic else image_tensor(current_image,cfg.size))[None].to(device)
        if cfg.policy_head in ('autoregressive','replacement'):
            goal_mae=observation_mae(target_tensor,current_tensor,cfg.semantic)
            if goal_threshold>=0 and goal_mae<=goal_threshold:
                stats['history'].append({'step':step,'action':None,'boxes':boxes,
                    'goal_mae':goal_mae,'elapsed_seconds':time.perf_counter()-start})
                stats['stop_reason']='goal';break
        if cfg.policy_head not in ('autoregressive','replacement'):
            with torch.inference_mode():current_tokens,current_map=policy.encode_image(current_tensor)
        sync(device);stats['current_encoding_seconds']+=time.perf_counter()-t;stats['current_image_encodings']+=1
        t=time.perf_counter()
        if cfg.policy_head=='replacement':
            from .tree_policy import tree_batch
            context={'current':current_tree,'current_boxes':boxes,'viewport':viewport}
            if getattr(cfg,'stylesheets',False):
                from . import css_owners
                owner_state=css_owners.read(browser,current_tree)
                context['css_owners']=owner_state['owners']
            batch=tree_batch([context],device,cfg.max_nodes)
        else:batch=visual_batch([current_features(current_tree,boxes,viewport,cfg.max_nodes)],device)
        with torch.inference_mode():
            if cfg.policy_head=='replacement':
                from .tree_edits import current_state
                state=owner_state['state'] if getattr(cfg,'stylesheets',False) else current_state(browser,current_tree)
                if getattr(cfg,'existing_values_only',False) and not policy.tokenizer.allowed([],len(state),state=state):
                    stats['stop_reason']='no_editable_declarations'
                    break
                sync(device);pair_start=time.perf_counter()
                pair=policy.encode_pair(target_tensor,current_tensor)
                sync(device);pair_elapsed=time.perf_counter()-pair_start
                stats['pair_encoding_seconds']+=pair_elapsed
                action=policy.predict(batch,target_tensor,current_tensor,[state],pair)
                stats['pair_image_encodings']+=1;stats['target_image_encodings']+=1
            elif cfg.policy_head=='autoregressive':
                sync(device);pair_start=time.perf_counter()
                tokens,feature_map=policy.encode_pair(target_tensor,current_tensor)
                sync(device);pair_elapsed=time.perf_counter()-pair_start
                stats['pair_encoding_seconds']+=pair_elapsed;stats['pair_image_encodings']+=1
                logits=policy.decode(batch,tokens,None,feature_map)[0]
                stats['target_image_encodings']+=1
            else:logits=policy.decode(batch,target_tokens,current_tokens,current_map)[0]
            if cfg.policy_head!='replacement':
                action=decode_action(int(policy.select(logits)),len(current_tree['nodes']))
        sync(device);stats['policy_seconds']+=time.perf_counter()-t-(pair_elapsed if cfg.policy_head in ('autoregressive','replacement') else 0.)
        history={'step':step,'action':action,'boxes':boxes,'elapsed_seconds':time.perf_counter()-start}
        if cfg.policy_head=='replacement' and getattr(cfg,'stylesheets',False) and action:
            history['css_owner']=owner_state['owners'][action[0]]
        stats['history'].append(history)
        if action is None:stats['stop_reason']='policy_stop';break
        if time_budget and time.perf_counter()-start>=time_budget:stats['stop_reason']='time_budget';break
        t=time.perf_counter()
        if cfg.policy_head=='replacement':
            from .tree_edits import execute
            if getattr(cfg,'stylesheets',False):css_owners.execute(browser,owner_state['owners'],action)
            else:execute(browser,current_tree,action)
        else:
            i,field,delta=action
            browser.edit_visual_action(current_html,viewport,current_tree['nodes'][i]['id'],field,delta)
        current_html=browser.page.content();stats['layout_seconds']+=time.perf_counter()-t;stats['actions']+=1
    sync(device);stats['seconds']=time.perf_counter()-start
    stats['browser_executions']=browser.executions-before;stats['browser_screenshots']=browser.screenshots-shots
    stats['time_budget_overshoot']=max(0.,stats['seconds']-time_budget) if time_budget else 0.
    # Diagnostic work runs after the repair timer and cannot affect time-budget decisions.
    diagnostic_start=time.perf_counter()
    if diagnostic_target_boxes is not None and stats['actions']==len(stats['history']) and stats['actions']:
        boxes=browser.tagged_boxes(ids);boxes[root]=[0,0,*viewport]
        stats['history'].append({'step':len(stats['history']),'action':None,'boxes':boxes,
            'elapsed_seconds':stats['seconds'],'phase':'final_observation'})
    if diagnostic_target_boxes is not None:
        for state in stats['history']:
            state['diagnostic_metrics']=box_metrics(state['boxes'],diagnostic_target_boxes,viewport,(root,))
    stats['diagnostic_seconds']=time.perf_counter()-diagnostic_start
    return current_html,stats,target


def evaluate(args):
    device=select_device(args.device)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    raw,raw_ck=load_policy(args.raw_checkpoint,device)
    abstract,abstract_ck=load_policy(args.abstract_checkpoint,device)
    if getattr(args,'decoding',None):
        raw.cfg.decoding=args.decoding;abstract.cfg.decoding=args.decoding
    parser,parser_ck=load_detector(args.detector_checkpoint,device)
    if raw.cfg.mode!='screenshot' or abstract.cfg.mode!='abstract':raise ValueError('Policy modality mismatch')
    for key in ('hidden','layers','heads','max_nodes','token_grid','policy_head','numeric_only','stylesheets','existing_values_only'):
        if getattr(raw.cfg,key,False)!=getattr(abstract.cfg,key,False):raise ValueError('Matched policies must share tree/policy architecture')
    rows=list(read_jsonl(args.data));rows=rows[:args.limit] if args.limit else rows
    if not rows:raise ValueError('No prepared evaluation pages')
    if args.repeats<1 or args.time_budget<0:raise ValueError('Invalid repeats/time budget')
    for row in rows:
        for ck in (raw_ck,abstract_ck,parser_ck):
            groups=set(ck.get('training_groups',[]))|set(ck.get('parser_training_groups',[]))
            hashes=set(ck.get('training_hashes',[]))|set(ck.get('parser_training_hashes',[]))
            if row['group'] in groups or (row.get('html') and digest(row['html']) in hashes) or row.get('source_sha') in hashes:
                raise ValueError('Evaluation overlaps policy/parser training corpus')
    out=Path(args.out).resolve()
    guard_run(out,{'kind':'visual-comparison-v1','data':digest(args.data),
        'checkpoints':[digest(p) for p in (args.raw_checkpoint,args.abstract_checkpoint,args.detector_checkpoint)],
        'assets':[{k:digest(r[k]) for k in ('screenshot','tagged_html') if r.get(k)} for r in rows],
        'settings':{k:v for k,v in vars(args).items() if k not in ('out','resume')}},args.resume)
    # Startup and warmup are excluded equally. No real target is parsed in warmup.
    if args.warmup:
        with torch.inference_mode():
            for policy in (raw,abstract):
                blank=torch.zeros(1,4 if policy.cfg.semantic else 3,policy.cfg.size,policy.cfg.size,device=device)
                if policy.cfg.policy_head in ('autoregressive','replacement'):policy.encode_pair(blank,blank)
                else:policy.encode_image(blank)
            parser([torch.zeros(3,parser_ck['config']['min_size'],parser_ck['config']['min_size'],device=device)])
        sync(device)
    methods={'screenshot-policy':raw,'abstract-policy':abstract}
    if args.oracle_ablation:methods['abstract-oracle']=abstract
    results=[];timings=[];rng=random.Random(args.seed)
    with HtmlBrowser() as browser:
        for index,source in enumerate(rows):
            work=out/'pages'/signature(source['id'])[:20];work.mkdir(parents=True,exist_ok=True)
            cache=work/'result.json'
            if args.resume and cache.exists():
                from .ir import read_json
                saved=read_json(cache);results+=saved['records'];timings+=saved['timings'];continue
            record=copy.deepcopy(source);local=[];by_method={k:[] for k in methods}
            for repeat in range(args.repeats):
                order=list(methods);rng.shuffle(order)
                for method in order:
                    initial=record['methods']['initial'];html=Path(initial['html']).read_text();error=None;stats={};target=None
                    started=time.perf_counter()
                    try:
                        if initial['failed']:raise ValueError(initial.get('error') or 'Initial HTML generation failed')
                        if (record.get('visual_contract')!=CONTRACT or
                                record.get('visual_action_contract')!=ACTION_CONTRACT):
                            raise ValueError('Run web-prepare --repair-conditioning visual with the v3 no-padding/gap action contract')
                        oracle=None
                        if method=='abstract-oracle':
                            if not str(record.get('construction','')).startswith('controlled') or 'evaluation_target_boxes' not in record:
                                raise ValueError('Oracle ablation only supports visual-build-data controlled pages')
                            oracle=elements(record['current'],record['evaluation_target_boxes'],record['viewport'])
                        html,stats,target=rollout(browser,Path(record['tagged_html']).read_text(),record['current'],
                            record['viewport'],record['screenshot'],methods[method],parser,args.threshold,args.steps,args.time_budget,
                            oracle,record.get('evaluation_target_boxes'),
                            getattr(args,'abstract_goal_threshold',-1) if methods[method].cfg.semantic else args.goal_threshold)
                    except Exception as exc:error=str(exc)
                    if 'seconds' not in stats:stats['seconds']=time.perf_counter()-started
                    path=work/f'{method}-r{repeat+1}.html';path.write_text(html)
                    if target is not None and method!='screenshot-policy':target.save(work/f'{method}-r{repeat+1}-target.png')
                    write_json(work/f'{method}-r{repeat+1}-trace.json',stats)
                    row={'id':source['id'],'method':method,'repeat':repeat+1,'failed':error is not None,
                         'error':error,'html':str(path),**{k:v for k,v in stats.items() if k!='history'}}
                    local.append(row);by_method[method].append(row)
            # Preserve every trial's HTML and exact timing for paired accuracy under
            # time budgets; never combine first-trial accuracy with averaged timing.
            page_results=[]
            for repeat in range(args.repeats):
                trial_record=copy.deepcopy(record);trial_record['id']=source['id']+f'/repeat-{repeat+1}'
                trial_record['page_id']=source['id'];trial_record['repeat']=repeat+1
                for method,trials in by_method.items():
                    trial=trials[repeat]
                    numeric={k:v for k,v in trial.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
                    trial_record['methods'][method]={'html':trial['html'],'failed':trial['failed'],'error':trial['error'],
                        'seconds':initial['seconds']+record.get('frame_seconds',0)+trial['seconds'],
                        'vlm_calls':initial['vlm_calls'],'browser_executions':record.get('frame_browser_executions',0)+trial.get('browser_executions',0),
                        'repair_vlm_calls':0,'repair_input_tokens':0,'repair_output_tokens':0,
                        'repair_seconds':trial['seconds'],
                        'feedback_browser_screenshots':trial.get('browser_screenshots',0),
                        'feedback_mode':method,'target_source':'image','visual_timing':numeric}
                page_results.append(trial_record)
            results+=page_results;timings+=local;write_json(cache,{'records':page_results,'timings':local})
            write_jsonl(out/'results.jsonl',results);write_jsonl(out/'timings.jsonl',timings)
            print(f'[{index+1}/{len(rows)}] {source["id"]}: screenshot/abstraction comparison complete',flush=True)
    write_jsonl(out/'results.jsonl',results);write_jsonl(out/'timings.jsonl',timings)
    summary={}
    for method in methods:
        selected=[r for r in timings if r['method']==method];valid=[r for r in selected if not r['failed']]
        summary[method]={'n':len(selected),'failed':sum(r['failed'] for r in selected),
            'successful_n':len(valid),'successful_means':{k:statistics.mean(r[k] for r in valid if k in r)
              for k in ('seconds','target_abstraction_seconds','target_encoding_seconds','current_render_seconds',
                        'current_encoding_seconds','pair_encoding_seconds','pair_image_encodings',
                        'policy_seconds','layout_seconds','browser_screenshots','browser_executions','dom_queries','actions')
              if any(k in r for r in valid)}}
    write_json(out/'timing-summary.json',{'methods':summary,
        'model_parameters':{name:sum(p.numel() for p in model.parameters()) for name,model in methods.items()},
        'note':'Target preprocessing included; startup/warmup and final evaluation excluded. Abstract feedback still uses browser layout/reflow. Joint pair encoding runs every step; only target abstraction is cached. Thresholds default to disabled for fixed-budget comparison.'})
    return results

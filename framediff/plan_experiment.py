"""Held-out plan-policy evaluation. Oracle plans are explicitly diagnostic."""
import time
import statistics
from pathlib import Path
from .ir import read_jsonl,write_json,write_jsonl,execute
from .plans import oracle_plan,plan_metrics
from .plan_search import repair_ir
from .metrics import box_metrics


def add_parsers(sub):
    p=sub.add_parser('plan-build-html-data',help='Build browser-measured teacher examples from explicitly supplied training HTML')
    for key in ('manifest','out'):p.add_argument('--'+key,required=True)
    for key,default in [('per-page',16),('seed',42),('max-nodes',128),('width',1280),('height',720)]:
        p.add_argument('--'+key,type=int,default=default)
    p=sub.add_parser('plan-evaluate',help='Evaluate fixed-plan repair on held-out synthetic/fitted IR')
    for key in ('data','checkpoint','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='auto');p.add_argument('--methods',default='none,coordinate,model')
    for key,default in [('steps',10),('beam',2),('topk',32),('budget',320),('limit',0),('cpu-threads',4)]:
        p.add_argument('--'+key,type=int,default=default)


def evaluate(args):
    import torch
    from .model import load_model
    from .train import select_device
    device=select_device(args.device)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    model,ck=load_model(args.checkpoint,device)
    if model.cfg.conditioning!='plan':raise ValueError('Use a plan-conditioned checkpoint')
    records=list(read_jsonl(args.data));records=records[:args.limit] if args.limit else records
    if not records:raise ValueError('No evaluation records')
    if any(r.get('source')=='browser_teacher' for r in records):
        raise ValueError('Browser teacher records require real web evaluation, not the IR proxy evaluator')
    if {r['group'] for r in records}&set(ck['training_groups']):raise ValueError('Training group overlap')
    methods=args.methods.split(',')
    if len(set(methods))!=len(methods) or not set(methods)<= {'none','coordinate','model'}:raise ValueError('Invalid methods')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,record in enumerate(records):
        obs=record['observations'][0]
        plan=record.get('plan') or oracle_plan(record['clean'],obs['target'],obs['viewport'])
        work=out/f'page-{index:05d}';write_json(work/'plan.json',plan)
        for method in methods:
            start=time.perf_counter()
            if method=='none':boxes=execute(record['current'],obs['viewport']);stats={'executions':1,'candidates':0}
            else:
                result,stats=repair_ir(record['current'],plan,model=model,method=method,steps=args.steps,
                    beam=args.beam,topk=args.topk,budget=args.budget,trace_dir=work/method)
                boxes=result['boxes'];write_json(work/f'{method}.json',result['tree'])
            seconds=time.perf_counter()-start
            rows.append({'id':record['id'],'method':method,'plan_source':plan.get('source','provided'),
                'seconds':seconds,**plan_metrics(plan,boxes),
                **box_metrics(boxes,obs['target'],obs['viewport'],(record['current']['nodes'][0]['id'],)),
                'executions':stats['executions'],'candidates':stats['candidates']})
            write_json(work/f'{method}-trace.json',stats)
        write_jsonl(out/'metrics.jsonl',rows)
        print(f'[{index+1}/{len(records)}] {record["id"]}: plan evaluation complete',flush=True)
    summary={m:{k:statistics.mean(r[k] for r in rows if r['method']==m)
        for k in ('box_iou','center_error','size_error','relation_accuracy','plan_loss','plan_satisfaction','seconds','executions')}
        for m in methods}
    write_json(out/'summary.json',{'n':len(records),'results':summary,
        'warning':'Oracle teacher plans are a policy diagnostic, not screenshot-to-HTML performance.'})
    return rows


def build_html_data(args):
    import random
    from .html_bridge import HtmlBrowser,embed_placeholder
    from .plans import dom_tree
    from .html_feedback import refresh_geometry,action_unit,GEOMETRY_FIELDS
    from .ir import LIMITS
    from .web_experiment import digest
    sources=list(read_jsonl(args.manifest));rng=random.Random(args.seed)
    if not sources or args.per_page<1:raise ValueError('Need training HTML and positive per-page')
    groups={}
    for source in sources:
        if source.get('split') not in ('train','val') or not source.get('group'):
            raise ValueError('Training manifest requires explicit group and split=train|val; never supply benchmark test pages')
        if source['group'] in groups and groups[source['group']]!=source['split']:raise ValueError('Group split leakage')
        groups[source['group']]=source['split']
    hashes={}
    for source in sources:
        sha=digest(source['html'])
        if sha in hashes and hashes[sha]!=source['split']:raise ValueError('Identical HTML crosses splits')
        hashes[sha]=source['split']
    out=Path(args.out).resolve();records=[];failures=[]
    with HtmlBrowser() as browser:
        for index,source in enumerate(sources):
            work=out/f'page-{index:05d}';work.mkdir(parents=True,exist_ok=True)
            viewport=source.get('viewport',[args.width,args.height]);raw=Path(source['html']).read_text()
            asset=Path(source['html']).parent/'rick.jpg'
            if asset.exists():raw=embed_placeholder(raw,asset)
            try:
                dom=browser.snapshot(raw,viewport,work/'target.png',args.max_nodes-1)
                tree,boxes,_=dom_tree(dom)
                html=dom['html'];root=tree['nodes'][0]['id'];ids=[n['id'] for n in tree['nodes'][1:]]
                plan=oracle_plan(tree,boxes,viewport)
                base={'group':source['group'],'split':source['split'],'source':'browser_teacher',
                    'plan':plan,'observations':[{'viewport':viewport,'target':boxes}],
                    'target_kind':'oracle_plan_training','source_html_sha256':digest(source['html'])}
                records.append({**base,'id':f'html-{index}-clean','current':tree,'current_boxes':boxes,'teacher_edits':[]})
                accepted=0
                for attempt in range(args.per_page*20):
                    if accepted>=args.per_page:break
                    i=rng.randrange(1,len(tree['nodes']));field=rng.choice(GEOMETRY_FIELDS)
                    delta=rng.choice((-1,1))*rng.choice((16,32,64))
                    browser.edit_property(html,viewport,tree['nodes'][i]['id'],field,delta)
                    corrupted_html=browser.page.content();observed=browser.tagged_boxes(ids);observed[root]=[0,0,*viewport]
                    current=refresh_geometry(tree,observed);old=current['nodes'][i]['props'][field]
                    axis={'width':2,'height':3,'dx':0,'dy':1}[field]
                    unit=action_unit(current,observed,i,field);lo,hi=LIMITS[field]
                    value=max(lo,min(hi,round(old+(boxes[tree['nodes'][i]['id']][axis]-observed[tree['nodes'][i]['id']][axis])/unit)))
                    before=plan_metrics(plan,observed)['plan_loss']
                    if value==old or before<1e-6:continue
                    browser.edit_property(corrupted_html,viewport,tree['nodes'][i]['id'],field,(value-old)*unit)
                    after=browser.tagged_boxes(ids);after[root]=[0,0,*viewport]
                    if plan_metrics(plan,after)['plan_loss']>=before-1e-6:continue
                    # Keep only actions actually verified by Chromium, not inverse-CSS guesses.
                    records.append({**base,'id':f'html-{index}-{accepted}','current':current,
                        'current_boxes':observed,'teacher_edits':[[i,field,value]]})
                    (work/f'corrupt-{accepted}.html').write_text(corrupted_html)
                    accepted+=1
                if accepted<args.per_page:failures.append({'id':index,'warning':f'Only {accepted}/{args.per_page} verified corruption examples'})
            except Exception as error:failures.append({'id':index,'error':str(error)})
            for split in ('train','val'):write_jsonl(out/f'{split}.jsonl',[r for r in records if r['split']==split])
            print(f'[{index+1}/{len(sources)}] browser teacher examples={len(records)}',flush=True)
    write_json(out/'report.json',{'records':len(records),'failures':failures,'source_manifest_sha':digest(args.manifest),
        'settings':vars(args),'note':'Oracle plans for training only. Frozen visual planner quality must be evaluated separately.'})
    return records

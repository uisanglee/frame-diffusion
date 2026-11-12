"""Locate first Box-IoU-worsening replayed edit; compare policies and finite CSS alternatives.

Reference-guided search is a diagnostic feasibility witness, NOT a repair baseline.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import re
from pathlib import Path

from .ir import read_jsonl, write_json, write_jsonl
from .tree_edits import FIELDS
from .policy_cause_audit import legality

DEFAULT_CASES=[('design2code','llava-next-7b','10210'),
               ('design2code','llava-next-7b','1002'),
               ('webui','ui2coden','1655906481949/default_1280-720')]


@contextmanager
def audit_browsers():
    # Every edit reloads the saved pre-edit HTML. Snapshot may retag its page,
    # but cannot mutate the saved string or the policy's owner identities.
    from .html_bridge import HtmlBrowser
    with HtmlBrowser() as browser:
        yield browser,browser


def first_decrease(scores,tolerance=1e-8):
    return next((i for i in range(1,len(scores)) if scores[i]<scores[i-1]-tolerance),None)


def candidate_edits(state,owners,priority_owner,reference,limits,tokenizer,maximum=120):
    """Round-robin owners/fields; bounded numeric proposals, no target/current ID alignment."""
    queues=[]
    order=[priority_owner]+[i for i in range(1,len(state)) if i!=priority_owner]
    dimensions={f:sorted(set(float(n['box'][axis]) for n in reference['nodes'] if n['box'][axis]>0))
                for f,axis in [('width',2),('height',3)]}
    for owner in order:
        for j,field in enumerate(FIELDS):
            value,priority=state[owner][j]
            match=re.fullmatch(r'([+-]?[\d.]+)px',value or '')
            if not match:continue
            try:old=float(match[1])
            except ValueError:continue
            numbers=[old*s for s in (.9,1.1,.75,1.25,.5,1.5)]
            numbers += [old+d for d in (-4,4,-16,16)]
            if field in dimensions:
                numbers+=sorted(dimensions[field],key=lambda x:abs(x-old))[:6]
            else:numbers=[0,old-8,old+8,*numbers]
            q=[]
            for number in numbers:
                if not math.isfinite(number) or abs(number-old)<1e-6:continue
                edit=[owner,field,f'{number:.4f}'.rstrip('0').rstrip('.')+'px',priority]
                if edit not in q and legality(tokenizer,edit,state,limits)['allowed']:q.append(edit)
            if q:queues.append(q)
    result=[]
    while queues and len(result)<maximum:
        remaining=[]
        for q in queues:
            if len(result)==maximum:break
            result.append(q.pop(0))
            if q:remaining.append(q)
        queues=remaining
    return result


def context(browser,html,source):
    from .visual import annotate
    from .html_feedback import refresh_geometry
    from .explicit_html import feedback_metadata
    from . import css_owners
    tree=copy.deepcopy(source['current']);viewport=source['viewport']
    browser.load(html,viewport);annotate(browser,tree,visual_boxes=True)
    boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
    boxes[tree['nodes'][0]['id']]=[0,0,*viewport]
    tree=refresh_geometry(tree,boxes)
    parsed=css_owners.read(browser,tree)
    return dict(current=tree,current_boxes=boxes,viewport=viewport,
                css_owners=parsed['owners'],declaration_state=parsed['state'],
                **feedback_metadata(browser,tree))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='runs/paper-color-pilot10-seed42-n2-v1')
    p.add_argument('--trace-root',default='runs/paper-color-pilot10-seed42-n2-target-budget-diagnosis-v1')
    p.add_argument('--out',default='runs/vlm-first-edit-audit-v1')
    p.add_argument('--stage1',default='runs/webui-color-images-v1/7000/tuide-s-abstract/replacement/abstract-stage1/best.pt')
    p.add_argument('--stage2',default='runs/webui-color-images-v1/7000/tuide-s-abstract/replacement/abstract-stage2/best.pt')
    p.add_argument('--detector',default='runs/webui-color-images-v1/7000/detector/best.pt')
    p.add_argument('--device',default='cuda');p.add_argument('--seed',default='42')
    p.add_argument('--threshold',type=float,default=.4)
    p.add_argument('--candidates',type=int,default=120)
    p.add_argument('--max-nodes',type=int,default=2048)
    p.add_argument('--cases',help='JSON list of [dataset,model,page_id]; defaults to three known failures')
    args=p.parse_args()
    if args.candidates<1:p.error('--candidates must be positive')
    import torch
    from .html_bridge import HtmlBrowser
    from .visual import load_policy
    from .visual_train import load_detector,detect,detector_input
    from .visual_experiment import rollout
    from .repair_budget_diagnosis import reference_elements
    from .tree_policy import batch_inputs
    from .metrics import hungarian_box_metrics
    from . import css_owners
    from .web_experiment import digest,guard_run
    from .scroll_guard import snapshot,compare
    cases=json.loads(Path(args.cases).read_text()) if args.cases else DEFAULT_CASES
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    sources=[];fingerprints={}
    for dataset,model,identity in cases:
        path=Path(args.root)/dataset/model/'seeds'/args.seed/'prepare/prepared.jsonl'
        source=next((r for r in read_jsonl(path) if r['id']==identity),None)
        if source is None:raise ValueError(f'Missing saved initial: {dataset}/{model}/{identity}')
        if source['methods']['initial'].get('failed'):raise ValueError(f'Initial generation failed: {identity}')
        sources.append((dataset,model,source));fingerprints[str(path)]=digest(path)
        for key in ('tagged_html','screenshot','reference_observation','html'):
            fingerprints[source[key]]=digest(source[key])
        for file in (Path(args.trace_root)/dataset/model/'seeds'/args.seed/'budget-20/repair/pages').glob('*/*-r1-trace.json'):
            fingerprints[str(file)]=digest(file)
    guard_run(out,dict(contract='vlm-first-worsening-edit-v1',settings=vars(args),sources=fingerprints,
                       checkpoints={p:digest(p) for p in (args.stage1,args.stage2,args.detector)}),True)
    nets={}
    for stage,path in [('stage1',args.stage1),('stage2',args.stage2)]:
        net,ck=load_policy(path,args.device)
        if not net.cfg.semantic or not net.cfg.stylesheets:raise ValueError('Semantic CSS-owner policies required')
        nets[stage]=net;del ck
    detector,ck=load_detector(args.detector,args.device)
    for net in nets.values():
        if ck['config']['visual_geometry_contract']!=net.cfg.visual_geometry_contract:raise ValueError('Visual label contracts differ')
    del ck
    results=[];errors=[]
    with audit_browsers() as (browser,evaluator),torch.inference_mode():
        for dataset,model,source in sources:
            identity=source['id'];page_key=hashlib.sha256(f'{dataset}/{model}/{identity}'.encode()).hexdigest()[:16]
            work=out/dataset/model/page_key;work.mkdir(parents=True,exist_ok=True)
            try:
                reference=json.loads(Path(source['reference_observation']).read_text())
                oracle=reference_elements(source,browser,out/'reference-label-cache')
                predicted=detect(detector,detector_input(source['screenshot']),args.threshold)
                def score(html,directory):
                    from .html_bridge import embed_placeholder
                    placeholder=Path(source['html']).parent/'rick.jpg'
                    html=embed_placeholder(html,placeholder if placeholder.is_file() else None)
                    directory.mkdir(parents=True,exist_ok=True)
                    observed=evaluator.snapshot(html,source['viewport'],directory/'render.png',args.max_nodes)
                    geometry=hungarian_box_metrics({n['id']:n['box'] for n in observed['nodes']},
                        {n['id']:n['box'] for n in reference['nodes']},source['viewport'],
                        tuple(n['id'] for n in observed['nodes'] if n['role']=='body'),
                        tuple(n['id'] for n in reference['nodes'] if n['role']=='body'))
                    (directory/'output.html').write_text(html)
                    write_json(directory/'geometry.json',geometry)
                    return geometry['box_iou']
                # Production zero-step path performs exactly the safe-size normalization.
                initial,stats,_=rollout(browser,Path(source['tagged_html']).read_text(),source['current'],
                    source['viewport'],source['screenshot'],nets['stage2'],None,args.threshold,
                    steps=0,oracle_elements=oracle,pixel_goal_fraction=-1)
                if stats['failed']:raise ValueError(stats['error'])
                write_json(work/'normalization.json',stats)
                for condition,method in [('detector','abstract-policy'),('oracle','abstract-oracle')]:
                    dest=work/condition;dest.mkdir(parents=True,exist_ok=True)
                    cache=dest/'result.json'
                    if cache.exists():
                        results.append(json.loads(cache.read_text()));continue
                    base=Path(args.trace_root)/dataset/model/'seeds'/args.seed/'budget-20/repair/pages'
                    trace=None;trace_path=None
                    for file in sorted(base.glob('*/result.json')):
                        saved=json.loads(file.read_text())
                        if any(t['id']==identity and t['method']==method for t in saved['timings']):
                            trace_path=file.parent/f'{method}-r1-trace.json'
                            trace=json.loads(trace_path.read_text());break
                    if trace is None:raise ValueError(f'Missing existing {condition} 20-step trace for {identity}')
                    html=initial;previous=score(html,dest/'step-0');trajectory=[];bad=None
                    for h in trace['history']:
                        if not h.get('accepted') or not h.get('action'):continue
                        row=context(browser,html,source)
                        # Stop if historical conditioning cannot be faithfully replayed.
                        mismatch=max((max(abs(a-b) for a,b in zip(box,h['boxes'][key]))
                                      for key,box in row['current_boxes'].items() if key in h['boxes']),default=0)
                        if mismatch>1:raise ValueError(f'Replay geometry mismatch at step {h["step"]}: {mismatch}px')
                        action=h['action'];owner=row['css_owners'][action[0]]
                        if h.get('css_owner') and owner!=h['css_owner']:raise ValueError('Replay CSS owner mismatch')
                        css_owners.execute(browser,row['css_owners'],action)
                        candidate=browser.page.content();value=score(candidate,dest/f'step-{h["step"]+1}')
                        event=dict(step=h['step']+1,action=action,before_iou=previous,after_iou=value,delta=value-previous)
                        trajectory.append(event)
                        if first_decrease([previous,value]) is not None:
                            bad=dict(event,html_before=html,context=row);break
                        html=candidate;previous=value
                    write_json(dest/'trajectory.json',trajectory)
                    result=dict(dataset=dataset,model=model,id=identity,condition=condition,trace=str(trace_path),
                                trace_sha=digest(trace_path),historical_scroll_rollback=trace.get('scroll_guard_rolled_back'),
                                first_worsening=None,policy_comparison=[],search={})
                    if bad is not None:
                        print(f'{identity} {condition}: first worsening at step {bad["step"]}',flush=True)
                        base_html=bad.pop('html_before');row=bad.pop('context');result['first_worsening']=bad
                        (dest/'before-wrong-edit.html').write_text(base_html)
                        row.update(target_elements=oracle,lazy_predicted_elements=predicted)
                        state=row['declaration_state']
                        # All four predictions receive exactly the same pre-failure state.
                        for stage,net in nets.items():
                            for target in ('oracle','detector'):
                                inputs=batch_inputs(net,[row],args.device,target=='detector')
                                action=net.predict(*inputs)
                                browser.load(base_html,source['viewport']);css_owners.execute(browser,row['css_owners'],action)
                                value=score(browser.page.content(),dest/f'{stage}-{target}')
                                result['policy_comparison'].append(dict(stage=stage,target=target,action=action,
                                    iou=value,delta=value-bad['before_iou'],same_as_failed_edit=action==bad['action'],
                                    evaluation_scope='static capture before application transfer/scroll rollback'))
                        inputs=batch_inputs(nets['stage2'],[row],args.device,False)
                        limits=inputs[0].get('explicit_owner_limits',[{}])[0]
                        proposals=candidate_edits(state,row['css_owners'],bad['action'][0],reference,limits,
                                                 nets['stage2'].tokenizer,args.candidates)
                        trials=[]
                        for i,action in enumerate(proposals):
                            if i%20==0:print(f'  CSS search {i}/{len(proposals)}',flush=True)
                            trial=dict(action=action)
                            try:
                                browser.load(base_html,source['viewport']);css_owners.execute(browser,row['css_owners'],action)
                                value=score(browser.page.content(),dest/'search'/str(i))
                                trial.update(iou=value,delta=value-bad['before_iou'],directory=str(dest/'search'/str(i)))
                            except Exception as exc:trial['error']=str(exc)
                            trials.append(trial)
                        write_json(dest/'search-trials.json',trials)
                        improving=sorted([t for t in trials if t.get('delta',0)>1e-8],key=lambda t:-t['delta'])
                        witness=None
                        # Validate an improving static edit under scroll/application constraints.
                        for trial in improving:
                            try:
                                browser.load(base_html,source['viewport']);before_scroll=snapshot(browser)
                                css_owners.execute(browser,row['css_owners'],trial['action'])
                                edited=browser.page.content();qa=compare(before_scroll,snapshot(browser))
                                if not qa['accepted']:trial['guard_error']='scroll_preservation';continue
                                app=source['methods']['initial']
                                if app.get('application_html'):
                                    from .application_repair import restore_application
                                    restored=restore_application(app,edited,source['viewport'],dest/'witness-app',browser_host=browser)
                                    edited=restored['html']
                                final=score(edited,dest/'witness')
                                if final>bad['before_iou']+1e-8:
                                    witness=dict(action=trial['action'],iou=final,delta=final-bad['before_iou'],
                                                 scroll_verified=True,application_verified=bool(app.get('application_html')))
                                    break
                            except Exception as exc:trial['guard_error']=str(exc)
                        write_json(dest/'search-trials.json',trials)
                        result['search']=dict(tested=len(trials),errors=sum('error' in t for t in trials),
                            static_improving=len(improving),witness=witness,
                            conclusion='improving_edit_found' if witness else 'inconclusive_bounded_search',
                            scope='One edit from this pre-failure state; does not establish complete layout recoverability')
                    write_json(cache,result);results.append(result);write_jsonl(out/'results.jsonl',results)
            except Exception as exc:
                error=dict(dataset=dataset,model=model,id=identity,error=str(exc));errors.append(error)
                print(error,flush=True);write_jsonl(out/'errors.jsonl',errors)
    write_jsonl(out/'results.jsonl',results)
    write_jsonl(out/'errors.jsonl',errors)
    lines=['# VLM first-worsening-edit audit','',
      'Wrong means a per-step decrease in reference DOM Box IoU. The search uses reference information for diagnosis only. No improvement found does NOT mean CSS repair is impossible.',
      'Historical raw edits are replayed before end-of-rollout scroll rollback/application transfer. Those flags are recorded separately.',
      '', '| Dataset | Model | Page | Replay | First drop step | Delta IoU | Verified improving alternative |',
      '|---|---|---|---|---:|---:|---|']
    for r in results:
        bad=r['first_worsening'];w=r['search'].get('witness')
        lines.append(f'| {r["dataset"]} | {r["model"]} | {r["id"]} | {r["condition"]} | {bad["step"] if bad else "none"} | {bad["delta"] if bad else "--"} | {w["action"] if w else "not established"} |')
    lines+=['',f'Errors: {len(errors)}. Read policy_comparison in results.jsonl for Stage1/Stage2 actions on the identical state.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    with (out/'error.log').open('w') as f:
        for name in ('README.md','results.jsonl','errors.jsonl'):
            path=out/name
            if path.exists():f.write(f'\n=== {name} ===\n'+path.read_text())
    print(out/'error.log',flush=True)

if __name__=='__main__':main()

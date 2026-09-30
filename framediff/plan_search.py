"""Plan-only candidate selection shared by proxy IR and real CSS feedback."""
import hashlib
import time
import torch
from .ir import FIELDS,LIMITS,editable,apply_edit,execute,write_json
from .model import collate,flat_logits
from .plans import encode_plan,plan_metrics
from .html_feedback import refresh_geometry,action_unit,frame_png,GEOMETRY_FIELDS


def proposals(tree, boxes, plan, model, method, topk, html=False):
    if method=='model':
        if model is None or model.cfg.conditioning!='plan':raise ValueError('A newly trained plan checkpoint is required')
        f=encode_plan(tree,plan,model.cfg.max_nodes,boxes)
        if html:
            for j,field in enumerate(FIELDS):
                if field not in GEOMETRY_FIELDS:f['legal'][:,j,:]=False
            f['legal'][0]=False
        with torch.inference_mode():
            logits=flat_logits(model(collate([f],next(model.parameters()).device)))[0]
        result=[];stop=len(tree['nodes'])*len(FIELDS)*257
        for a in torch.topk(logits,min(topk,logits.numel())).indices.tolist():
            if a==stop or not torch.isfinite(logits[a]):continue
            i,fv=divmod(a,len(FIELDS)*257);j,v=divmod(fv,257)
            result.append((i,FIELDS[j],v))
        return result
    if method!='coordinate':raise ValueError('Unknown plan proposal method')
    # Plan-residual priority, but no oracle target coordinate proposals.
    f=encode_plan(tree,plan,max(128,len(tree['nodes'])),boxes)
    priority=f['plan'].reshape(len(tree['nodes']),-1,6)[:,:,3].abs().sum(1)
    locations=sorted(editable(tree),key=lambda p:(-float(priority[p[0]]),p[0],FIELDS.index(p[1])))
    locations=[(i,field) for i,field in locations if not html or (i>0 and field in GEOMETRY_FIELDS)]
    rows=[]
    for i,field in locations:
        old=tree['nodes'][i]['props'][field];lo,hi=LIMITS[field]
        values=range(lo,hi+1) if field in ('flow','align','columns') else [old+d for d in (-4,4,-1,1,-16,16)]
        rows.append([(i,field,v) for v in values if lo<=v<=hi and v!=old])
    # Evaluate same topk/budget for both policies. Avoid exhausting budget on one node.
    from itertools import zip_longest
    return [e for group in zip_longest(*rows) for e in group if e is not None][:topk]


def search(initial,plan,observe,mutate,model=None,method='model',steps=10,beam=2,topk=8,budget=160,html=False,trace_dir=None):
    from pathlib import Path
    work=Path(trace_dir) if trace_dir else None
    if work:work.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();calls=0;failures=[];log=[];history=[];images=0
    def assess(state):
        nonlocal calls,images
        state=observe(state);calls+=1
        state.update(plan_metrics(plan,state['boxes']))
        # Abstract raster is an audit representation of exactly the scored boxes.
        state['png']=frame_png(state['tree'],state['boxes'],plan['viewport'],max_pixels=262144);images+=1
        return state
    def save(state,step):
        item={'step':step,'plan_loss':state['plan_loss'],'plan_satisfaction':state['plan_satisfaction'],
              'boxes':state['boxes'],'edits':state['path'],'executions':calls}
        history.append(item)
        if work:
            write_json(work/f'step-{step:03d}.json',item)
            (work/f'step-{step:03d}.png').write_bytes(state['png'])
    state=assess(initial);state['path']=[];states=[state];save(state,0);attempted=0;reason='steps'
    for step in range(1,steps+1):
        if states[0]['plan_loss']<1e-6:reason='plan_satisfied';break
        candidates=list(states);old=states[0]['plan_loss'];previous=attempted
        for state in states:
            edits=proposals(state['tree'],state['boxes'],plan,model,method,topk,html)
            for edit in edits:
                if attempted>=budget:break
                attempted+=1
                try:
                    candidate=assess(mutate(state,edit))
                    candidate['path']=state['path']+[list(edit)];candidates.append(candidate)
                    log.append({'step':step,'edit':list(edit),'plan_loss':candidate['plan_loss'],
                                'changed_nodes':[k for k,b in candidate['boxes'].items() if max(abs(v-t) for v,t in zip(b,state['boxes'][k]))>.05]})
                except Exception as error:failures.append({'edit':list(edit),'error':str(error)})
        candidates.sort(key=lambda s:s['plan_loss']);seen=set();states=[]
        for candidate in candidates:
            key=candidate.get('html') or str(candidate['tree'])
            key=hashlib.sha256(key.encode()).hexdigest()
            if key in seen:continue
            seen.add(key);states.append(candidate)
            if len(states)>=beam:break
        save(states[0],step)
        if attempted>=budget:reason='budget';break
        if previous==attempted:reason='no_proposals';break
        if beam==1 and states[0]['plan_loss']>=old-1e-10:reason='no_improvement';break
    return states[0],{'history':history,'edits':states[0]['path'],'candidate_log':log,
        'candidates':attempted,'candidate_failures':failures,'executions':calls,'stop_reason':reason,
        'seconds':time.perf_counter()-started,'feedback_images':images,'browser_screenshots':0,
        'model_input':'current_geometry_and_fixed_plan_constraints','render_mode':'frames',
        'plan_loss':states[0]['plan_loss'],'plan_satisfaction':states[0]['plan_satisfaction'],
        'plan_node_coverage':states[0]['plan_node_coverage']}


def repair_ir(tree,plan,**kwargs):
    def observe(state):return {**state,'boxes':execute(state['tree'],plan['viewport'])}
    def mutate(state,edit):return {'tree':apply_edit(state['tree'],edit)}
    return search({'tree':tree},plan,observe,mutate,**kwargs)


def repair_html(browser,html,tree,plan,**kwargs):
    root=tree['nodes'][0]['id'];ids=[n['id'] for n in tree['nodes'][1:]]
    before=browser.executions;shots=browser.screenshots
    def observe(state):
        if not state.pop('_already_loaded',False):browser.load(state['html'],plan['viewport'])
        boxes=browser.tagged_boxes(ids);boxes[root]=[0,0,*plan['viewport']]
        return {**state,'boxes':boxes,'tree':refresh_geometry(state['tree'],boxes)}
    def mutate(state,edit):
        i,field,value=edit
        delta=(value-state['tree']['nodes'][i]['props'][field])*action_unit(state['tree'],state['boxes'],i,field)
        browser.edit_property(state['html'],plan['viewport'],state['tree']['nodes'][i]['id'],field,delta)
        return {'tree':state['tree'],'html':browser.page.content(),'_already_loaded':True}
    result,stats=search({'tree':tree,'html':html},plan,observe,mutate,html=True,**kwargs)
    stats.update(browser_executions=browser.executions-before,browser_screenshots=browser.screenshots-shots,
                 proxy_executions=0,feedback_image_seconds=None)
    return result,stats

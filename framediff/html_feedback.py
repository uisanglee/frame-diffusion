"""Closed-loop repair on the real DOM. Every candidate is laid out by Chromium.

There is no proxy geometry in the decision loop. The existing denoiser consumes
measured boxes (not raster pixels); named frame rasters are optional observations
for auditing and cost comparison. Only one element/property is edited per action.
"""
import copy
import hashlib
import io
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .ir import FIELDS, LIMITS, write_json
from .metrics import objective
from .model import encode, collate, flat_logits

GEOMETRY_FIELDS = ('width','height','dx','dy')


def refresh_geometry(tree, boxes):
    """Quantized model features only; keep fixed identities/ancestry and real boxes separately."""
    tree = copy.deepcopy(tree)
    for n in tree['nodes'][1:]:
        x,y,w,h = boxes[n['id']]; px,py,pw,ph = boxes[n['parent']]
        values = {'width':w/max(pw,1)*256,'height':h/4,
                  'dx':(x-px)/max(pw,1)*256,'dy':(y-py)/max(ph,1)*256}
        for f,v in values.items():
            lo,hi = LIMITS[f]; n['props'][f] = max(lo,min(hi,round(v)))
    return tree


def action_unit(tree, boxes, i, field):
    n = tree['nodes'][i]
    if field == 'height': return 4.
    parent = boxes[n['parent']]
    return max(1.,parent[3 if field == 'dy' else 2])/256


def propose_edits(model, tree, observations, boxes, k, method):
    if method == 'model':
        features = encode(tree,observations,model.cfg.max_nodes,[boxes])
        for j,field in enumerate(FIELDS):
            if field not in GEOMETRY_FIELDS: features['legal'][:,j,:] = False
        features['legal'][0,:,:] = False
        device = next(model.parameters()).device
        with torch.inference_mode(): logits = flat_logits(model(collate([features],device)))[0]
        choices = torch.topk(logits,min(k,logits.numel())).indices.tolist()
        stop = len(tree['nodes'])*len(FIELDS)*257
        result = []
        for a in choices:
            if a == stop: continue
            i,fv = divmod(a,len(FIELDS)*257); f,v = divmod(fv,257)
            if torch.isfinite(logits[a]): result.append((i,FIELDS[f],v))
        return result
    target = observations[0]['target']; directed = []; nearby = []
    for i,n in enumerate(tree['nodes'][1:],1):
        b = boxes[n['id']]; t = target[n['id']]
        for f,axis in (('width',2),('height',3),('dx',0),('dy',1)):
            old = n['props'][f]; lo,hi = LIMITS[f]
            residual = t[axis]-b[axis]
            direct = max(lo,min(hi,round(old+residual/action_unit(tree,boxes,i,f))))
            if direct != old: directed.append((abs(residual),(i,f,direct)))
            for d in (-1,1,-4,4):
                if lo<=old+d<=hi: nearby.append((i,f,old+d))
    result = []; seen = set()
    for edit in [e for _,e in sorted(directed,key=lambda x:-x[0])] + nearby:
        if edit not in seen: result.append(edit); seen.add(edit)
        if len(result)>=k: break
    return result


def frame_png(tree, boxes, viewport):
    image = Image.new('RGB',tuple(viewport),'white'); draw = ImageDraw.Draw(image)
    for n in tree['nodes'][1:]:
        x,y,w,h = boxes[n['id']]
        if w<=0 or h<=0: continue
        color = '#'+hashlib.sha256(n['id'].encode()).hexdigest()[:6]
        draw.rectangle((x,y,x+w,y+h),outline=color,width=2)
        label = f"{n['id']} {n.get('name',n['role'])}"[:65].replace('\n',' ')
        draw.text((max(0,x)+2,max(0,y)+2),label,fill=color)
    stream = io.BytesIO(); image.save(stream,format='PNG'); return stream.getvalue()


def repair_html(browser, html, tree, observations, model=None, method='model', steps=10,
                beam=2, topk=8, budget=160, render_mode='frames', trace_dir=None):
    if len(observations)!=1: raise ValueError('HTML feedback currently requires one viewport')
    if render_mode not in ('boxes','frames','raster'): raise ValueError('Unknown feedback render mode')
    if method not in ('model','coordinate'): raise ValueError('Unknown feedback proposal method')
    viewport = observations[0]['viewport']; root = tree['nodes'][0]['id']
    ids = [n['id'] for n in tree['nodes'][1:]]
    if set(observations[0]['target']) != {root,*ids}: raise ValueError('Target IDs must cover all tracked nodes')
    before = browser.executions; screenshots_before = browser.screenshots
    started = time.perf_counter(); feedback_seconds = 0.; layout_seconds = 0.; feedback_images = 0
    candidate_count = 0; failures = []; candidate_log = []; history = []
    trace_dir = Path(trace_dir) if trace_dir else None
    if trace_dir: trace_dir.mkdir(parents=True,exist_ok=True)

    def observe(current_tree):
        nonlocal feedback_seconds, feedback_images
        boxes = browser.tagged_boxes(ids); boxes[root] = [0,0,*viewport]
        current_tree = refresh_geometry(current_tree,boxes)
        t = time.perf_counter(); png = None
        if render_mode == 'frames': png = frame_png(current_tree,boxes,viewport)
        elif render_mode == 'raster':
            png = browser.page.screenshot(animations='disabled'); browser.screenshots += 1
        if png is not None: feedback_images += 1
        feedback_seconds += time.perf_counter()-t
        return {'tree':current_tree,'html':browser.page.content(),'boxes':boxes,'image':png,
                'score':objective(boxes,observations[0]['target'],viewport,root)}

    def save_state(state, step):
        item = {'step':step,'score':state['score'],'boxes':state['boxes'],'edits':state['path'],
                'browser_executions':browser.executions-before,'candidates':candidate_count}
        history.append(item)
        if trace_dir:
            write_json(trace_dir/f'step-{step:03d}.json',item)
            (trace_dir/f'step-{step:03d}.html').write_text(state['html'])
            if state['image']: (trace_dir/f'step-{step:03d}.png').write_bytes(state['image'])

    t = time.perf_counter(); browser.load(html,viewport)
    initial = observe(tree); initial['path'] = []
    layout_seconds += time.perf_counter()-t
    states = [initial]; save_state(initial,0); reason = 'steps'
    for step in range(1,steps+1):
        if states[0]['score']<1e-6: reason='matched'; break
        candidates = list(states); previous_score = states[0]['score']; attempted = 0
        for state in states:
            edits = propose_edits(model,state['tree'],observations,state['boxes'],topk,method)
            for i,field,value in edits:
                if candidate_count>=budget: break
                if i<=0 or field not in GEOMETRY_FIELDS: raise ValueError('Unsupported proposal')
                n = state['tree']['nodes'][i]; old = n['props'][field]
                lo,hi = LIMITS[field]
                if not lo<=value<=hi: raise ValueError('Proposal value out of bounds')
                delta = (value-old)*action_unit(state['tree'],state['boxes'],i,field)
                candidate_count += 1; attempted += 1
                t = time.perf_counter()
                try:
                    browser.edit_property(state['html'],viewport,n['id'],field,delta)
                    candidate = observe(state['tree'])
                    changed = [nid for nid in ids if max(abs(a-b) for a,b in zip(candidate['boxes'][nid],state['boxes'][nid]))>.05]
                    edit = {'node':n['id'],'field':field,'value':value,'delta_px':delta,
                            'changed_nodes':changed,'score':candidate['score']}
                    candidate['path'] = state['path']+[edit]; candidates.append(candidate)
                    candidate_log.append({'step':step,**edit})
                except Exception as error:
                    failures.append({'step':step,'node':n['id'],'field':field,'error':str(error)})
                layout_seconds += time.perf_counter()-t
            if candidate_count>=budget: break
        candidates.sort(key=lambda x:x['score'])
        selected = []; seen = set()
        for state in candidates:
            # HTML, not quantized IR/boxes, defines a state: CSS can have latent effects.
            key = hashlib.sha256(state['html'].encode()).hexdigest()
            if key not in seen: seen.add(key); selected.append(state)
            if len(selected)>=beam: break
        states = selected; save_state(states[0],step)
        if candidate_count>=budget: reason='budget'; break
        if attempted==0: reason='no_proposals'; break
        if beam==1 and states[0]['score']>=previous_score-1e-10: reason='no_improvement'; break
    stats = {'feedback':'real_html_browser','render_mode':render_mode,'model_input':'measured_boxes_not_image_pixels',
             'browser_executions':browser.executions-before,'proxy_executions':0,
             'browser_screenshots':browser.screenshots-screenshots_before,'feedback_images':feedback_images,
             'candidates':candidate_count,'candidate_failures':failures,'candidate_log':candidate_log,
             'history':history,'edits':states[0]['path'],'stop_reason':reason,
             'seconds':time.perf_counter()-started,'feedback_image_seconds':feedback_seconds,
             'layout_and_feedback_seconds':layout_seconds}
    return states[0]['html'],states[0]['tree'],stats

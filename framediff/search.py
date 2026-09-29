from __future__ import annotations
import random
from itertools import zip_longest
import torch
from .ir import execute,editable,apply_edit,FIELDS,LIMITS
from .model import encode,collate,flat_logits
from .metrics import objective

class Executor:
    def __init__(self,browser=None):
        self.browser=browser;self.calls=0;self.cache={}
    def frames(self,tree,observations):
        key=tuple(tuple(n['props'][f] for f in FIELDS) for n in tree['nodes'])
        key=(tuple((n['id'],n['parent']) for n in tree['nodes']),key,tuple(tuple(o['viewport']) for o in observations))
        if key not in self.cache:
            values=[]
            for o in observations:
                self.calls+=1
                values.append(self.browser.render(tree,o['viewport']) if self.browser else execute(tree,o['viewport']))
            self.cache[key]=values
        return self.cache[key]
    def score(self,tree,observations):
        boxes=self.frames(tree,observations)
        return sum(objective(b,o['target'],o['viewport'],tree['nodes'][0]['id']) for b,o in zip(boxes,observations))/len(observations)

def proposals(model,tree,observations,k,executor):
    device=next(model.parameters()).device
    features=encode(tree,observations,model.cfg.max_nodes,executor.frames(tree,observations))
    batch=collate([features],device)
    with torch.inference_mode():scores=flat_logits(model(batch))[0]
    stop=len(tree['nodes'])*len(FIELDS)*257
    top=torch.topk(scores,min(k,scores.numel())).indices.tolist()
    result=[]
    for a in top:
        if a==stop:continue
        i,fv=divmod(a,len(FIELDS)*257);f,v=divmod(fv,257)
        if torch.isfinite(scores[a]):result.append((i,FIELDS[f],v))
    return result

def deterministic_proposals(tree,observations):
    """Non-neural finite-difference + target-directed coordinate descent baseline."""
    v=observations[0]['viewport'];boxes=execute(tree,v);target=observations[0]['target']
    by_id={n['id']:n for n in tree['nodes']}
    directed=[];neighborhoods=[]
    def residual(location):
        n=tree['nodes'][location[0]];a=boxes[n['id']];b=target.get(n['id'],a)
        return sum(abs(x-y) for x,y in zip(a,b))
    for i,f in sorted(editable(tree),key=residual,reverse=True):
        n=tree['nodes'][i];old=n['props'][f];lo,hi=LIMITS[f]
        values={old-1,old+1}
        if f in ('width','height','dx','dy'):values|={old-4,old+4,old-16,old+16}
        if f in ('flow','columns','align','order'):values.update(range(lo,min(hi,8)+1))
        if n['id'] in target:
            a=boxes[n['id']];b=target[n['id']]
            if f in ('dx','dy'):
                axis=0 if f=='dx' else 1
                parent=by_id.get(n['parent'])
                unit=4
                if parent and parent['props']['flow']==3:
                    unit=max(1e-6,(boxes[parent['id']][axis+2]-8*parent['props']['padding'])/256)
                direct=round(old+(b[axis]-a[axis])/unit)
            elif f=='height':direct=round(b[3]/4)
            elif f=='width' and a[2]>1e-6:direct=round(old*b[2]/a[2])
            else:direct=old
            if lo<=direct<=hi and direct!=old:directed.append((i,f,direct));values.discard(direct)
        neighborhoods.append([(i,f,val) for val in sorted(values,key=lambda x:(abs(x-old),x)) if lo<=val<=hi and val!=old])
    yield from directed
    # Round robin avoids spending the entire budget on early nodes/properties.
    for row in zip_longest(*neighborhoods):
        yield from (edit for edit in row if edit is not None)

def repair(tree,observations,method='model',model=None,steps=10,beam=2,proposals_per_state=8,budget=500,executor=None,seed=42):
    executor=executor or Executor();rng=random.Random(seed)
    initial=executor.score(tree,observations)
    states=[(initial,tree,[])];history=[{'step':0,'score':initial,'calls':executor.calls}]
    considered=0
    for step in range(1,steps+1):
        candidates=list(states)
        for score,current,path in states:
            if method=='model':edits=proposals(model,current,observations,proposals_per_state,executor)
            else:
                edits=list(deterministic_proposals(current,observations))
                if method=='random':rng.shuffle(edits);edits=edits[:proposals_per_state]
            for edit in edits:
                if considered>=budget:break
                candidate=apply_edit(current,edit);considered+=1
                value=executor.score(candidate,observations)
                candidates.append((value,candidate,path+[list(edit)]))
            if considered>=budget:break
        candidates.sort(key=lambda x:x[0])
        distinct=[];seen=set()
        for item in candidates:
            fingerprint=tuple(tuple(n['props'][f] for f in FIELDS) for n in item[1]['nodes'])
            if fingerprint not in seen:distinct.append(item);seen.add(fingerprint)
            if len(distinct)>=beam:break
        old=states[0][0];states=distinct
        history.append({'step':step,'score':states[0][0],'calls':executor.calls})
        if states[0][0]<1e-6 or considered>=budget:break
        if states[0][0]>=old-1e-10 and beam==1:break
    return states[0][1],{'edits':states[0][2],'history':history,'candidates':considered,'executions':executor.calls}

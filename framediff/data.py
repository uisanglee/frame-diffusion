from __future__ import annotations
import copy
import hashlib
import random
from .ir import node, validate, execute, editable, apply_edit, LIMITS, write_jsonl

VIEWPORTS = [(1024,768),(768,1024),(390,844)]

def group_split(group):
    bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
    return "train" if bucket<80 else "val" if bucket<90 else "test"

def synthetic(seed):
    """Varied small executable component trees, not scraped website ground truth."""
    r=random.Random(seed)
    nodes=[node("page",None,"page",padding=r.choice([2,4,6]),gap=r.choice([2,4,6]))]
    nodes += [node("header","page","header",flow=0,height=r.choice([14,18,22]),gap=4,align=1),
              node("logo","header","image",width=48,height=10),
              node("nav","header","nav",width=192,height=10,order=1,flow=0,gap=2)]
    for i in range(r.randint(2,4)):
        nodes.append(node(f"link{i}","nav","button",width=48,height=8,order=i))
    nodes.append(node("body","page","main",flow=0,order=1,height=132,gap=4))
    sidebar = r.random()<.7
    if sidebar:
        nodes.append(node("sidebar","body","sidebar",width=48,height=120,padding=2,gap=3))
        for i in range(r.randint(2,4)):
            nodes.append(node(f"item{i}","sidebar","text",height=8,order=i))
    nodes.append(node("main","body","section",width=200 if sidebar else 256,height=128,order=1,gap=4,padding=2))
    nodes.append(node("title","main","title",height=r.choice([8,12,16]),width=r.choice([128,192,256])))
    cols=r.choice([2,3,4])
    nodes.append(node("cards","main","list",flow=2,columns=cols,height=100,order=1,gap=r.choice([2,3,4])))
    for i in range(r.randint(cols,cols*2)):
        nodes.append(node(f"card{i}","cards","card",height=r.choice([28,32,36]),order=i,padding=2,gap=1))
        nodes.append(node(f"card{i}-image",f"card{i}","image",height=12))
        nodes.append(node(f"card{i}-text",f"card{i}","text",height=6,order=1,width=r.choice([128,192,256])))
    nodes.append(node("footer","page","footer",height=12,order=2))
    return validate({"version":1,"nodes":nodes})

def differences(current,clean):
    if [(n['id'],n['parent']) for n in current['nodes']] != [(n['id'],n['parent']) for n in clean['nodes']]:
        raise ValueError("Supervised pairs require the same node set, order and topology")
    return [(i,f,clean['nodes'][i]['props'][f]) for i,f in editable(current)
            if current['nodes'][i]['props'][f]!=clean['nodes'][i]['props'][f]]

def corrupt(clean,rng,steps=3,mode="mixed"):
    current=copy.deepcopy(clean)
    allowed=list(editable(clean))
    if mode=="geometry":
        allowed=[(i,f) for i,f in allowed if f in ('width','height','dx','dy')]
    elif mode=="structure":
        allowed=[(i,f) for i,f in allowed if f in ('flow','columns','order','gap','padding','align')]
    if not allowed:
        return current
    for _ in range(steps):
        for attempt in range(30):
            i,f=rng.choice(allowed)
            old=current['nodes'][i]['props'][f]
            lo,hi=LIMITS[f]
            if f in ('flow','columns','align','order'):
                value=rng.randint(lo,hi if f!='order' else 8)
            else:
                step=rng.choice([-1,1])*rng.randint(1,16 if f in ('width','height') else 4)
                value=max(lo,min(hi,old+step))
            if value!=old:
                candidate=apply_edit(current,(i,f,value))
                if execute(candidate)!=execute(current):
                    current=candidate; break
    return current

def make_record(clean,record_id,group,seed,steps=3,viewports=None,mode="mixed"):
    viewports=viewports or VIEWPORTS
    return {"id":record_id,"group":group,"split":group_split(group),"source":"synthetic",
            "target_kind":"oracle_frames","clean":clean,
            "current":corrupt(clean,random.Random(seed),steps,mode),
            "observations":[{"viewport":list(v),"target":execute(clean,v)} for v in viewports]}

def generate(out,count=1000,seed=42):
    records=[]
    for i in range(count):
        clean=synthetic(seed+i)
        records.append(make_record(clean,f"synthetic-{seed+i}",f"synthetic-{seed+i}",seed*100003+i,1+i%5))
    for split in ('train','val','test'):
        selected=[r for r in records if r['split']==split]
        write_jsonl(f"{out}/{split}.jsonl",selected)
    return {s:sum(r['split']==s for r in records) for s in ('train','val','test')}

def assert_disjoint(train,val):
    groups={r['group'] for r in train}&{r['group'] for r in val}
    ids={r['id'] for r in train}&{r['id'] for r in val}
    if groups or ids:
        raise ValueError(f"Train/validation leakage: {len(groups)} groups, {len(ids)} IDs")

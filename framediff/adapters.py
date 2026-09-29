"""WebUI observations -> quantized executable surrogates, NOT original CSS labels."""
import gzip
import json
import math
from pathlib import Path
from urllib.parse import urlparse
from .ir import node, validate, execute, read_json, read_jsonl, write_json, write_jsonl
from .data import make_record, group_split
from .metrics import box_metrics

def load_gzip(path):
    with gzip.open(path,'rt') as f:return json.load(f)

def import_webui(args):
    out=Path(args.out);records=[];rejected=[]
    for path in sorted(Path(args.root).rglob('*axtree.json.gz')):
        if args.limit and len(records)>=args.limit:break
        prefix=path.name.removesuffix('axtree.json.gz')
        try:
            ax=load_gzip(path)['nodes'];bb=load_gzip(path.with_name(prefix+'bb.json.gz'))
            root=next(n for n in ax if n.get('role',{}).get('value')=='RootWebArea')
            rb=bb[str(root['backendDOMNodeId'])];viewport=[round(rb['width']),round(rb['height'])]
            if min(viewport)<=0:raise ValueError('Empty viewport')
            all_nodes={n['nodeId']:n for n in ax};selected={}
            for n in ax:
                b=bb.get(str(n.get('backendDOMNodeId')))
                if n.get('ignored') or not b or min(b['width'],b['height'])<=0:continue
                if b['x']>=viewport[0] or b['y']>=viewport[1] or b['x']+b['width']<=0 or b['y']+b['height']<=0:continue
                selected[n['nodeId']]=(n,b)
            if len(selected)>args.max_nodes:raise ValueError(f'{len(selected)} nodes exceeds limit; no silent truncation')
            nodes=[]
            for nid,(n,b) in selected.items():
                parent=n.get('parentId');seen=set()
                while parent and parent not in selected:
                    if parent in seen:raise ValueError('Cycle')
                    seen.add(parent);parent=all_nodes.get(parent,{}).get('parentId')
                nodes.append({'id':str(nid),'parent':str(parent) if parent else None,
                              'role':n.get('role',{}).get('value','unknown'),
                              'name':n.get('name',{}).get('value',str(nid))[:120],
                              'box':[b[k] for k in ('x','y','width','height')]})
            url_file=path.with_name(prefix+'url.txt')
            url=url_file.read_text().strip() if url_file.exists() else ''
            group=urlparse(url).hostname or path.parent.name
            rid=str(path.relative_to(args.root)).replace('/','__').removesuffix('axtree.json.gz')
            records.append({'id':rid,'group':group,'split':args.split or group_split(group),'source':'webui_accessibility',
                            'url':url,'viewport':viewport,'nodes':nodes})
        except (KeyError,ValueError,OSError,StopIteration) as error:
            rejected.append({'path':str(path),'reason':str(error)})
    write_jsonl(out/'observations.jsonl',records)
    write_json(out/'import-report.json',{'accepted':len(records),'rejected':rejected,
               'note':'AX boxes are observations, not original executable CSS. Fit before supervised training.'})
    print(f'Imported {len(records)} observations; rejected {len(rejected)}')

def fit_observation(obs):
    """Preserve containment ancestry when possible. Explicitly report reparenting."""
    viewport=obs['viewport'];w,h=viewport
    if w<=0 or h<=0:raise ValueError('Invalid viewport')
    original={str(n['id']):n for n in obs['nodes']}
    if len(original)!=len(obs['nodes']):raise ValueError('Duplicate IDs')
    root_id='__viewport__'
    while root_id in original:root_id+='0'
    nodes=[node(root_id,None,'page',flow=3,padding=0,gap=0)];targets={root_id:[0,0,w,h]}
    pending=list(original.values());done={root_id};reparented=0
    while pending:
        progress=False
        for n in pending[:]:
            nid=str(n['id']);parent=n.get('parent')
            if parent is not None:parent=str(parent)
            if parent in original and parent not in done:continue
            box=list(map(float,n['box']));x,y,bw,bh=box
            if any(not math.isfinite(v) for v in box) or min(bw,bh)<=0:raise ValueError('Invalid box')
            parent=parent if parent in done else root_id
            px,py,pw,ph=targets[parent]
            if x<px or y<py or x+bw>px+pw+1 or y+bh>py+ph+1:
                parent=root_id;px,py,pw,ph=targets[parent];reparented+=1
            # Out-of-viewport/oversize boxes may have high fit error and be rejected.
            def q(value):
                return max(0,min(256,round(value)))
            role=str(n.get('role','unknown')).lower()
            role={'rootwebarea':'page','genericcontainer':'container','statictext':'text','heading':'title','img':'image','div':'container','p':'text'}.get(role,role)
            item=node(nid,parent,role,flow=3,padding=0,gap=0,width=max(1,q(bw/pw*256)),
                      height=max(1,q(bh/4)),dx=q((x-px)/pw*256),dy=q((y-py)/ph*256))
            item['name']=str(n.get('name',nid))[:120]
            nodes.append(item);targets[nid]=box;done.add(nid);pending.remove(n);progress=True
        if not progress:raise ValueError('Cyclic observation tree')
    tree=validate({'version':1,'nodes':nodes})
    fidelity=box_metrics(execute(tree,viewport),targets,viewport,exclude=(root_id,))
    return tree,targets,{'metrics':fidelity,'reparented':reparented}

def fit_frames(args):
    path=Path(args.input)
    observations=list(read_jsonl(path)) if path.suffix=='.jsonl' else [read_json(path)]
    records=[];report=[]
    for i,obs in enumerate(observations):
        rid=obs.get('id',f'frames-{i}');group=obs.get('group',rid)
        try:
            tree,original,fidelity=fit_observation(obs)
            accepted=1-fidelity['metrics']['box_iou']<=args.max_error
            report.append({'id':rid,'accepted':accepted,**fidelity})
            if not accepted:continue
            record=make_record(tree,rid,group,args.seed+i,viewports=[obs['viewport']],mode='geometry')
            record.update(source='fitted_frame_surrogate',split=obs.get('split',group_split(group)),
                          noise_mode='geometry',original_target=original,fit=fidelity)
            records.append(record)
        except (ValueError,KeyError,ZeroDivisionError) as error:
            report.append({'id':rid,'accepted':False,'reason':str(error)})
    for split in ('train','val','test'):write_jsonl(Path(args.out)/f'{split}.jsonl',[r for r in records if r['split']==split])
    write_json(Path(args.out)/'fit-report.json',{'accepted':len(records),'total':len(observations),'records':report,
        'note':'Training targets are projected surrogate frames. Original observed boxes are preserved separately; this is not original web CSS recovery.'})
    print(f'Fitted {len(records)}/{len(observations)} observations')

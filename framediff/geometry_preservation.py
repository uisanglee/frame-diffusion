"""Conservative class/box matching, independent of RGB and reference DOM IDs."""
import numpy as np

CONTRACT = 'mutual-box-match-protection-v1'


def match_geometry(target, current, dom_boxes, viewport, tolerance_px=2.,
                   tolerance_fraction=.002, min_iou=.5, ambiguity_margin=.1):
    """Mutual, unambiguous best IoU match of same-class visible primitives.

    Only protect a DOM owner if ALL its visible primitives have matched and
    aligned target primitives. No matches or ambiguous matches imply unknown,
    never correct. Coordinates are viewport CSS pixels; boxes are xyxy.
    """
    if not (np.isfinite(tolerance_px) and tolerance_px>=0 and
            np.isfinite(tolerance_fraction) and 0<=tolerance_fraction<=.1 and
            0<min_iou<=1 and 0<=ambiguity_margin<=1):
        raise ValueError('Invalid geometry protection thresholds')
    tol=np.array([max(tolerance_px,viewport[0]*tolerance_fraction),
                  max(tolerance_px,viewport[1]*tolerance_fraction)]*2)
    n,m=len(current),len(target)
    report=dict(contract=CONTRACT,current_count=n,target_count=m,matches=[],
                protected_ids=[],unknown_current=n,unmatched_target=m,goal=False,
                tolerance_xywh=tol.tolist())
    if not n or not m:return report
    a=np.asarray([x['box'] for x in current],dtype=float)
    b=np.asarray([x['box'] for x in target],dtype=float)
    if a.shape!=(n,4) or b.shape!=(m,4) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid geometry boxes')
    if (a[:,2:]<=a[:,:2]).any() or (b[:,2:]<=b[:,:2]).any():
        raise ValueError('Geometry boxes must have positive visible extent')
    intersection=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    union=(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-intersection
    score=intersection/np.maximum(union,1e-12)
    same=np.asarray([x['label'] for x in current])[:,None]==np.asarray([x['label'] for x in target])[None,:]
    score=np.where(same,score,0.)
    row_best=score.argmax(1);col_best=score.argmax(0)
    row_second=np.partition(score,-2,axis=1)[:,-2] if m>1 else np.zeros(n)
    col_second=np.partition(score,-2,axis=0)[-2,:] if n>1 else np.zeros(m)
    good=set();matched_target=set();matched_current=set()
    for i,j in enumerate(row_best):
        s=float(score[i,j])
        if col_best[j]!=i or s<min_iou or s-row_second[i]<ambiguity_margin or s-col_second[j]<ambiguity_margin:
            continue
        aa=np.r_[a[i,:2],a[i,2:]-a[i,:2]];bb=np.r_[b[j,:2],b[j,2:]-b[j,:2]]
        error=np.abs(aa-bb);aligned=bool((error<=tol).all())
        report['matches'].append(dict(current_index=i,target_index=int(j),owner_id=current[i].get('owner_id'),
                                      iou=s,error_xywh=error.tolist(),aligned=aligned))
        matched_target.add(int(j));matched_current.add(i)
        if aligned:good.add(i)
    owners={}
    for i,item in enumerate(current):
        if item.get('owner_id') is not None:owners.setdefault(item['owner_id'],[]).append(i)
    report['protected_ids']=[id for id,indices in owners.items() if id in dom_boxes and all(i in good for i in indices)]
    report.update(unknown_current=n-len(matched_current),unmatched_target=m-len(matched_target),
                  goal=len(good)==n==m and all(item.get('owner_id') in dom_boxes for item in current))
    return report


def violated_boxes(anchors, boxes, tolerance_px=.25):
    """Lock previously correct DOM geometry, including across successive edits."""
    result=[]
    for id,before in anchors.items():
        after=boxes.get(id)
        if after is None or not np.isfinite(after).all() or np.max(np.abs(np.asarray(after)-before))>tolerance_px:
            result.append(id)
    return result

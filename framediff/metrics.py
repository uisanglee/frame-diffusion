import statistics
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a,b):
    x = max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))
    y = max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    area = x*y
    return area/max(1e-8,a[2]*a[3]+b[2]*b[3]-area)

def box_metrics(pred, target, viewport, exclude=()):
    """Paired, identity-aware metrics; missing elements are never dropped silently."""
    ids = [k for k in target if k not in exclude]
    if not ids:
        raise ValueError("No target elements to evaluate")
    w,h = viewport
    overlaps,centers,sizes = [],[],[]
    for k in ids:
        b = target[k]
        a = pred.get(k)
        if a is None:
            overlaps.append(0.); centers.append(1.); sizes.append(1.); continue
        overlaps.append(iou(a,b))
        centers.append((abs(a[0]+a[2]/2-b[0]-b[2]/2)/w+abs(a[1]+a[3]/2-b[1]-b[3]/2)/h)/2)
        sizes.append((abs(a[2]-b[2])/w+abs(a[3]-b[3])/h)/2)
    def relation(a,b):
        return (a[0]+a[2]<=b[0]+1e-5,b[0]+b[2]<=a[0]+1e-5,
                a[1]+a[3]<=b[1]+1e-5,b[1]+b[3]<=a[1]+1e-5)
    relation_hits=[]
    for i,k in enumerate(ids):
        for j in ids[i+1:]:
            relation_hits.append(int(k in pred and j in pred and relation(pred[k],pred[j])==relation(target[k],target[j])))
    return {"box_iou":statistics.mean(overlaps),"center_error":statistics.mean(centers),
            "size_error":statistics.mean(sizes),"element_success_09":sum(v>=.9 for v in overlaps)/len(ids),
            "page_success_09":float(all(v>=.9 for v in overlaps)),
            "relation_accuracy":statistics.mean(relation_hits) if relation_hits else 1.,
            "recall":sum(k in pred for k in ids)/len(ids),
            "overflow_rate":sum(k not in pred or pred[k][0]<-.1 or pred[k][1]<-.1 or pred[k][0]+pred[k][2]>w+.1 or pred[k][1]+pred[k][3]>h+.1 for k in ids)/len(ids)}

def hungarian_box_metrics(pred,target,viewport,pred_exclude=(),target_exclude=()):
    """Identity-free geometry metric for generated trees vs reference DOM blocks.

    Matching maximizes IoU with a small normalized center-distance tie breaker. This
    is intentionally named differently from the official Design2Code block metric.
    """
    pred_items=[(k,v) for k,v in pred.items() if k not in pred_exclude]
    target_items=[(k,v) for k,v in target.items() if k not in target_exclude]
    if not target_items:raise ValueError('No target elements to evaluate')
    if not pred_items:
        return box_metrics({},dict(target_items),viewport)
    w,h=viewport;cost=np.zeros((len(pred_items),len(target_items)),dtype=np.float64)
    for i,(_,a) in enumerate(pred_items):
        for j,(_,b) in enumerate(target_items):
            center=(abs(a[0]+a[2]/2-b[0]-b[2]/2)/w+abs(a[1]+a[3]/2-b[1]-b[3]/2)/h)/2
            cost[i,j]=1-iou(a,b)+1e-3*center
    rows,cols=linear_sum_assignment(cost)
    # Rename predictions to their assigned reference IDs, then reuse all metrics.
    aligned={target_items[j][0]:pred_items[i][1] for i,j in zip(rows,cols)}
    result=box_metrics(aligned,dict(target_items),viewport)
    result['matched_pairs']=len(rows)
    return result

def objective(pred,target,viewport,root=None):
    m=box_metrics(pred,target,viewport,exclude=(root,))
    return 1-m["box_iou"]+m["center_error"]+m["size_error"]

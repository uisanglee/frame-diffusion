"""Conservative screenshot-based protection; no reference DOM or IoU oracle."""
import math
import numpy as np


def matching_regions(target, current, boxes, tolerance=.005, padding=8):
    a=np.asarray(target,dtype=np.int16);b=np.asarray(current,dtype=np.int16)
    if a.shape!=b.shape:raise ValueError('Preservation requires equal target/current viewport dimensions')
    height,width=a.shape[:2];result={}
    for id,box in boxes.items():
        x,y,w,h=box
        if w<=0 or h<=0:continue
        left=max(0,math.floor(x-padding));top=max(0,math.floor(y-padding))
        right=min(width,math.ceil(x+w+padding));bottom=min(height,math.ceil(y+h+padding))
        if right<=left or bottom<=top:continue
        # Every pixel, not mostly empty background, must match within tolerance.
        if np.max(np.abs(a[top:bottom,left:right]-b[top:bottom,left:right]))<=255*tolerance:
            result[id]=(left,top,right,bottom)
    return result


def violated_regions(target, candidate, regions, tolerance=.005):
    a=np.asarray(target,dtype=np.int16);b=np.asarray(candidate,dtype=np.int16)
    if a.shape!=b.shape:raise ValueError('Preservation candidate viewport changed')
    return [id for id,(l,t,r,d) in regions.items()
            if np.max(np.abs(a[t:d,l:r]-b[t:d,l:r]))>255*tolerance]


def editable_owners(owners, protected, blocked=()):
    # A shared rule is protected only if ALL its matched elements already agree.
    return [i for i,o in enumerate(owners) if i and i not in blocked and
            (not o.get('matches') or not all(id in protected for id in o['matches']))]

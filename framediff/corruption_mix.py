"""Requested corruption profiles and measured same-DOM geometry (not RGB)."""
import math
from collections import Counter

CONTRACT = 'geometry-mixture-v1'
PROFILES = ('clean', 'mild', 'medium', 'strong', 'multi')
# Fractions of viewport axis; measured severity is reported independently.
RANGES = {'mild': (.01, .04), 'medium': (.04, .12), 'strong': (.12, .30), 'multi': (.04, .20)}


def geometry_errors(reference, current, viewport, root=None):
    w,h=viewport; values=[]; missing=0
    for key,a in reference.items():
        if key==root or a[2]<=0 or a[3]<=0: continue
        # Only target elements intersecting the evaluation viewport.
        if a[0]>=w or a[1]>=h or a[0]+a[2]<=0 or a[1]+a[3]<=0: continue
        b=current.get(key)
        if b is None or not all(math.isfinite(x) for x in b): missing+=1;continue
        dx,dy=abs(a[0]-b[0]),abs(a[1]-b[1])
        dw,dh=abs(a[2]-b[2]),abs(a[3]-b[3])
        values.append((max(dx/w,dy/h,dw/w,dh/h),(dx+dy)/2,(dw+dh)/2,
                       (dx/w+dy/h)/2,(dw/w+dh/h)/2,int(max(dx,dy,dw,dh)>1)))
    n=len(values)
    peak=max((v[0] for v in values),default=0.)
    level=('unmeasurable' if not n or missing else 'clean' if not any(v[5] for v in values)
           else 'mild' if peak<=.04 else 'medium' if peak<=.12 else 'strong')
    return dict(measured_level=level,matched_elements=n,missing_elements=missing,
                changed_elements=sum(v[5] for v in values),max_axis_error=peak,
                **{name:sum(v[i] for v in values)/max(n,1) for i,name in enumerate(
                    ('position_mae_px','size_mae_px','position_mae_normalized','size_mae_normalized'),start=1)})


def summary(rows):
    entries=[r['corruption_geometry'] for r in rows if 'corruption_geometry' in r]
    return dict(n=len(entries), requested=dict(Counter(r['requested_profile'] for r in entries)),
                measured=dict(Counter(r['measured_level'] for r in entries)),
                requested_to_measured=dict(Counter(r['requested_profile']+'/'+r['measured_level'] for r in entries)),
                means={k:sum(r[k] for r in entries)/max(len(entries),1) for k in
                       ('position_mae_px','size_mae_px','position_mae_normalized','size_mae_normalized','changed_elements')})

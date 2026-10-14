"""Bounded pixel checks at document and nested-scroller positions."""
from pathlib import Path
from .explicit_html import SIZE_BOXES_JS,compare_renders


def capture(browser,out,reference=None,max_frames=128):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    page=browser.page
    dimensions=page.evaluate('''()=>({width:document.documentElement.scrollWidth,
      height:document.documentElement.scrollHeight,w:innerWidth,h:innerHeight,
      scrollers:[...document.querySelectorAll('[data-tuide-verify-id]')].filter(e=>{
        const s=getComputedStyle(e);return e!==document.body&&e!==document.documentElement&&
          ((/auto|scroll/.test(s.overflowY)&&e.scrollHeight>e.clientHeight)||
           (/auto|scroll/.test(s.overflowX)&&e.scrollWidth>e.clientWidth));
      }).map(e=>({id:e.dataset.tuideVerifyId,x:e.scrollLeft,y:e.scrollTop,
        width:e.scrollWidth,height:e.scrollHeight,w:e.clientWidth,h:e.clientHeight}))})''')
    def positions(length,view):
        end=max(0,length-view)
        yield from range(0,end,max(1,view))
        yield end
    probes=[]
    def add(probe):
        if len(probes)>=max_frames:raise ValueError(f'Scroll verification exceeds frame limit={max_frames}; not verified')
        probes.append(probe)
    if reference is None:
        for y in positions(dimensions['height'],dimensions['h']):
            for x in positions(dimensions['width'],dimensions['w']):add(dict(x=x,y=y,id=None))
        for s in dimensions['scrollers']:
            for y in positions(s['height'],s['h']):
                for x in positions(s['width'],s['w']):add(dict(x=x,y=y,id=s['id']))
    else:probes=reference['probes']
    frames=[]
    try:
        for index,p in enumerate(probes):
            page.evaluate('''({p,scrollers})=>{
              for(const s of scrollers){const e=document.querySelector(`[data-tuide-verify-id="${s.id}"]`);
                if(e)e.scrollTo({left:s.x,top:s.y,behavior:'instant'});}
              window.scrollTo({left:0,top:0,behavior:'instant'});
              if(p.id===null)window.scrollTo({left:p.x,top:p.y,behavior:'instant'});
              else {const e=document.querySelector(`[data-tuide-verify-id="${p.id}"]`);
                if(!e)throw Error('Missing scroller');e.scrollIntoView({block:'center',inline:'center',behavior:'instant'});
                e.scrollTo({left:p.x,top:p.y,behavior:'instant'});}
            }''',dict(p=p,scrollers=dimensions['scrollers']))
            # Scripts are disabled in the dataset browser; do not await rAF.
            page.wait_for_timeout(30)
            path=out/f'{index}.png';page.screenshot(path=str(path),animations='disabled')
            frames.append(dict(png=str(path),boxes=page.evaluate(SIZE_BOXES_JS)))
    finally:
        page.evaluate('''scrollers=>{for(const s of scrollers){const e=document.querySelector(`[data-tuide-verify-id="${s.id}"]`);
          if(e)e.scrollTo({left:s.x,top:s.y,behavior:'instant'});}window.scrollTo({left:0,top:0,behavior:'instant'});}''',dimensions['scrollers'])
    return dict(dimensions=dimensions,probes=probes,frames=frames)


def compare(before,after,ids,max_box_error,max_pixel_mae):
    failures=[]
    if before['dimensions']!=after['dimensions']:failures.append(dict(reason='scroll_extent_or_container_changed'))
    if len(before['frames'])!=len(after['frames']):failures.append(dict(reason='missing_scroll_frames'))
    for i,(a,b) in enumerate(zip(before['frames'],after['frames'])):
        result=compare_renders(a['boxes'],b['boxes'],a['png'],b['png'],max_box_error,max_pixel_mae,
                               geometry_target_ids=set(ids))
        if not result['accepted']:failures.append(dict(frame=i,probe=before['probes'][i],**result))
    return dict(accepted=not failures,frames=len(before['frames']),failures=failures,
                contract='box-scroll-pixels-v1',limitations='Finite viewport tiles; not all combinations of nested scroll offsets.')

"""Document-space preservation guard for content below the target viewport.

Checks geometry/visibility, not RGB similarity or unknown below-fold targets.
It intentionally permits ordinary document-flow translations.
"""

SNAPSHOT_JS = r'''()=>{
 const all=[...document.querySelectorAll('body,body *')];
 const index=new Map(all.map((e,i)=>[e,i]));
 const rows=[];
 for(let i=0;i<all.length;i++){
   const e=all[i],s=getComputedStyle(e),r=e.getBoundingClientRect();
   if(['STYLE','SCRIPT','LINK','META','NOSCRIPT'].includes(e.tagName))continue;
   const ownText=[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim());
   const painted=!['transparent','rgba(0, 0, 0, 0)'].includes(s.backgroundColor);
   if(!ownText&&!painted&&!e.matches('img,svg,canvas,video,input,button,select,textarea'))continue;
   let l=r.left,t=r.top,rr=r.right,bb=r.bottom,visible=true;
   for(let p=e;p;p=p.parentElement){
     const ps=getComputedStyle(p);
     if(ps.visibility!=='visible'||ps.display==='none'||parseFloat(ps.opacity)===0)visible=false;
     if(p===e)continue;
     // Root scrolling is the page, not clipping to the initial viewport.
     if(p===document.body||p===document.documentElement)continue;
     const b=p.getBoundingClientRect();
     if(['hidden','clip','scroll','auto'].includes(ps.overflowX)){
       l=Math.max(l,b.left+p.clientLeft);rr=Math.min(rr,b.left+p.clientLeft+p.clientWidth);}
     if(['hidden','clip','scroll','auto'].includes(ps.overflowY)){
       t=Math.max(t,b.top+p.clientTop);bb=Math.min(bb,b.top+p.clientTop+p.clientHeight);}
   }
   const area=Math.max(0,r.width)*Math.max(0,r.height);
   rows.push({id:String(i),tag:e.tagName,parent:index.get(e.parentElement),
     box:[r.left+scrollX,r.top+scrollY,r.width,r.height],
     visible_fraction:visible&&area?Math.max(0,rr-l)*Math.max(0,bb-t)/area:0});
 }
 return {viewport_height:innerHeight,viewport_width:innerWidth,
   document_height:document.documentElement.scrollHeight,elements:rows};
}'''


def snapshot(browser):
    return browser.page.evaluate(SNAPSHOT_JS)


def compare(before,after):
    actual={r['id']:r for r in after['elements']};failures=[];checked=0
    protected=[r for r in before['elements'] if r['box'][1]>=before['viewport_height']
               and r['visible_fraction']>.01 and min(r['box'][2:])>0]
    for r in protected:
        checked+=1;new=actual.get(r['id'])
        reason=None
        if new is None or new['tag']!=r['tag']:reason='missing_element'
        elif min(new['box'][2:])<=0 or new['visible_fraction']<=.01:reason='disappeared'
        elif new['visible_fraction']<r['visible_fraction']-.05:reason='increased_clipping'
        elif (r['box'][0]+r['box'][2]>0 and r['box'][0]<before['viewport_width']
              and (new['box'][0]+new['box'][2]<=0 or new['box'][0]>=after['viewport_width'])):
            # Detect a regression, not pre-existing offscreen carousel/hidden
            # content. An unchanged snapshot must always pass this check.
            reason='outside_horizontal_view'
        if reason:failures.append(dict(id=r['id'],reason=reason,before=r,after=new))
    # New intersections between previously disjoint siblings. Translation of a
    # whole section does not change this relation. Ancestor overlap is normal.
    def overlap(a,b):
        x,y,w,h=a;xx,yy,ww,hh=b
        return max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    groups={}
    for r in protected:groups.setdefault(r.get('parent'),[]).append(r)
    for siblings in groups.values():
        for i,a in enumerate(siblings):
            for b in siblings[i+1:]:
                aa=actual.get(a['id']);bb=actual.get(b['id'])
                if aa is None or bb is None:continue
                area=min(aa['box'][2]*aa['box'][3],bb['box'][2]*bb['box'][3])
                if overlap(a['box'],b['box'])<=1 and overlap(aa['box'],bb['box'])>max(4,.05*area):
                    failures.append(dict(id=a['id'],other_id=b['id'],reason='new_sibling_overlap'))
    return dict(accepted=not failures,checked_elements=checked,failures=failures,
                contract='below-fold-structural-v2',
                limitations='Not pixel/occlusion equivalence; nested scroller hidden contents are not exhaustively checked.')

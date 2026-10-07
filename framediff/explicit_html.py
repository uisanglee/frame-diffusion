"""Parent-preserving, single-viewport CSS normalization and visual feedback."""
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .html_bridge import HtmlBrowser
from .ir import read_jsonl, write_json, write_jsonl

def reject_flat_html(html):
    """Old flattened assets must be regenerated from their original HTML."""
    if 'data-tuide-flat' in html or 'tuide-hierarchy' in html:
        raise ValueError('Flattened HTML is no longer supported; use original parent-preserving HTML')


def explicit_size_limits(browser, ids):
    """Current physical border+padding floors; only normalized elements opt in."""
    return browser.page.evaluate('''ids=>Object.fromEntries(ids.map(id=>{
      const e=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.dataset.fdId===id);
      if(!e||!e.hasAttribute('data-tuide-explicit-id'))return [id,null];
      const s=getComputedStyle(e),n=k=>parseFloat(s[k])||0;
      return [id,{width:n('borderLeftWidth')+n('borderRightWidth')+n('paddingLeft')+n('paddingRight'),
                  height:n('borderTopWidth')+n('borderBottomWidth')+n('paddingTop')+n('paddingBottom')}];
    }))''',ids)


def check_explicit_edit(browser, ids, field, value):
    if field not in ('width','height'):return
    limits=explicit_size_limits(browser,ids)
    for bound in limits.values():
        if bound is None:continue
        # Normalized boxes have fixed px geometry. Removing the declaration or
        # replacing it with auto/% would reintroduce hidden layout constraints.
        from .tree_edits import explicit_value_valid
        if not explicit_value_valid(value,bound[field]):
            raise ValueError(f'Explicit {field} must be px and >= {bound[field]:g}; got {value!r}')


def owner_size_limits(row):
    """Map DOM-ID bounds to decoder owner indices, including shared rules."""
    limits=row.get('explicit_size_limits',{})
    if not limits:return {}
    owners=row.get('css_owners') or [{'matches':[n['id']]} for n in row['current']['nodes']]
    result={}
    for i,owner in enumerate(owners):
        bounds=[limits[id] for id in owner.get('matches',[]) if limits.get(id) is not None]
        if bounds:result[i]={field:max(b[field] for b in bounds) for field in ('width','height')}
    return result


def check_record_edit(row,edit):
    from .tree_edits import explicit_value_valid
    minimum=owner_size_limits(row).get(edit[0],{}).get(edit[1])
    if minimum is not None and not explicit_value_valid(edit[2],minimum):
        raise ValueError('Explicit size below border/padding minimum or non-px value')


def feedback_metadata(browser, tree):
    """Only current HTML constraints; never target sizes or target clipping."""
    ids=[n['id'] for n in tree['nodes'][1:]]
    limits=explicit_size_limits(browser,ids)
    return {'explicit_size_limits':limits,'current_clip_boxes':current_clip_boxes(browser,ids),
            'explicit_feedback_contract':'dom-preserved-boundary-v2'}


def current_clip_boxes(browser, ids=None):
    return browser.page.evaluate(r'''ids=>{
      const all=[...document.querySelectorAll('[data-tuide-explicit-id]')];
      const byId=new Map(all.map(e=>[e.dataset.tuideExplicitId,e]));
      const tagged=ids===null?byId:new Map([...document.querySelectorAll('[data-fd-id]')].map(e=>[e.dataset.fdId,e]));
      return Object.fromEntries((ids===null?[...byId.keys()]:ids).map(id=>{
        const e=tagged.get(id);if(!e)return [id,null];
        const r=e.getBoundingClientRect();let l=Math.max(0,r.left),t=Math.max(0,r.top),rr=Math.min(innerWidth,r.right),bb=Math.min(innerHeight,r.bottom);
        const seen=new Set();let p=e;
        while(p){
          p=p.parentElement;
          if(!p)break;if(seen.has(p))throw Error('Cyclic DOM hierarchy');seen.add(p);
          const s=getComputedStyle(p),b=p.getBoundingClientRect();
          if(['hidden','clip','scroll','auto'].includes(s.overflowX)){
            l=Math.max(l,b.left+p.clientLeft);rr=Math.min(rr,b.left+p.clientLeft+p.clientWidth);
          }
          if(['hidden','clip','scroll','auto'].includes(s.overflowY)){
            t=Math.max(t,b.top+p.clientTop);bb=Math.min(bb,b.top+p.clientTop+p.clientHeight);
          }
        }
        return [id,[l,t,Math.max(0,rr-l),Math.max(0,bb-t)]];
      }));
    }''',ids)


def feedback_screenshot(browser, **kwargs):
    """Real DOM rendering; no temporary reparenting or virtual clips."""
    return browser.page.screenshot(**kwargs)


def compare_renders(before, after, before_png, after_png, max_box_error=1., max_pixel_mae=.01):
    """Local ROIs prevent a vanished tiny element from hiding in global MAE."""
    actual={r['id']:r for r in after}
    if set(actual)!={r['id'] for r in before}:raise ValueError('Visual verification element IDs changed')
    with Image.open(io.BytesIO(before_png) if isinstance(before_png,bytes) else before_png) as a,Image.open(io.BytesIO(after_png) if isinstance(after_png,bytes) else after_png) as b:
        aa=np.asarray(a.convert('RGB'),dtype=float);bb=np.asarray(b.convert('RGB'),dtype=float)
    if aa.shape!=bb.shape:raise ValueError('Viewport changed during verification')
    delta=np.abs(aa-bb)/255;mae=float(delta.mean());box_error=0.;failures=[]
    for r in before:
        other=actual[r['id']]
        error=max(abs(x-y) for x,y in zip(r['box'],other['box']));box_error=max(box_error,error)
        x,y,w,h=r['box'];xx,yy,ww,hh=other['box']
        left=max(0,math.floor(min(x,xx)));top=max(0,math.floor(min(y,yy)))
        right=min(delta.shape[1],math.ceil(max(x+w,xx+ww)));bottom=min(delta.shape[0],math.ceil(max(y+h,yy+hh)))
        local=float(delta[top:bottom,left:right].mean()) if right>left and bottom>top else 0.
        clip_error=max((abs(a-b) for a,b in zip(r.get('visible_clip',[]),other.get('visible_clip',[]))),default=0.)
        if error>max_box_error or local>max_pixel_mae or clip_error>max_box_error:
            failures.append(dict(id=r['id'],box_error=error,local_pixel_mae=local,clip_error=clip_error))
    return dict(accepted=not failures and mae<=max_pixel_mae,pixel_mae=mae,max_box_error_px=box_error,
                element_failures=failures,max_pixel_mae=max_pixel_mae,max_box_error=max_box_error)


FREEZE_JS = r'''(mode) => {
 const textAnchors=[];
 if(mode==='boxes') {
   // Direct text has no independently positionable box. Wrap only text whose
   // parent loses padding; all:unset avoids inheriting its border/background.
   for(const parent of [document.body,...document.body.querySelectorAll('*')]) {
     if(parent.namespaceURI!=='http://www.w3.org/1999/xhtml')continue;
     const s=getComputedStyle(parent);
     if(!['paddingTop','paddingRight','paddingBottom','paddingLeft'].some(k=>parseFloat(s[k])>0))continue;
     if(['INPUT','TEXTAREA','SELECT'].includes(parent.tagName))
       throw Error('Native control padding cannot be removed without materializing internal content');
     for(const node of [...parent.childNodes]) {
       if(node.nodeType!==Node.TEXT_NODE || !node.textContent.trim())continue;
       const range=document.createRange();range.selectNodeContents(node);
       if(range.getClientRects().length>1)throw Error('Multiline padded text requires line fragmentation');
       const r=range.getBoundingClientRect();
       if(!r.width||!r.height)continue;
       const wrapper=document.createElement('tuide-text');
       wrapper.setAttribute('data-tuide-padding-text','');
       wrapper.style.setProperty('all','unset','important');
       wrapper.style.setProperty('display','inline','important');
       node.before(wrapper);wrapper.appendChild(node);
       textAnchors.push({wrapper,node,x:r.x,y:r.y});
     }
   }
 }
 const all=[document.documentElement,...document.querySelectorAll('body, body *')]
   .filter(e=>!['STYLE','SCRIPT','LINK','META','NOSCRIPT'].includes(e.tagName));
 const snapshots=all.map((e,i)=>{
   const s=getComputedStyle(e),r=e.getBoundingClientRect();
   const svg=e.namespaceURI==='http://www.w3.org/2000/svg' && e.tagName.toLowerCase()!=='svg';
   const visible=r.width>0 && r.height>0 && s.display!=='none';
   if(e.shadowRoot || ['IFRAME','CANVAS'].includes(e.tagName))throw Error('Unsupported embedded/shadow content');
   for(const pseudo of ['::before','::after']) {
     const p=getComputedStyle(e,pseudo);
     if(p.content!=='none' && p.content!=='normal')throw Error('Generated pseudo-element requires materialization: '+pseudo);
   }
   if(visible && !svg && (s.transform!=='none' || s.rotate!=='none' || s.scale!=='none' || s.translate!=='none'))
     throw Error('Transformed element requires transform baking');
   if(visible && !svg && e.getClientRects().length>1)throw Error('Fragmented inline element requires text fragmentation');
   const style={};
   for(const p of s) {
     // Do not retain logical aliases that compete with editable physical values.
     if(/(^|-)inline(-|$)|(^|-)block(-|$)/.test(p))continue;
     style[p]=s.getPropertyValue(p);
   }
   const parent=e.parentElement,pr=parent?.getBoundingClientRect();
   const ps=parent?getComputedStyle(parent):null;
   e.setAttribute('data-tuide-explicit-id',String(i));
   return {e,style,svg,visible,box:[r.x,r.y,r.width,r.height],
     left:pr?r.x-pr.x-(parseFloat(ps.borderLeftWidth)||0)+parent.scrollLeft:r.x,
     top:pr?r.y-pr.y-(parseFloat(ps.borderTopWidth)||0)+parent.scrollTop:r.y};
 });
 // Snapshot ALL styles before touching any declaration or stylesheet.
 document.querySelectorAll('style,link[rel~="stylesheet"],script').forEach(e=>e.remove());
 for(const a of snapshots) {
   const e=a.e;e.removeAttribute('style');
   for(const [p,v] of Object.entries(a.style))e.style.setProperty(p,v);
   const set=(p,v)=>e.style.setProperty(p,String(v));
   set('animation','none');set('transition','none');
   if(a.svg || !a.visible)continue;
   if(e===document.documentElement) {
     set('position','relative');set('margin','0');set('padding','0');
     continue;
   }
   set('box-sizing','border-box');set('width',a.box[2]+'px');set('height',a.box[3]+'px');
   for(const p of ['min-width','max-width','min-height','max-height'])e.style.removeProperty(p);
   set('flex','0 0 auto');set('aspect-ratio','auto');
   if(mode==='boxes') {
     set('padding','0');
     set('display','block');set('position','absolute');set('float','none');
     set('margin','0');set('left',a.left+'px');set('top',a.top+'px');
     set('right','auto');set('bottom','auto');
   }
 }
 // Inline glyph boxes and block line boxes have different baseline offsets.
 // Align the actual text range, not just the new wrapper's outer rectangle.
 for(const a of textAnchors) {
   const range=document.createRange();range.selectNodeContents(a.node);
   const r=range.getBoundingClientRect(),e=a.wrapper;
   e.style.left=(parseFloat(e.style.left)+a.x-r.x)+'px';
   e.style.top=(parseFloat(e.style.top)+a.y-r.y)+'px';
 }
 return snapshots.filter(a=>a.visible&&!a.svg).map(a=>({id:a.e.getAttribute('data-tuide-explicit-id'),box:a.box}));
}'''


def convert(browser, html, viewport, out, mode='boxes', max_pixel_mae=.01, max_box_error=1.):
    """Emit normalized.html only when a fresh reload passes geometry/pixel QA."""
    if mode not in ('boxes','sizes'):raise ValueError('Parent separation was removed; use boxes or sizes')
    reject_flat_html(html)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    browser.reset_context();browser.load(html,viewport)
    browser.page.screenshot(path=str(out/'before.png'),animations='disabled')
    before=browser.page.evaluate(FREEZE_JS,mode)
    candidate=browser.page.content()
    browser.reset_context();browser.load(candidate,viewport)
    browser.page.screenshot(path=str(out/'after.png'),animations='disabled')
    actual=browser.page.evaluate('''()=>Object.fromEntries([...document.querySelectorAll('[data-tuide-explicit-id]')].map(e=>{
      let r=e.getBoundingClientRect();
      if(e.hasAttribute('data-tuide-padding-text')){const range=document.createRange();range.selectNodeContents(e);r=range.getBoundingClientRect();}
      return [e.getAttribute('data-tuide-explicit-id'),[r.x,r.y,r.width,r.height]];
    }))''')
    error=max((max(abs(a-b) for a,b in zip(r['box'],actual[r['id']])) for r in before),default=0.)
    with Image.open(out/'before.png') as a,Image.open(out/'after.png') as b:
        mae=float(np.abs(np.asarray(a.convert('RGB'),dtype=float)-np.asarray(b.convert('RGB'),dtype=float)).mean()/255)
    verification=compare_renders(before,[{'id':r['id'],'box':actual[r['id']]} for r in before],
                                 out/'before.png',out/'after.png',max_box_error,max_pixel_mae)
    accepted=verification['accepted']
    report=dict(accepted=accepted,mode=mode,viewport=viewport,pixel_mae=mae,max_box_error_px=error,
                max_pixel_mae=max_pixel_mae,max_box_error=max_box_error,network_blocked=True,
                warning='Single viewport only; responsive behavior and CSS interactions are not preserved.')
    report.update(verification)
    write_json(out/'report.json',report)
    (out/('normalized.html' if accepted else 'rejected.html')).write_text(candidate)
    return report


def add_parser(sub):
    p=sub.add_parser('visual-explicit-html',help='Bake computed styles into inline CSS and verify a fresh render')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--html');source.add_argument('--manifest',help='JSONL containing html, viewport, id, group, split')
    p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['sizes','boxes'],default='boxes')
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    p.add_argument('--max-pixel-mae',type=float,default=.01)
    p.add_argument('--max-box-error',type=float,default=1.)

def run(args):
    if min(args.width,args.height)<=0 or not 0<=args.max_pixel_mae<=1 or args.max_box_error<0:
        raise ValueError('Invalid viewport or verification thresholds')
    out=Path(args.out).resolve()
    if out.exists():raise ValueError('Use a new output directory to preserve existing artifacts')
    sources=list(read_jsonl(args.manifest)) if args.manifest else [dict(id='page',html=args.html,viewport=[args.width,args.height])]
    if not sources:raise ValueError('Empty manifest')
    out.mkdir(parents=True);accepted=[];reports=[]
    with HtmlBrowser() as browser:
        for i,row in enumerate(sources):
            folder=out/str(i) if args.manifest else out
            try:
                source=Path(row['html']).resolve();html=source.read_text()
                report=convert(browser,html,row.get('viewport',[args.width,args.height]),folder,args.mode,
                               args.max_pixel_mae,args.max_box_error)
                if report['accepted']:
                    # Do not inherit cached boxes/labels/detector targets of the old HTML.
                    clean={k:row[k] for k in ('id','group','split') if k in row}
                    accepted.append({**clean,'html':str(folder/'normalized.html'),
                        'viewport':report['viewport'],'original_html':str(source),
                        'original_sha256':hashlib.sha256(html.encode()).hexdigest()})
            except Exception as exc:
                report={'accepted':False,'error':str(exc)}
                folder.mkdir(parents=True,exist_ok=True);write_json(folder/'report.json',report)
            reports.append({'id':row.get('id',str(i)),**report})
            print(f'[{i+1}/{len(sources)}] accepted={report["accepted"]}',flush=True)
    write_jsonl(out/'manifest.jsonl',accepted)
    write_json(out/'summary.json',{'accepted':len(accepted),'total':len(sources),'records':reports})
    if not accepted:raise ValueError(f'No page passed normalization; inspect {out}/summary.json')

"""Single-viewport HTML bridge. Keeps real DOM/content; does not reconstruct CSS."""
import base64
import math
import re
from pathlib import Path

from .browser import Browser


def html_answer(answer):
    answer = re.sub(r'^\s*```(?:html)?\s*', '', answer, flags=re.I)
    answer = re.sub(r'\s*```\s*$', '', answer).strip()
    if not re.search(r'<html\b', answer, re.I) or not re.search(r'</html\s*>', answer, re.I):
        raise ValueError('Expected a complete HTML document; possible truncated generation')
    return answer


def embed_placeholder(html, path=None):
    # Only this explicitly provided benchmark asset can be embedded; no fetching URLs.
    if path:
        data = base64.b64encode(Path(path).read_bytes()).decode()
        html = html.replace('rick.jpg', 'data:image/jpeg;base64,' + data)
    return html


class HtmlBrowser(Browser):
    def load(self, html, viewport):
        self.page.set_viewport_size({'width': int(viewport[0]), 'height': int(viewport[1])})
        self.page.set_content(html, wait_until='load', timeout=30000)
        self.executions += 1
        self.page.evaluate("""()=>{const s=document.querySelector('style[data-framediff-static]')||document.createElement('style');
          s.setAttribute('data-framediff-static','');
          s.textContent='*,*::before,*::after{animation:none!important;transition:none!important}';
          document.head.appendChild(s);} """)

    def snapshot(self, html, viewport, screenshot=None, max_nodes=127):
        self.load(html, viewport)
        # Stable IDs are assigned on the initial DOM and persisted in serialized HTML.
        # Existing data-frame-id values are intentionally ignored.
        result = self.page.evaluate('''(limit) => {
          document.querySelectorAll('[data-fd-id]').forEach(e=>e.removeAttribute('data-fd-id'));
          const all=[document.body,...document.body.querySelectorAll('*')].filter(e=>{
            const r=e.getBoundingClientRect(),s=getComputedStyle(e);
            return !['SCRIPT','STYLE','META','LINK','NOSCRIPT'].includes(e.tagName) &&
              r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none';
          });
          const ranked=all.filter(e=>e!==document.body).map((e,i)=>({e,i,score:
            (['IMG','BUTTON','INPUT','SVG'].includes(e.tagName)?1e12:0)+
            (e.children.length===0 && e.textContent.trim()?1e11:0)+
            Math.min(e.getBoundingClientRect().width*e.getBoundingClientRect().height,1e10)
          })).sort((a,b)=>b.score-a.score||a.i-b.i).slice(0,limit-1);
          const keep=new Set([document.body,...ranked.map(x=>x.e)]);
          const selected=all.filter(e=>keep.has(e));
          selected.forEach((e,i)=>e.setAttribute('data-fd-id',`fd-${i}`));
          const nodes=selected.map(e=>{
            let p=e.parentElement;while(p&&!p.hasAttribute('data-fd-id'))p=p.parentElement;
            const r=e.getBoundingClientRect();
            return {id:e.getAttribute('data-fd-id'),parent:p?p.getAttribute('data-fd-id'):null,
              role:e.tagName.toLowerCase(),name:(e.getAttribute('aria-label')||e.textContent||e.tagName).trim().slice(0,120),
              box:[r.x,r.y,r.width,r.height]};
          });
          return {nodes,total_visible_nodes:all.length};
        }''', max_nodes)
        if not result['nodes']:
            raise ValueError('No visible DOM elements')
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(screenshot), animations='disabled')
            self.screenshots += 1
        result.update(viewport=list(viewport), html=self.page.content())
        return result

    def tagged_boxes(self, ids):
        """Read stable identities without resampling/reassigning IDs after a reflow."""
        boxes = self.page.evaluate('''()=>Object.fromEntries(
          [...document.querySelectorAll('[data-fd-id]')].map(e=>{
            const r=e.getBoundingClientRect();return [e.getAttribute('data-fd-id'),[r.x,r.y,r.width,r.height]];
          }))''')
        if set(ids) != set(boxes):
            raise ValueError('DOM identities changed during feedback; refusing to rematch silently')
        return boxes

    def edit_property(self, html, viewport, node_id, field, delta):
        """One CSS property edit, not a simultaneous box projection.

        Width/height change the used border-box size via the existing box-sizing.
        dx/dy change margins, allowing normal-flow reflow. Other elements and auto
        heights remain untouched. CSS constraints/transforms may prevent exact motion.
        """
        if field not in ('width','height','dx','dy') or not math.isfinite(delta):
            raise ValueError('Unsupported HTML feedback action')
        self.load(html, viewport)
        self.page.evaluate('''({id,field,delta})=>{
          const e=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.getAttribute('data-fd-id')===id);
          if(!e)throw new Error('Missing edit target');
          const s=getComputedStyle(e),r=e.getBoundingClientRect();
          const num=k=>parseFloat(s.getPropertyValue(k))||0;
          let property,value;
          if(field==='width'||field==='height'){
            const horizontal=field==='width';
            const edges=horizontal?['padding-left','padding-right','border-left-width','border-right-width']:
              ['padding-top','padding-bottom','border-top-width','border-bottom-width'];
            const inset=s.boxSizing==='border-box'?0:edges.reduce((a,k)=>a+num(k),0);
            property=field;value=Math.max(0,(horizontal?r.width:r.height)+delta-inset);
          }else{
            property=field==='dx'?'margin-left':'margin-top';value=num(property)+delta;
          }
          e.style.setProperty(property,`${value}px`,'important');
        }''', {'id':node_id,'field':field,'delta':delta})

    def edit_visual_action(self, html, viewport, node_id, field, value, return_inverse=False):
        """Apply one v3 structured action under the normalized size contract.

        Numeric values are viewport-axis fractions, never raw pixels. Categorical
        values replace one grammar-constrained flex declaration. The returned
        inverse is used only while constructing verified corruption trajectories.
        """
        from .visual import (NUMERIC_FIELDS,ACTION_TO_INDEX,
                             action_delta_px,validate_action)
        validate_action((0,field,value),viewport)
        numeric=field in NUMERIC_FIELDS
        delta=action_delta_px(field,value,viewport) if numeric else None
        self.load(html,viewport)
        old=self.page.evaluate('''({id,field,value,delta,numeric})=>{
          const e=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.getAttribute('data-fd-id')===id);
          if(!e)throw new Error('Missing edit target');
          const s=getComputedStyle(e),r=e.getBoundingClientRect();
          const num=k=>{const v=parseFloat(s.getPropertyValue(k));return Number.isFinite(v)?v:0};
          if(!numeric){
            const old=s.getPropertyValue(field).trim();
            e.style.setProperty(field,value,'important');return old;
          }
          let current,minimum=-Infinity;
          if(field==='width'||field==='height'){
            const horizontal=field==='width';
            const edges=horizontal?['padding-left','padding-right','border-left-width','border-right-width']:
              ['padding-top','padding-bottom','border-top-width','border-bottom-width'];
            const inset=s.boxSizing==='border-box'?0:edges.reduce((a,k)=>a+num(k),0);
            current=(horizontal?r.width:r.height)-inset;minimum=0;
          }else{current=num(field)}
          e.style.setProperty(field,`${Math.max(minimum,current+delta)}px`,'important');return null;
        }''', {'id':node_id,'field':field,'value':value,'delta':delta,'numeric':numeric})
        if not return_inverse:return None
        inverse=(field,-value) if numeric else (field,old)
        if inverse not in ACTION_TO_INDEX:
            raise ValueError(f'CSS value has no reversible action: {field}={old!r}')
        return inverse


    def patch(self, html, viewport, desired):
        """Calibrate geometry in DOM order, then serialize inline CSS (no JS in output).

        Parent changes may alter children. Each element is measured after ancestors.
        Complex transforms/table constraints can prevent exact transfer: caller measures it.
        """
        for box in desired.values():
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or min(box[2:]) <= 0:
                raise ValueError('Invalid transfer box')
        self.load(html, viewport)
        actual = self.page.evaluate('''(boxes)=>{
          const els=[...document.querySelectorAll('[data-fd-id]')];
          for(const e of els){
            const b=boxes[e.getAttribute('data-fd-id')];if(!b)continue;
            let r=e.getBoundingClientRect();
            if(Math.max(...b.map((v,i)=>Math.abs(v-[r.x,r.y,r.width,r.height][i])))<0.01)continue;
            const set=(k,v)=>e.style.setProperty(k,v,'important');
            set('box-sizing','border-box');set('width',b[2]+'px');set('height',b[3]+'px');
            set('min-width','0');set('min-height','0');set('max-width','none');set('max-height','none');
            set('flex-shrink','0');
            const s=getComputedStyle(e);
            if(s.display==='inline')set('display','inline-block');
            // Individual translate composes with existing transform; remove its old value first.
            set('translate','none');r=e.getBoundingClientRect();
            set('translate',`${b[0]-r.x}px ${b[1]-r.y}px`);
          }
          return Object.fromEntries(els.filter(e=>boxes[e.getAttribute('data-fd-id')]).map(e=>{
            const r=e.getBoundingClientRect();return [e.getAttribute('data-fd-id'),[r.x,r.y,r.width,r.height]];
          }));
        }''', desired)
        if set(actual) != set(desired):
            raise ValueError('DOM identity lost during HTML transfer')
        error = max((abs(actual[k][i]-b[i]) for k,b in desired.items() for i in range(4)), default=0)
        return self.page.content(), {'transfer_max_error_px': error, 'actual': actual}

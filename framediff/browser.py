from pathlib import Path
from .ir import compile_html

class Browser:
    """One persistent browser; count geometry executions separately from screenshots."""
    def __init__(self):
        self.executions=0;self.screenshots=0
    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self.pw=sync_playwright().start()
        try:self.browser=self.pw.chromium.launch(headless=True)
        except Exception:
            self.pw.stop();raise
        self.context=self.browser.new_context(java_script_enabled=False)
        self.context.route('**/*',lambda route:route.abort())
        self.page=self.context.new_page()
        return self
    def __exit__(self,*args):
        self.browser.close();self.pw.stop()
    def render(self,tree,viewport,screenshot=None):
        self.page.set_viewport_size({'width':viewport[0],'height':viewport[1]})
        self.page.set_content(compile_html(tree,viewport),wait_until='load')
        self.executions+=1
        boxes=self.page.locator('[data-frame-id]').evaluate_all('els => Object.fromEntries(els.map(e=>{const r=e.getBoundingClientRect();return [e.dataset.frameId,[r.x,r.y,r.width,r.height]]}))')
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            self.page.screenshot(path=str(screenshot));self.screenshots+=1
        return boxes
    def extract_html(self,html,viewport=(1024,768),screenshot=None,max_nodes=128,overflow='error'):
        self.page.set_viewport_size({'width':viewport[0],'height':viewport[1]})
        self.page.set_content(html,wait_until='load');self.executions+=1
        data=self.page.locator('body').evaluate('''(body,options)=>{
          const all=[body,...body.querySelectorAll('*')].filter(e=>{
            const r=e.getBoundingClientRect(),s=getComputedStyle(e);
            return !['SCRIPT','STYLE','META','LINK','NOSCRIPT'].includes(e.tagName)&&r.width>0&&r.height>0&&s.display!=='none'&&s.visibility!=='hidden';
          });
          if(all.length>options.maxNodes&&options.overflow==='error') throw new Error(`Visible DOM has ${all.length} nodes; max ${options.maxNodes}`);
          let selected=all;
          if(all.length>options.maxNodes){
            const ranked=all.slice(1).map((e,index)=>{const r=e.getBoundingClientRect(),tag=e.tagName;
              const semantic=['IMG','SVG','BUTTON','INPUT','TEXTAREA','SELECT','CANVAS','VIDEO'].includes(tag);
              const leaf=e.children.length===0, text=(e.innerText||'').trim().length>0;
              return {e,index,score:(semantic?1e12:0)+(leaf&&text?1e11:0)+Math.min(r.width*r.height,1e10)};
            }).sort((a,b)=>b.score-a.score||a.index-b.index).slice(0,options.maxNodes-1);
            const keep=new Set([body,...ranked.map(x=>x.e)]);selected=all.filter(e=>keep.has(e));
          }
          const ids=new Map(selected.map((e,i)=>[e,e.getAttribute('data-frame-id')||`dom-${i}`]));
          const nodes=selected.map(e=>{let p=e.parentElement;while(p&&!ids.has(p))p=p.parentElement;
            const r=e.getBoundingClientRect(),s=getComputedStyle(e);
            return {id:ids.get(e),parent:p?ids.get(p):null,role:e.getAttribute('role')||e.tagName.toLowerCase(),
              name:e.getAttribute('aria-label')||e.textContent.trim().slice(0,80)||e.tagName,
              box:[r.x,r.y,r.width,r.height],style:{display:s.display,flexDirection:s.flexDirection,padding:s.padding,gap:s.gap,width:s.width,height:s.height}};
          });
          return {nodes,totalVisible:all.length};
        }''',{'maxNodes':max_nodes,'overflow':overflow})
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            self.page.screenshot(path=str(screenshot));self.screenshots+=1
        return {'viewport':list(viewport),'nodes':data['nodes'],'total_visible_nodes':data['totalVisible'],
                'selection':overflow,'network_blocked':True,'javascript_enabled':False}

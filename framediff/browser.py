from pathlib import Path
from .ir import compile_html

class _Runtime:
    def __init__(self):
        self.pw = self.browser = None
        self.active = False

    def start(self):
        if self.pw is not None:
            return
        from playwright.sync_api import sync_playwright
        self.pw = sync_playwright().start()
        try:
            self.browser = self.pw.chromium.launch(headless=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        browser, pw = self.browser, self.pw
        self.browser = self.pw = None
        try:
            if browser is not None:
                browser.close()
        finally:
            if pw is not None:
                pw.stop()


_worker_runtime = None


def enable_worker_browser():
    """Only preparation workers opt into process-local browser reuse."""
    global _worker_runtime
    if _worker_runtime is not None:
        raise RuntimeError('Worker browser session already enabled')
    _worker_runtime = _Runtime()


def close_worker_browser():
    global _worker_runtime
    runtime, _worker_runtime = _worker_runtime, None
    if runtime is not None:
        runtime.close()


class Browser:
    """One persistent browser; count geometry executions separately from screenshots."""
    def __init__(self):
        self.executions=0;self.screenshots=0
    def __enter__(self):
        self._runtime = _worker_runtime if _worker_runtime is not None else _Runtime()
        self._shared = _worker_runtime is not None
        if self._runtime.active:
            raise RuntimeError('Overlapping browser leases in one worker are not supported')
        self._runtime.active = True
        self.context = self.page = None
        try:
            self._start_runtime()
            self._new_context()
        except BaseException:
            try:self._runtime.close()
            finally:self._runtime.active = False
            raise
        return self
    def _start_runtime(self):
        self._runtime.start()
        self.pw, self.browser = self._runtime.pw, self._runtime.browser
    def _new_context(self):
        self.context=self.browser.new_context(java_script_enabled=False)
        self.context.route('**/*',lambda route:route.abort())
        self.page=self.context.new_page()
    def reset_context(self):
        """Isolate untrusted pages and recover after a renderer/page crash."""
        restart = not self.browser.is_connected()
        try:
            if self.context is not None:self.context.close()
        except Exception:
            restart = True
        self.context = self.page = None
        try:
            if restart:
                self._runtime.close()
                self._start_runtime()
            self._new_context()
        except BaseException:
            self._runtime.close()
            raise
    def __exit__(self,*args):
        try:
            if self._shared:
                try:
                    if self.context is not None:self.context.close()
                except BaseException:
                    self._runtime.close()
                    raise
            else:
                self._runtime.close()
        finally:
            self.context = self.page = None
            self._runtime.active = False
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

"""Execute generated apps and verify an editable static view of their current state.

This renderer is deliberately separate from the network/JS-disabled training
renderer. It supports a bounded set of browser libraries, not arbitrary apps.
Original application source remains the deployable artifact; snapshots are an
intermediate representation for CSS repair, never a replacement for that source.
"""
import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .html_bridge import HtmlBrowser

CONTRACT = 'generated-app-static-bridge-v1'


def embed_application_assets(source):
    """Resolve reserved WebUI image files in HTML/CSS and browser app source.

    JSX and JavaScript strings are opaque to the static HTML asset resolver.
    Only exact quoted reserved filenames are replaced here; this is not a
    general-purpose JavaScript rewrite or an external asset downloader.
    """
    from .webui_placeholders import embed_placeholders, image_uri
    source=embed_placeholders(source)
    def script(match):
        body=re.sub(r'''(["'`])rick([1-5])\.jpg\1''',
                    lambda asset:asset[1]+image_uri(int(asset[2])-1)+asset[1],match[2])
        return match[1]+body+match[3]
    return re.sub(r'(<script\b[^>]*>)(.*?)(</script\s*>)',script,source,flags=re.I|re.S)


def allowed_dependency(url):
    parts = urlsplit(url)
    if parts.scheme != 'https' or parts.username or parts.password or parts.port not in (None, 443):
        return False
    host, path = parts.hostname, parts.path
    if host in ('cdn.jsdelivr.net', 'unpkg.com'):
        prefix = '/npm/' if host == 'cdn.jsdelivr.net' else '/'
        return bool(re.fullmatch(re.escape(prefix) +
            r'(?:(?:react|react-dom|@babel/standalone)@[0-9][^/]*/[^?]+|@babel/standalone/babel(?:\.min)?\.js)', path))
    if host == 'cdn.tailwindcss.com':
        return path in ('', '/') or bool(re.fullmatch(r'/[0-9]+(?:\.[0-9]+){1,2}/?', path))
    if host == 'cdnjs.cloudflare.com':
        return bool(re.fullmatch(r'/ajax/libs/font-awesome/[0-9][^/]*/(?:css|webfonts|fonts)/.+', path))
    if host == 'fonts.googleapis.com':
        return path in ('/css', '/css2')
    if host == 'fonts.gstatic.com':
        return path.startswith('/s/')
    return False


class AppBrowser(HtmlBrowser):
    def __init__(self, cache_dir=None, timeout_ms=30000, browser_host=None):
        super().__init__()
        self.browser_host = browser_host
        self.cache_dir = Path(cache_dir or os.environ.get('TUIDE_APP_ASSET_CACHE', '.cache/app-dependencies')).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_ms = int(timeout_ms)
        if not 0 < self.timeout_ms <= 30000:
            raise ValueError('Application readiness timeout must be in (0, 30000] ms')
        self.resources = {}; self.resource_errors = []; self.script_errors = []
        self.deadline = None

    def __enter__(self):
        if self.browser_host is None:
            return super().__enter__()
        host=self.browser_host
        if host.page is None or not host.browser.is_connected():
            raise ValueError('Application browser_host must be an active browser')
        # Playwright cannot nest two sync runtimes on one thread. Borrow only
        # the browser process; never share the host's static browser context.
        self._runtime=host._runtime
        self.pw,self.browser=host.pw,host.browser
        self.context=self.page=None
        self._new_context()
        return self

    def __exit__(self,*args):
        if self.browser_host is None:
            return super().__exit__(*args)
        try:
            if self.context is not None:self.context.close()
        finally:
            self.context=self.page=None

    def reset_context(self):
        if self.browser_host is None:
            return super().reset_context()
        if not self.browser.is_connected():
            raise ValueError('Application host browser disconnected')
        if self.context is not None:self.context.close()
        self.context=self.page=None
        self._new_context()

    def _remaining(self):
        return max(1, int((self.deadline-time.monotonic())*1000)) if self.deadline else self.timeout_ms

    def _new_context(self):
        self.context = self.browser.new_context(java_script_enabled=True, service_workers='block',
                                               accept_downloads=False)
        self._runtime.contexts += 1
        self.context.route('**/*', self._route)
        self.page = self.context.new_page()
        self.page.on('pageerror', lambda error: self.script_errors.append(str(error)))
        self.page.on('dialog', lambda dialog: dialog.dismiss())
        self.page.on('popup', lambda page: page.close())

    def _resource(self, url, headers=None):
        if not allowed_dependency(url):
            raise ValueError('Resource not allowlisted: '+url)
        if url in self.resources:return self.resources[url]
        key=hashlib.sha256(url.encode()).hexdigest()
        metadata_path=self.cache_dir/(key+'.json')
        if metadata_path.exists():
            meta=json.loads(metadata_path.read_text())
            body=(self.cache_dir/meta['sha256']).read_bytes()
            if hashlib.sha256(body).hexdigest()!=meta['sha256']:
                raise ValueError('Cached dependency digest mismatch')
        else:
            current=url
            for _ in range(6):
                if self._remaining()<=1:raise ValueError('Application dependency deadline exceeded')
                response=self.context.request.get(current,headers=headers,
                    timeout=min(10000,self._remaining()),max_redirects=0)
                if 300<=response.status<400:
                    target=urljoin(current,response.headers.get('location',''))
                    response.dispose()
                    if target==current or not allowed_dependency(target):
                        raise ValueError('Unsafe dependency redirect: '+target)
                    current=target;continue
                try:
                    if not response.ok:raise ValueError(f'Dependency HTTP {response.status}: {current}')
                    body=response.body()
                    if len(body)>15*1024*1024:raise ValueError('Dependency exceeds 15 MiB')
                    meta=dict(url=url,resolved_url=current,sha256=hashlib.sha256(body).hexdigest(),
                        content_type=response.headers.get('content-type','application/octet-stream'))
                finally:response.dispose()
                break
            else:raise ValueError('Too many dependency redirects')
            destination=self.cache_dir/meta['sha256']
            temporary=destination.with_name(destination.name+f'.{os.getpid()}.tmp')
            temporary.write_bytes(body);os.replace(temporary,destination)
            temporary=metadata_path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(meta));os.replace(temporary,metadata_path)
        self.resources[url]=dict(meta,body=body)
        return self.resources[url]

    def _route(self, route):
        request=route.request
        url=request.url
        if request.method!='GET' or request.is_navigation_request() or not allowed_dependency(url):
            self.resource_errors.append({'url':url,'error':'resource_not_allowlisted'})
            route.abort();return
        try:
            resource=self._resource(url,headers=request.headers)
            route.fulfill(body=resource['body'],content_type=resource['content_type'],
                          headers={'access-control-allow-origin':'*'})
        except Exception as exc:
            self.resource_errors.append({'url':url,'error':str(exc)})
            route.abort()

    def load(self, html, viewport):
        from .explicit_html import reject_flat_html
        html=embed_application_assets(html)
        reject_flat_html(html)
        # Each source runs in a new document/context: timers/listeners from the
        # preceding app cannot leak into another page's generated result.
        self.reset_context()
        self.resources={};self.resource_errors=[];self.script_errors=[]
        self.source_html=html
        self.deadline=time.monotonic()+self.timeout_ms/1000
        self.page.set_viewport_size({'width':int(viewport[0]),'height':int(viewport[1])})
        self.executions += 1
        self.page.set_content(html, wait_until='load', timeout=self._remaining())
        # Inline scripts and JSX compilation may mount after the load event.
        # Wait for actual content, never treat a full-size empty root as ready.
        while True:
            if self.resource_errors:
                raise ValueError('Application resource loading failed: '+json.dumps(self.resource_errors[:3]))
            if self.script_errors:
                raise ValueError('Application JavaScript failed: '+self.script_errors[0])
            try:
                self.require_visible_content()
                break
            except ValueError:
                if self._remaining() <= 1:
                    raise ValueError('Application did not produce visible content before readiness timeout')
                self.page.wait_for_timeout(min(100,self._remaining()))
        self.page.evaluate('''()=>{const s=document.createElement('style');
          s.setAttribute('data-framediff-static','');
          s.textContent='*,*::before,*::after{animation:none!important;transition:none!important}';
          document.head.appendChild(s);}''')
        # Bounded frame stability is an operational readiness check, not a
        # guarantee against future timers. The subsequent freeze comparison
        # rejects a changed state instead of silently accepting a mismatch.
        # HtmlBrowser's diagnostic helper reports its sample count only on
        # success. Count completed captures as they happen so unstable apps
        # still report the work spent before their readiness failure.
        capture=self.page.screenshot
        def counted_capture(*args,**kwargs):
            result=capture(*args,**kwargs)
            self.screenshots+=1
            return result
        self.page.screenshot=counted_capture
        try:
            self.stable_capture(timeout_ms=self._remaining(), interval_ms=100)
        finally:
            self.page.screenshot=capture
        if self.resource_errors or self.script_errors:
            raise ValueError('Application failed while stabilizing: '+str(self.resource_errors or self.script_errors))
        overlay_errors=self.page.evaluate('()=>window.__tuideApplicationEdits?.errors || []')
        if overlay_errors:
            raise ValueError('Application CSS overlay mapping failed: '+json.dumps(overlay_errors))
        self._bind_nodes()

    def _bind_nodes(self):
        self.page.evaluate('''()=>{
          const ignore=new Set(['SCRIPT','STYLE','META','LINK','NOSCRIPT','TEMPLATE','SOURCE']);
          const selector=e=>{if(e===document.body)return 'body';
            const tag=e.localName;
            const same=[...e.parentElement.children].filter(x=>x.localName===tag);
            return selector(e.parentElement)+' > '+tag+':nth-of-type('+(same.indexOf(e)+1)+')';};
          for(const e of [document.body,...document.body.querySelectorAll('*')])
            if(!ignore.has(e.tagName))e.setAttribute('data-tuide-app-node',selector(e));
        }''')

    def _static_html(self):
        # Export the CSSOM, not only style.textContent: insertRule() and
        # Tailwind's runtime can create rules that page.content() omits.
        payload=self.page.evaluate('''()=>{
          const all=[...document.querySelectorAll('*')];
          if(all.some(e=>e.shadowRoot)||document.querySelector('canvas,iframe,object,embed'))
            throw new Error('Static bridge does not support shadow DOM, canvas or embedded documents');
          if(document.adoptedStyleSheets.length)throw new Error('Adopted stylesheets are unsupported');
          const copy=document.documentElement.cloneNode(true);
          const original=[document.documentElement,...all.filter(e=>e!==document.documentElement)];
          const clones=[copy,...copy.querySelectorAll('*')];
          for(let i=0;i<original.length;i++){
            const e=original[i],c=clones[i];
            // A later CSSOM edit serializes the entire style attribute.
            // Canonicalize now so unchanged declarations retain identical
            // values when validating which CSS properties were repaired.
            if(e.hasAttribute('style'))c.setAttribute('style',e.style.cssText);
            if(e.tagName==='INPUT'){c.setAttribute('value',e.value);e.checked?c.setAttribute('checked',''):c.removeAttribute('checked');}
            if(e.tagName==='TEXTAREA')c.textContent=e.value;
            if(e.tagName==='OPTION')e.selected?c.setAttribute('selected',''):c.removeAttribute('selected');
            if(e.tagName==='IMG'&&e.currentSrc){c.setAttribute('src',e.currentSrc);c.removeAttribute('srcset');}
          }
          const styles=[];
          for(const sheet of document.styleSheets){
            if(sheet.disabled)continue;
            let css=null;try{css=[...sheet.cssRules].map(r=>r.cssText).join('\\n');}catch(_){}
            styles.push({css,href:sheet.href,media:sheet.media.mediaText});
          }
          copy.querySelectorAll('script,noscript,style,link[rel="stylesheet"],source,base,meta[http-equiv]').forEach(e=>e.remove());
          for(const e of [copy,...copy.querySelectorAll('*')])
            for(const a of [...e.attributes])if(a.name.toLowerCase().startsWith('on'))e.removeAttribute(a.name);
          const d=document.doctype;
          return {html:(d?new XMLSerializer().serializeToString(d):'')+copy.outerHTML,styles};
        }''')
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(payload['html'],'html.parser')
        for sheet in payload['styles']:
            css=sheet['css']
            if css is None:
                resource=self.resources.get(sheet['href'])
                if resource is None:raise ValueError('Cannot preserve external stylesheet: '+str(sheet['href']))
                css=resource['body'].decode('utf-8')
            def replace(match):
                value=match.group(1).strip().strip('"\'')
                if value.startswith(('data:','#')):return match.group(0)
                url=urljoin(sheet['href'] or '',value)
                resource=self._resource(url)
                uri='data:'+resource['content_type'].split(';')[0]+';base64,'+base64.b64encode(resource['body']).decode()
                return 'url("'+uri+'")'
            css=re.sub(r'url\(([^)]+)\)',replace,css)
            if re.search(r'@import\b',css,re.I):raise ValueError('Static bridge does not support nested CSS @import')
            tag=soup.new_tag('style');tag.string=css
            if sheet['media']:tag['media']=sheet['media']
            soup.head.append(tag)
        # Source images are normally already embedded benchmark assets. Never
        # write a snapshot that silently depends on the live network or blob URL.
        for image in soup.find_all('img'):
            src=image.get('src','')
            if src and not src.startswith('data:'):
                resource=self.resources.get(src)
                if not resource:raise ValueError('Cannot preserve image resource: '+src)
                image['src']='data:'+resource['content_type'].split(';')[0]+';base64,'+base64.b64encode(resource['body']).decode()
        return str(soup)

    def snapshot(self, html, viewport, screenshot=None, max_nodes=127, visual_geometry=True,
                 require_content=False):
        result=super().snapshot(html,viewport,screenshot,max_nodes,visual_geometry,require_content)
        result['html']=self._static_html()
        result['app_contract']=CONTRACT
        result['app_resources']=[{k:v for k,v in r.items() if k!='body'} for r in self.resources.values()]
        return result


def freeze_application(source,viewport,out_dir,cache_dir=None,max_nodes=2048,browser_host=None):
    """Execute source, export editable DOM/CSS and verify the *same* viewport."""
    import numpy as np
    from contextlib import nullcontext
    from PIL import Image
    started=time.perf_counter();out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    original=out/'application.html';static=out/'static.html'
    before=out/'application.png';after=out/'static.png'
    report=dict(source_html=str(original),static_html=str(static),screenshot=str(before),
                accepted=False,pixel_mae=None,max_box_error_px=None,
                contract=CONTRACT,resources=[],viewport=list(viewport))
    app=verification=None
    verification_start=(0,0)
    host_start=(browser_host.executions,browser_host.screenshots) if browser_host is not None else (0,0)

    def save_report():
        executions=app.executions if app is not None else 0
        captures=app.screenshots if app is not None else 0
        if verification is not None:
            executions+=verification.executions-verification_start[0]
            captures+=verification.screenshots-verification_start[1]
        report.update(seconds=time.perf_counter()-started,
                      browser_executions=executions,browser_screenshots=captures,
                      host_browser_executions=(browser_host.executions-host_start[0] if browser_host is not None else 0),
                      host_browser_screenshots=(browser_host.screenshots-host_start[1] if browser_host is not None else 0))
        if app is not None:
            report['resources']=[{k:v for k,v in r.items() if k!='body'} for r in app.resources.values()]
        (out/'freeze.json').write_text(json.dumps(report,indent=2))

    try:
        source=embed_application_assets(source)
        original.write_text(source)
        app=AppBrowser(cache_dir,browser_host=browser_host)
        with app as browser:
            observed=browser.snapshot(source,viewport,before,max_nodes,require_content=True)
            static.write_text(observed['html'])
            first=browser.page.locator('[data-tuide-app-node]').evaluate_all('els=>Object.fromEntries(els.map(e=>{const r=e.getBoundingClientRect();return [e.getAttribute("data-tuide-app-node"),[r.x,r.y,r.width,r.height]]}))')
        verification=browser_host if browser_host is not None else HtmlBrowser()
        verification_start=(verification.executions,verification.screenshots)
        with (nullcontext(verification) if browser_host is not None else verification) as browser:
            browser.snapshot(static.read_text(),viewport,after,max_nodes,require_content=True)
            second=browser.page.locator('[data-tuide-app-node]').evaluate_all('els=>Object.fromEntries(els.map(e=>{const r=e.getBoundingClientRect();return [e.getAttribute("data-tuide-app-node"),[r.x,r.y,r.width,r.height]]}))')
        if first.keys()!=second.keys():raise ValueError('Application static export changed DOM identities')
        max_error=max((abs(a-b) for key in first for a,b in zip(first[key],second[key])),default=0.)
        with Image.open(before) as a,Image.open(after) as b:
            pixel_mae=float(np.abs(np.asarray(a.convert('RGB'),dtype=float)-np.asarray(b.convert('RGB'),dtype=float)).mean()/255)
        report.update(pixel_mae=pixel_mae,max_box_error_px=max_error)
        if pixel_mae>.001 or max_error>1:
            raise ValueError(f'Application static export failed visual preservation: MAE={pixel_mae}, max_box_error_px={max_error}')
    except BaseException as exc:
        report.update(error=str(exc),error_type=type(exc).__name__)
        # Keep the original failure and attach accounting even when writing
        # diagnostics itself fails (for example, a disk-full interruption).
        try:
            save_report()
        except Exception as diagnostic_error:
            report['diagnostic_error']=str(diagnostic_error)
        exc.application_capture=report
        raise
    report['accepted']=True
    save_report()
    return report

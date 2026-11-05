"""Paired, frozen-corruption image-asset transfer pilot; no training or VLM calls."""
import argparse
import base64
import copy
import hashlib
import html as html_lib
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .ir import read_jsonl, write_json, write_jsonl

URL = re.compile(r'url\(\s*[\"\']?([^\s)\"\']+)[\"\']?\s*\)', re.I)


def css_image_urls(text):
    declarations=re.findall(r'(?:^|[;{])\s*(?:background(?:-image)?|border-image(?:-source)?|list-style(?:-image)?|(?:-webkit-)?mask(?:-image)?|content)\s*:\s*([^;}]+)',text,re.I)
    return [url for value in declarations for url in URL.findall(value)]


def image_references(text, base):
    """Only image attributes and inline CSS image URLs; never external CSS/fonts."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    tag = soup.find('base', href=True)
    if tag: base = urljoin(base, tag['href'])
    refs = []
    for tag in soup.find_all(['img', 'source', 'input']):
        if tag.name == 'input' and tag.get('type', '').lower() != 'image': continue
        if tag.name == 'source' and not tag.find_parent('picture'): continue
        if tag.get('src'): refs.append(tag['src'])
        if tag.get('srcset') and not tag['srcset'].lstrip().startswith('data:'):
            refs.extend(part.strip().split()[0] for part in tag['srcset'].split(',') if part.strip())
    for tag in soup.find_all(style=True): refs.extend(css_image_urls(tag['style']))
    for tag in soup.find_all('style'): refs.extend(css_image_urls(tag.get_text()))
    return {ref: urljoin(base, ref) for ref in dict.fromkeys(refs)
            if ref and not ref.startswith(('data:', '#', 'blob:'))
            and urlparse(urljoin(base, ref)).scheme in ('https', 'http')}


def fetch_image(url, cache, timeout):
    # Views share URL downloads across processes. Only a complete atomic JSON
    # becomes visible, and only one worker fetches a missing URL.
    from .cache_lock import cache_lock
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(url.encode()).hexdigest()
    with (cache/f'{key}.lock').open('a') as lock,cache_lock(lock):
        return _fetch_image_locked(url,cache,timeout)


def _fetch_image_locked(url,cache,timeout):
    from PIL import Image
    key = hashlib.sha256(url.encode()).hexdigest()
    path = cache / f'{key}.json'
    if path.exists(): return json.loads(path.read_text())
    result = dict(url=url, ok=False)
    try:
        start = time.monotonic()
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=timeout) as response:
            chunks=[];size=0
            while True:
                if time.monotonic()-start > timeout: raise TimeoutError('Image download deadline exceeded')
                part=response.read(65536)
                if not part: break
                size+=len(part)
                if size>10*1024*1024: raise ValueError('Image exceeds 10 MiB')
                chunks.append(part)
            data=b''.join(chunks)
        try:
            with Image.open(io.BytesIO(data)) as image:
                mime=Image.MIME[image.format];image.verify()
        except Exception:
            from xml.etree import ElementTree
            if ElementTree.fromstring(data).tag.split('}')[-1] != 'svg': raise ValueError('Not an image')
            mime='image/svg+xml'
        result.update(ok=True, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                      uri=f'data:{mime};base64,'+base64.b64encode(data).decode())
    except Exception as exc:
        result['error']=str(exc)
    from .preparation_workers import write_json as atomic_json
    atomic_json(path,result)
    return result


def embed(text, replacements):
    # Rewrite image-bearing attributes/CSS only: a relative filename must never
    # accidentally replace ordinary text, selectors, scripts or other URLs.
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(text,'html.parser')
    def css(value):
        return URL.sub(lambda m: 'url("'+replacements[m[1]]+'")' if m[1] in replacements else m[0],value)
    for tag in soup.find_all(['img','source','input']):
        if tag.name=='input' and tag.get('type','').lower()!='image':continue
        if tag.name=='source' and not tag.find_parent('picture'):continue
        if tag.get('src') in replacements:tag['src']=replacements[tag['src']]
        if tag.get('srcset') and not tag['srcset'].lstrip().startswith('data:'):
            parts=[]
            for part in tag['srcset'].split(','):
                fields=part.strip().split()
                if fields:fields[0]=replacements.get(fields[0],fields[0])
                parts.append(' '.join(fields))
            tag['srcset']=', '.join(parts)
    for tag in soup.find_all(style=True):tag['style']=css(tag['style'])
    for tag in soup.find_all('style'):
        tag.string=css(tag.get_text())
    return str(soup)


def prepare(args):
    from .html_bridge import HtmlBrowser
    from .fixed_test import boxes, pixel_mae
    from .html_feedback import refresh_geometry
    from .visual import annotate
    from . import css_owners
    from .web_experiment import digest, guard_run, signature
    data=Path(args.data);out=Path(args.out).resolve()
    guard_run(out,dict(contract='image-asset-pilot-v1',data=digest(data/'policy-test.jsonl'),
                       sources=digest(data.parent/'source/manifest-test.jsonl'),
                       count=args.count,max_candidates=args.max_candidates,max_images=args.max_images,
                       timeout=args.timeout),True)
    if (out/'complete.json').exists():
        print(f'Reusing prepared image pairs: {out}',flush=True);return
    cache=out/'downloads';cache.mkdir(exist_ok=True)
    sources={r['id']:r for r in read_jsonl(data.parent/'source/manifest-test.jsonl')}
    rows={mode:[] for mode in ('blocked','images')};pages={mode:[] for mode in rows}
    reports=[];seen=set();attempted=set()
    with HtmlBrowser() as browser:
        for row in read_jsonl(data/'policy-test.jsonl'):
            identity=row['id'].rsplit('/',1)[0]
            source=sources.get(identity)
            if source is None:continue
            page_id=source.get('page_id',identity.split('/')[0])
            if page_id in attempted:continue
            if len(attempted)>=args.max_candidates or len(seen)>=args.count:break
            attempted.add(page_id)
            audit=dict(id=row['id'],page_id=page_id,accepted=False)
            try:
                target=Path(row['target_html']).read_text();initial=Path(row['current_html']).read_text()
                url_file=Path(str(source['html']).replace('-html.html','-url.txt'))
                base=url_file.read_text().strip() if url_file.is_file() else ''
                refs=image_references(target,base)
                # Initial may contain additional URL spellings; use the same downloaded bytes.
                refs.update(image_references(initial,base))
                replacements={};downloads=[]
                for ref,url in list(refs.items())[:args.max_images]:
                    result=fetch_image(url,cache,args.timeout)
                    downloads.append({k:v for k,v in result.items() if k!='uri'})
                    if result['ok']:replacements[ref]=result['uri']
                audit.update(referenced_images=len(refs),downloaded_images=len(replacements),downloads=downloads)
                if not replacements:raise ValueError('No downloadable external image in frozen HTML')
                work=out/'pages'/signature(row['id'])[:20];work.mkdir(parents=True,exist_ok=True)
                prepared={};prepared_pages={}
                for mode in rows:
                    folder=work/mode;folder.mkdir(exist_ok=True)
                    target_text=embed(target,replacements if mode=='images' else {})
                    initial_text=embed(initial,replacements if mode=='images' else {})
                    target_path=folder/'target.html';target_path.write_text(target_text)
                    initial_path=folder/'initial.html';initial_path.write_text(initial_text)
                    browser.reset_context();browser.load(target_text,row['viewport'])
                    # Wait for embedded images to decode before measuring either condition.
                    browser.stable_capture(timeout_ms=15000)
                    target_boxes=boxes(browser,row['current'],row['viewport'])
                    target_state=css_owners.read(browser,row['current'])['state']
                    browser.page.screenshot(path=str(folder/'target.png'),animations='disabled')
                    browser.reset_context();browser.load(initial_text,row['viewport'])
                    browser.stable_capture(timeout_ms=15000)
                    current_boxes=boxes(browser,row['current'],row['viewport'])
                    tree=refresh_geometry(row['current'],current_boxes);annotate(browser,tree)
                    state=css_owners.read(browser,tree)['state']
                    browser.page.screenshot(path=str(folder/'initial.png'),animations='disabled')
                    new=copy.deepcopy(row)
                    new.update(current=tree,current_boxes=current_boxes,declaration_state=state,
                               target_declaration_state=target_state,target_html=str(target_path),
                               current_html=str(initial_path),target_image=str(folder/'target.png'),
                               current_image=str(folder/'initial.png'))
                    prepared[mode]=new
                    prepared_pages[mode]=dict(id=identity,evaluation_target_boxes=target_boxes)
                difference=pixel_mae(work/'blocked/target.png',work/'images/target.png')
                audit['target_image_change_mae']=difference
                if difference<=0:raise ValueError('Downloaded images produced no visible target change')
                for mode in rows:
                    rows[mode].append(prepared[mode]);pages[mode].append(prepared_pages[mode])
                seen.add(page_id);audit['accepted']=True
            except Exception as exc:
                from .runtime_errors import raise_if_fatal
                raise_if_fatal(exc);audit['error']=str(exc)
            reports.append(audit);write_jsonl(out/'screening.jsonl',reports)
            print(json.dumps({k:v for k,v in audit.items() if k!='downloads'}),flush=True)
    for mode in rows:
        folder=out/mode;folder.mkdir(exist_ok=True)
        write_jsonl(folder/'policy-test.jsonl',rows[mode]);write_jsonl(folder/'pages-test.jsonl',pages[mode])
    if not seen:raise ValueError('No usable image pairs; see screening.jsonl. No repair was run.')
    write_json(out/'complete.json',dict(selected=len(seen),requested=args.count,attempted=len(attempted)))
    print(f'Prepared {len(seen)}/{args.count} paired pages; all exclusions recorded.',flush=True)


def report(args):
    import statistics
    out=Path(args.out).resolve();results={}
    for mode in ('blocked','images'):
        results[mode]={r['id']:r for r in read_jsonl(out/mode/'repair/results.jsonl')}
    if set(results['blocked']) != set(results['images']):raise ValueError('Paired results differ; complete both rollouts')
    records=[];sections=[]
    for identity,left in results['blocked'].items():
        right=results['images'][identity]
        records.append(dict(id=identity,blocked=left,images=right,
                            transfer_delta=(right['final_box_iou']-right['initial_box_iou'])-
                                           (left['final_box_iou']-left['initial_box_iou'])))
        sections.append(f'<h2>{html_lib.escape(identity)}</h2>')
        for mode,row in (('blocked',left),('images',right)):
            work=Path(row['png']).parent
            sections.append(f'<h3>{mode}: IoU {row["initial_box_iou"]:.3f} → {row["final_box_iou"]:.3f}; failed={row["failed"]}</h3><div>')
            for label,path in [('Target',work/'target.png'),('Initial',work/'initial.png'),('Repaired',Path(row['png']))]:
                sections.append(f'<figure><figcaption>{label}</figcaption><img src="{path.relative_to(out).as_posix()}"></figure>')
            sections.append('</div>')
    write_jsonl(out/'paired-results.jsonl',records)
    lines=['| Condition | N | Initial Box IoU | Final Box IoU | Improved (%) | Failed |', '|---|---:|---:|---:|---:|---:|']
    for mode,group in results.items():
        values=list(group.values())
        lines.append(f'| {mode} | {len(values)} | {statistics.mean(r["initial_box_iou"] for r in values):.4f} | '
                     f'{statistics.mean(r["final_box_iou"] for r in values):.4f} | '
                     f'{100*statistics.mean(r["final_box_iou"]>r["initial_box_iou"] for r in values):.1f} | {sum(r["failed"] for r in values)} |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Image asset transfer</title>'
        '<style>body{font-family:sans-serif}div{display:flex}figure{width:32%;margin:6px}img{width:100%}</style>'
        '<h1>Frozen CSS corruption: image-only transfer pilot</h1>'+''.join(sections))
    print(out/'comparison.md');print(out/'index.html')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','report'])
    p.add_argument('--data',default='data/webui-full-dom-screened-7k-v1/7000/css')
    p.add_argument('--out',default='runs/webui-image-assets-pilot10')
    p.add_argument('--count',type=int,default=10)
    p.add_argument('--max-candidates',type=int,default=50)
    p.add_argument('--max-images',type=int,default=32)
    p.add_argument('--timeout',type=float,default=10)
    args=p.parse_args()
    if min(args.count,args.max_candidates,args.max_images,args.timeout)<=0:p.error('Limits must be positive')
    (prepare if args.stage=='prepare' else report)(args)


if __name__=='__main__':main()

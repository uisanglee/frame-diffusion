"""Offline, intrinsic-size-preserving five-colour WebUI image protocol."""
import argparse
import base64
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote_to_bytes
from bs4 import BeautifulSoup
from .ir import read_jsonl, write_json, write_jsonl

CONTRACT='webui-five-color-images-v1'
COLORS=('red','green','yellow','blue','black')
PROMPT=('Image assets are provided as solid-color placeholders: '+
        ', '.join(f'{c}=rick{i+1}.jpg' for i,c in enumerate(COLORS))+'. '
        'Use these exact filenames for the corresponding colored image regions. '
        'Set both image width and height with CSS to match the target screenshot; '
        'do not fetch external images. These files are available despite the no-external-dependencies rule.\n')

def intrinsic_uri(uri):
    from PIL import Image
    header,payload=uri.split(',',1)
    data=base64.b64decode(payload) if ';base64' in header else unquote_to_bytes(payload)
    try:
        with Image.open(io.BytesIO(data)) as image:return image.size
    except Exception:
        from xml.etree import ElementTree
        svg=ElementTree.fromstring(data)
        def pixels(value):return int(float(value.removesuffix('px')))
        # Refuse ambiguous percentages/viewBox-only SVG sizing.
        return pixels(svg.attrib['width']),pixels(svg.attrib['height'])

@lru_cache(maxsize=512)
def image_uri(index,width=1,height=1):
    width,height=int(width),int(height)
    if not (0<width<=20000 and 0<height<=20000 and width*height<=40_000_000):
        raise ValueError('Invalid or excessive intrinsic image dimensions')
    # SVG preserves intrinsic dimensions without allocating a full-size bitmap.
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="{COLORS[index]}"/></svg>'
    return 'data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()

def embed_placeholders(text):
    """Resolve only reserved asset references; leave unrelated HTML byte-identical."""
    if not any(f'rick{i}.jpg' in text for i in range(1,6)):return text
    soup=BeautifulSoup(text,'html.parser')
    names={f'rick{i+1}.jpg':i for i in range(5)}
    for tag in soup.find_all(['img','input']):
        if tag.get('src') in names:
            tag['src']=image_uri(names[tag['src']],tag.get('data-tuide-natural-width',1),tag.get('data-tuide-natural-height',1))
    from .image_asset_pilot import URL
    def css(value):
        return URL.sub(lambda m:'url("'+image_uri(names[m[1]])+'")' if m[1] in names else m[0],value)
    for tag in soup.find_all(style=True):tag['style']=css(tag['style'])
    for tag in soup.find_all('style'):tag.string=css(tag.get_text())
    return str(soup)

def convert_html(text,base,cache,browser,viewport):
    from .image_asset_pilot import image_references,fetch_image,embed
    measure='''()=>[...document.querySelectorAll('body,body *')].filter(e=>!['STYLE','SCRIPT'].includes(e.tagName)).map(e=>{
      const r=e.getBoundingClientRect();return {box:[r.x,r.y,r.width,r.height]}})'''
    # Let Chromium select picture/srcset and active CSS for this viewport before
    # downloading. Network remains blocked during this selection pass.
    eager=BeautifulSoup(text,'html.parser')
    for tag in eager.find_all('img'):tag['loading']='eager'
    text=str(eager)
    refs=image_references(text,base);replacements={};downloads={}
    browser.load(embed(text,refs),viewport)
    selected=browser.page.evaluate('''()=>{
      const urls=[...document.querySelectorAll('img,input[type=image]')].map(e=>e.currentSrc||e.src);
      const props=['backgroundImage','borderImageSource','listStyleImage','maskImage','webkitMaskImage','content'];
      for(const e of document.querySelectorAll('body,body *')) {
        if(!e.getClientRects().length)continue;
        for(const pseudo of [null,'::before','::after']) {
          const s=getComputedStyle(e,pseudo);
          for(const p of props)for(const m of (s[p]||'').matchAll(/url\\(["']?([^"')]+)["']?\\)/g))urls.push(m[1]);
        }
      }
      return [...new Set(urls)];
    }''')
    for url in dict.fromkeys(refs.values()):
        if url not in selected:continue
        result=fetch_image(url,cache,10)
        if not result['ok']:raise ValueError(f'Image unavailable: {url}: {result.get("error")}')
        downloads[url]=result['uri']
    replacements={ref:downloads[url] for ref,url in refs.items() if url in downloads}
    original=embed(text,replacements)
    browser.load(original,viewport);browser.stable_capture()
    before=browser.page.evaluate(measure)
    # Use active picture/srcset source selected by the viewport. Freeze it for this view.
    images=browser.page.evaluate('''()=>[...document.querySelectorAll('img,input[type=image]')].map(e=>({
      width:e.naturalWidth||0,height:e.naturalHeight||0,src:e.currentSrc||e.src}))''')
    soup=BeautifulSoup(original,'html.parser');mapping=[]
    tags=[t for t in soup.find_all(['img','input']) if t.name=='img' or t.get('type','').lower()=='image']
    if len(tags)!=len(images):raise ValueError('Image DOM extraction mismatch')
    for i,(tag,info) in enumerate(zip(tags,images)):
        # HTMLInputElement has no naturalWidth/naturalHeight. Read the decoded
        # source dimensions, not its CSS box (which may already scale the image).
        if tag.name=='input' and info['src'].startswith('data:image/'):
            info['width'],info['height']=intrinsic_uri(info['src'])
        if not info['width'] or not info['height']:raise ValueError('Image has no decoded intrinsic size')
        name=f'rick{i%5+1}.jpg'
        mapping.append(dict(index=i,file=name,color=COLORS[i%5],**info))
        tag['src']=name;tag.attrs.pop('srcset',None);tag.attrs.pop('sizes',None)
        tag['data-tuide-natural-width']=str(info['width']);tag['data-tuide-natural-height']=str(info['height'])
        if tag.parent.name=='picture':
            for source in tag.parent.find_all('source'):source.attrs.pop('srcset',None)
    # Backgrounds are different from image elements; preserve their intrinsic sizing too.
    from .image_asset_pilot import URL
    counter=len(tags)
    def css(value):
        nonlocal counter
        def replace(m):
            nonlocal counter
            uri=m[1]
            if not uri.startswith('data:image/'):return m[0]
            dims=intrinsic_uri(uri)
            result='url("'+image_uri(counter%5,*dims)+'")';counter+=1;return result
        return URL.sub(replace,value)
    for tag in soup.find_all(style=True):tag['style']=css(tag['style'])
    for tag in soup.find_all('style'):tag.string=css(tag.get_text())
    if not soup.html:raise ValueError('Expected complete source HTML')
    soup.html['data-tuide-image-contract']=CONTRACT
    result=str(soup)
    browser.load(result,viewport);browser.stable_capture()
    after=browser.page.evaluate(measure)
    if len(before)!=len(after) or any(abs(a-b)>1 for x,y in zip(before,after) for a,b in zip(x['box'],y['box'])):
        raise ValueError('Placeholder replacement changed DOM geometry')
    return result,mapping

def prepare(args):
    from .html_bridge import HtmlBrowser
    from .web_experiment import digest,guard_run
    source=Path(args.manifest).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    guard_run(out,dict(contract=CONTRACT,source=digest(source),implementation=digest(__file__)),True)
    cache=out/'downloads';cache.mkdir(exist_ok=True)
    from PIL import Image
    for i,color in enumerate(COLORS):
        Image.new('RGB',(1,1),color).save(out/f'rick{i+1}.jpg',quality=100,subsampling=0)
    rows=[];errors=[]
    with HtmlBrowser() as browser:
        for row in read_jsonl(source):
            folder=out/'pages'/hashlib.sha256(row['id'].encode()).hexdigest();folder.mkdir(parents=True,exist_ok=True)
            record=folder/'record.json'
            if record.exists():
                saved=json.loads(record.read_text())
                if saved.get('raw_input_sha')!=digest(row['html']) or saved['source_sha']!=digest(saved['html']):
                    raise ValueError('Source/output HTML changed; use a new output directory')
                if not Path(saved['screenshot']).is_file():raise ValueError('Missing cached placeholder screenshot')
                rows.append(saved);continue
            try:
                path=Path(row['html']);text=path.read_text()
                if 'data-tuide-safe-sizes' in text:raise ValueError('Use raw source HTML, not already-normalized HTML')
                url=Path(str(path).replace('-html.html','-url.txt'))
                base=url.read_text().strip() if url!=path and url.exists() else row.get('url','')
                result,mapping=convert_html(text,base,cache,browser,row['viewport'])
                target=folder/'target.html';target.write_text(result)
                screenshot=folder/'target.png';browser.page.screenshot(path=str(screenshot),animations='disabled')
                new={**row,'html':str(target),'screenshot':str(screenshot),'source_sha':digest(target),
                     'original_source_sha':row.get('source_sha'),
                     'raw_input_sha':digest(path),
                     'image_asset_contract':CONTRACT,'original_html':str(path)}
                new.pop('source_files',None)
                write_json(folder/'images.json',mapping);write_json(record,new);rows.append(new)
                print(json.dumps({'id':row['id'],'status':'prepared','images':len(mapping)}),flush=True)
            except Exception as error:
                errors.append(dict(id=row['id'],error=str(error)));print(json.dumps(errors[-1]),flush=True)
    write_jsonl(out/'manifest.jsonl',rows);write_jsonl(out/'rejections.jsonl',errors)
    for split in ('train','val','test'):write_jsonl(out/f'manifest-{split}.jsonl',[r for r in rows if r['split']==split])
    write_json(out/'report.json',dict(kept=len(rows),failed=len(errors),contract=CONTRACT))
    if not rows:raise ValueError('No pages prepared')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    prepare(p.parse_args())

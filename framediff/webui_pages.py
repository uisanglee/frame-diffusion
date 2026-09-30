"""WebUI raw screenshots -> real HTML experiment inputs (not fitted synthetic pairs)."""
import math
import re
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from .adapters import load_gzip


def clipped_boxes(boxes, viewport, exclude=()):
    w,h = viewport; result = {}
    for key,box in boxes.items():
        if key in exclude: continue
        x,y,bw,bh = box
        if not all(math.isfinite(v) for v in box) or min(bw,bh)<=0: continue
        x1,y1 = max(0,x),max(0,y); x2,y2 = min(w,x+bw),min(h,y+bh)
        if x2>x1 and y2>y1: result[key] = [x1,y1,x2-x1,y2-y1]
    return result


def discover_webui(root, view='default_1280-720', limit=0):
    root = Path(root).resolve(); result = []
    images = sorted([*root.rglob('*-screenshot.webp'),*root.rglob('*-screenshot.png')])
    for image in images:
        prefix = image.stem.removesuffix('-screenshot')
        if view!='all' and prefix!=view: continue
        stem = image.parent/prefix
        def sibling(suffix): return Path(str(stem)+suffix)
        ax_path,bb_path = sibling('-axtree.json.gz'),sibling('-bb.json.gz')
        html_path,url_path = sibling('-html.html'),sibling('-url.txt')
        with Image.open(image) as source: viewport = list(source.size)
        url = url_path.read_text().strip() if url_path.exists() else ''
        item = {'id':image.relative_to(root).as_posix(),'dataset':'webui','screenshot':str(image),
                'html':str(html_path) if html_path.exists() else None,
                'group':urlparse(url).hostname or image.parent.relative_to(root).as_posix(),
                'reference_kind':'webui_recorded_boxes','viewport':viewport,'webui_view':prefix,
                'reference_boxes':{},'reference_box_error':None,
                'source_files':[str(p) for p in (ax_path,bb_path,url_path) if p.exists()]}
        try:
            size = re.fullmatch(r'default_(\d+)-(\d+)',prefix)
            if size and viewport!=[int(size[1]),int(size[2])]:
                raise ValueError('Screenshot dimensions differ from CSS viewport; DPR scaling requires explicit alignment')
            nodes = load_gzip(ax_path)['nodes']; boxes = load_gzip(bb_path)
            roots = {n['nodeId'] for n in nodes if n.get('role',{}).get('value')=='RootWebArea'}
            if not roots: raise ValueError('Missing AX RootWebArea; cannot identify reference root')
            targets = {}
            for n in nodes:
                b = boxes.get(str(n.get('backendDOMNodeId')))
                if n.get('ignored') or n['nodeId'] in roots or not b: continue
                targets[str(n['nodeId'])] = [float(b[k]) for k in ('x','y','width','height')]
            if not size:
                root_box = next((boxes.get(str(n.get('backendDOMNodeId'))) for n in nodes if n['nodeId'] in roots),None)
                if root_box is None or abs(root_box['width']-viewport[0])>1:
                    raise ValueError('Cannot establish screenshot/CSS coordinate scale for this device')
            item['reference_boxes'] = clipped_boxes(targets,viewport)
            if not item['reference_boxes']: raise ValueError('No visible recorded reference boxes')
        except (OSError,EOFError,KeyError,ValueError,TypeError) as error:
            # Keep the screenshot/page. Image evaluation can still run; record missing geometry.
            item['reference_box_error'] = str(error)
        result.append(item)
        if limit and len(result)>=limit: break
    if not result:
        raise ValueError(f'No WebUI *-screenshot.webp/png for view={view} under {root}; use extracted raw files')
    return result

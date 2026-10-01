"""Domain-disjoint WebUI sampling for visual detector/policy fine-tuning."""
import gzip
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from .ir import write_json,write_jsonl
from .visual import CONTRACT
from .web_experiment import digest,guard_run


TEXT_ROLES={'StaticText','InlineTextBox'}
IMAGE_ROLES={'img','image','graphics-object','canvas'}
CONTROL_ROLES={'button','textbox','searchbox','checkbox','radio','combobox','slider','switch','spinbutton','listbox'}


def _read_gzip(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def _native_elements(ax_path,bb_path,viewport):
    nodes=_read_gzip(ax_path).get('nodes',[]);boxes=_read_gzip(bb_path);w,h=viewport
    result=[];seen=set()
    for node in nodes:
        if node.get('ignored'):continue
        role=node.get('role',{}).get('value','')
        label=1 if role in TEXT_ROLES else 2 if role in IMAGE_ROLES else 3 if role in CONTROL_ROLES else 0
        box=boxes.get(str(node.get('backendDOMNodeId')))
        if not label or not box:continue
        x,y,bw,bh=(float(box[k]) for k in ('x','y','width','height'))
        x1,y1,x2,y2=max(0.,x),max(0.,y),min(float(w),x+bw),min(float(h),y+bh)
        key=(label,round(x1,2),round(y1,2),round(x2,2),round(y2,2))
        if x2<=x1 or y2<=y1 or key in seen:continue
        seen.add(key);result.append({'box':[x1,y1,x2,y2],'label':label})
    return result


def _assign(groups,counts,seed):
    rng=random.Random(seed);items=list(groups.items());rng.shuffle(items)
    selected={split:[] for split in counts};assigned={}
    for group,rows in items:
        available=[s for s in counts if len(selected[s])<counts[s]]
        if not available:break
        split=max(available,key=lambda s:(counts[s]-len(selected[s]))/counts[s])
        rng.shuffle(rows);take=min(len(rows),counts[split]-len(selected[split]))
        selected[split].extend(rows[:take]);assigned[group]=split
    missing={s:counts[s]-len(selected[s]) for s in counts if len(selected[s])<counts[s]}
    if missing:raise ValueError(f'Not enough domain-disjoint WebUI pages for requested split: {missing}')
    return selected,assigned


def import_webui(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    if min(args.train_count,args.val_count,args.test_count)<1:raise ValueError('All split counts must be positive')
    images=sorted([*root.rglob(f'{args.view}-screenshot.webp'),*root.rglob(f'{args.view}-screenshot.png')])
    candidates=[];excluded=Counter();seen_hashes=set()
    for image in images:
        stem=image.parent/args.view
        html=Path(str(stem)+'-html.html');url_path=Path(str(stem)+'-url.txt')
        ax=Path(str(stem)+'-axtree.json.gz');bb=Path(str(stem)+'-bb.json.gz')
        if not html.exists():excluded['missing_html']+=1;continue
        source_hash=digest(html)
        if source_hash in seen_hashes:excluded['duplicate_html']+=1;continue
        seen_hashes.add(source_hash)
        url=url_path.read_text(errors='replace').strip() if url_path.exists() else ''
        host=(urlparse(url).hostname or '').lower()
        group='webui-domain:'+host if host else 'webui-page:'+image.parent.relative_to(root).as_posix()
        with Image.open(image) as source:viewport=list(source.size)
        expected=args.view.removeprefix('default_').split('-')
        if len(expected)==2 and all(v.isdigit() for v in expected) and viewport!=list(map(int,expected)):
            excluded['viewport_mismatch']+=1;continue
        candidates.append({'id':image.relative_to(root).as_posix(),'group':group,'html':str(html),
            'screenshot':str(image),'viewport':viewport,'source_sha':source_hash,'ax':ax,'bb':bb})
    grouped=defaultdict(list)
    for row in candidates:grouped[row['group']].append(row)
    counts={'train':args.train_count,'val':args.val_count,'test':args.test_count}
    selected,assigned=_assign(grouped,counts,args.seed)
    sources=[];native_rows=[]
    for split,rows in selected.items():
        for row in rows:
            source={k:row[k] for k in ('id','group','html','screenshot','viewport','source_sha')}
            source['split']=split;sources.append(source)
            native=[]
            if row['ax'].exists() and row['bb'].exists():
                try:native=_native_elements(row['ax'],row['bb'],row['viewport'])
                except (OSError,EOFError,ValueError,KeyError,TypeError):excluded['invalid_ax_boxes']+=1
            else:excluded['missing_ax_boxes']+=1
            if native:
                native_rows.append({'id':row['id'],'group':row['group'],'split':split,
                    'source_sha':row['source_sha'],'contract':CONTRACT,'viewport':row['viewport'],
                    'image':row['screenshot'],'elements':native,
                    'annotation_scope':'webui-ax-semantic-v1-no-painted-region'})
    config={'kind':'webui-visual-import-v1','root':str(root),'view':args.view,'seed':args.seed,
        'counts':counts,'sources':[{k:r[k] for k in ('id','group','source_sha','split')} for r in sources]}
    guard_run(out,config,args.resume)
    order={'train':0,'val':1,'test':2};sources.sort(key=lambda r:(order[r['split']],r['id']))
    native_rows.sort(key=lambda r:(order[r['split']],r['id']))
    write_jsonl(out/'manifest.jsonl',sources)
    for split in counts:
        write_jsonl(out/f'manifest-{split}.jsonl',[r for r in sources if r['split']==split])
        write_jsonl(out/f'native-detector-{split}.jsonl',[r for r in native_rows if r['split']==split])
    report={'selected':{s:sum(r['split']==s for r in sources) for s in counts},
        'selected_domains':{s:len({r['group'] for r in sources if r['split']==s}) for s in counts},
        'native_detector_pages':{s:sum(r['split']==s for r in native_rows) for s in counts},
        'native_class_boxes':dict(Counter(e['label'] for r in native_rows for e in r['elements'])),
        'available_images':len(images),'eligible_unique_html':len(candidates),'excluded':dict(excluded),
        'note':'Native AX labels cover text/image/control only; painted-region requires browser-rendered labels.'}
    write_json(out/'report.json',report);print(json.dumps(report,indent=2));return sources

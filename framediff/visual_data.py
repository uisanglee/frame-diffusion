"""Browser-produced supervision shared by both visual policy conditions."""
import copy
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .ir import read_json,read_jsonl
from .preparation_workers import page_map,write_json,write_jsonl
from .html_bridge import HtmlBrowser,embed_placeholder
from .html_feedback import refresh_geometry
from .plans import dom_tree
from .visual import (CONTRACT,ACTION_CONTRACT,ACTION_VALUES,
                     annotate,elements,abstract_image,VISUAL_GEOMETRY_CONTRACT)
from .web_experiment import guard_run,digest


def validate_splits(records):
    assignments={}
    for r in records:
        if not r.get('group') or r.get('split') not in ('train','val','test'):
            raise ValueError('Explicit group and train/val/test split required')
        for key in ('group','source_sha','page_id'):
            value=r.get(key)
            if value is None:continue
            token=(key,value)
            if token in assignments and assignments[token]!=r['split']:raise ValueError(f'Cross-split leakage: {key}')
            assignments[token]=r['split']


def generate(args):
    """Self-contained styled HTML bootstrap corpus; not an actual-web benchmark."""
    if args.count<30:raise ValueError('Use at least 30 pages for train/val/test generation')
    out=Path(args.out).resolve()
    guard_run(out,{'kind':'visual-html-generator-v1','count':args.count,'seed':args.seed},False)
    records=[]
    for i in range(args.count):
        rng=random.Random(args.seed+i);w,h=rng.choice([(1280,800),(960,720),(720,960)])
        width=rng.choice([160,200,240]);count=rng.randint(3,8);gap=rng.choice([8,12,20,24])
        hue=rng.randrange(360);cards=[]
        for j in range(count):
            text=' '.join(rng.choices(['Design','research','layout','browser','visual','element','screen','sample'],k=rng.randint(6,22)))
            cards.append(f'<article style="width:{width}px;padding:12px;background:white;border:1px solid #aaa;border-radius:6px">'
                         f'<svg width="{width-24}" height="{rng.choice([60,80,100])}" viewBox="0 0 200 100"><rect width="200" height="100" fill="hsl({(hue+j*35)%360},45%,72%)"/><circle cx="90" cy="50" r="30" fill="#fff"/></svg>'
                         f'<h3>Card {j+1}</h3><p>{text}</p><button>View {j+1}</button></article>')
        html=f'''<!doctype html><html><head><meta charset="utf-8"><style>
        *{{box-sizing:border-box}}body{{margin:0;background:#f2f4f7;font:{rng.choice([14,16,18])}px/{rng.choice([1.3,1.5])} sans-serif;color:#253247}}
        header{{background:hsl({hue},45%,25%);color:white;padding:16px;display:flex;justify-content:space-between;align-items:center}}
        nav{{display:flex;gap:16px}}main{{margin:20px auto;width:{rng.choice([82,90,96])}%;}}
        section{{display:flex;flex-wrap:wrap;gap:{gap}px;align-items:flex-start}}
        button{{background:#2455aa;color:white;border:0;padding:8px 12px;border-radius:4px}}
        p{{margin:8px 0}}h3{{margin:12px 0}}footer{{padding:20px;background:#dce2eb}}
        </style></head><body><header><strong>Studio {i}</strong><nav><span>Home</span><span>Projects</span><button>Search</button></nav></header>
        <main><h1>Collection {i}</h1><p>Layout experiment {args.seed+i}</p><section>{''.join(cards)}</section></main>
        <footer>Contact and information {i}</footer></body></html>'''
        path=out/f'page-{i:05d}.html';path.write_text(html)
        split='test' if i%10==0 else 'val' if i%10==1 else 'train'
        records.append({'id':f'procedural-{args.seed+i}','group':f'procedural-{args.seed+i}','split':split,
                        'html':str(path),'viewport':[w,h]})
    write_jsonl(out/'manifest.jsonl',records)
    return records


def _render_page(task):
    args, out, index, source = task
    geometry_contract=getattr(args,'visual_geometry_contract',VISUAL_GEOMETRY_CONTRACT)
    if getattr(args,'lazy_train',False) and source['split']=='train':
        from .lazy_webui import CONTRACT as LAZY
        settings={k:getattr(args,k) for k in ('seed','max_nodes','abstract_size','min_elements','max_source_mae')}
        settings.update(resume=True,clean_only=True,lazy_train=False,visual_geometry_contract=geometry_contract)
        row={**source,'id':source['id']+'/clean','lazy_contract':LAZY,
             'visual_geometry_contract':geometry_contract,
             'lazy_source':source,'lazy_settings':settings,'contract':CONTRACT}
        return {'page':{**row,'id':source['id']},'policy':[row],'detection':[row]},None,False
    work=out/'pages'/hashlib.sha256(source['id'].encode()).hexdigest()[:20]
    work.mkdir(parents=True,exist_ok=True)
    cache=work/'records.json'
    if args.resume and cache.exists():
        try:
            saved=read_json(cache)
            assets={saved['page']['html'], saved['page']['screenshot']}
            for record in saved['policy']:
                assets.update(record[k] for k in ('current_html','current_image','current_abstract',
                                                   'target_image','target_abstract'))
            if (saved['page']['source_sha']==source['source_sha'] and saved['policy'] and
                saved['detection'] and saved.get('visual_geometry_contract')==geometry_contract
                and all(Path(path).is_file() for path in assets)):
                return saved, None, True
        except (OSError, ValueError, KeyError, TypeError):
            pass  # An interrupted/invalid completion record is regenerated.
    rng=random.Random(args.seed+index);local_policy=[];local_detection=[]
    try:
        with HtmlBrowser() as browser:
            viewport=source.get('viewport',[1280,800]);raw=Path(source['html']).read_text()
            asset=Path(source['html']).parent/'rick.jpg'
            if asset.exists():raw=embed_placeholder(raw,asset)
            dom=browser.snapshot(raw,viewport,work/'target.png',args.max_nodes-1,visual_geometry=bool(geometry_contract))
            tree,target_boxes,_=dom_tree(dom);annotate(browser,tree,visual_boxes=bool(geometry_contract))
            from .explicit_html import feedback_metadata
            target_feedback=feedback_metadata(browser,tree)
            target_elements=elements(tree,target_boxes,viewport,target_feedback.get('current_clip_boxes'))
            min_elements=getattr(args,'min_elements',1)
            if len(target_elements)<min_elements:raise ValueError(f'Only {len(target_elements)} visible abstraction elements')
            source_mae=None
            if source.get('screenshot'):
                with Image.open(source['screenshot']) as reference,Image.open(work/'target.png') as rendered:
                    if reference.size!=rendered.size:raise ValueError('Source screenshot and rerender dimensions differ')
                    source_mae=float(np.abs(np.asarray(reference.convert('RGB'),dtype=np.float32)-
                        np.asarray(rendered.convert('RGB'),dtype=np.float32)).mean()/255)
                max_source_mae=getattr(args,'max_source_mae',1.)
                if source_mae>max_source_mae:
                    raise ValueError(f'Source rerender pixel MAE {source_mae:.4f} exceeds {max_source_mae}')
            tagged=dom['html'];(work/'reference.html').write_text(tagged)
            ids=[n['id'] for n in tree['nodes'][1:]];root=tree['nodes'][0]['id']
            base={k:source[k] for k in ('id','group','split','source_sha')}
            for key in ('page_id','view'):
                if key in source:base[key]=source[key]
            base.update(contract=CONTRACT,action_contract=ACTION_CONTRACT,visual_geometry_contract=geometry_contract,
                        viewport=viewport,target_image=str(work/'target.png'),
                        target_abstract=str(work/'target-abstract.png'),
                        teacher_strategy='unlabeled-css-corruption-v1',target_elements=target_elements)
            abstract_image(target_elements,viewport,args.abstract_size).save(base['target_abstract'])
            def observe():
                boxes=browser.tagged_boxes(ids);boxes[root]=[0,0,*viewport];return boxes
            def save_example(name,current_html,boxes,corruption=None):
                from .explicit_html import feedback_screenshot
                screenshot=work/f'{name}.png';feedback_screenshot(browser,path=str(screenshot),animations='disabled')
                browser.screenshots+=1
                boxes=observe()
                current_tree=refresh_geometry(tree,boxes)
                abstract=work/f'{name}-abstract.png'
                feedback=feedback_metadata(browser,current_tree)
                labels=elements(current_tree,boxes,viewport,feedback.get('current_clip_boxes'))
                abstract_image(labels,viewport,args.abstract_size).save(abstract)
                html_path=work/f'{name}.html';html_path.write_text(current_html)
                record={**base,'id':source['id']+'/'+name,'current':current_tree,'current_boxes':copy.deepcopy(boxes),
                    'current_image':str(screenshot),'current_abstract':str(abstract),'current_html':str(html_path),
                    'corruption_edit':corruption,**feedback}
                local_policy.append(record)
                local_detection.append({**base,'id':record['id'],'image':str(screenshot),'elements':labels})
            browser.load(tagged,viewport);save_example('clean',tagged,target_boxes)
            # Screenshot waits for fonts. Use geometry measured after that same
            # capture for both supervision and the clean corruption reference.
            clean=local_policy[0]
            tree=copy.deepcopy(clean['current']);target_boxes=copy.deepcopy(clean['current_boxes'])
            target_elements=local_detection[0]['elements']
            base['target_elements']=target_elements
            for record in (local_policy[0],local_detection[0]):record['target_elements']=target_elements
            (work/'target.png').write_bytes(Path(clean['current_image']).read_bytes())
            abstract_image(target_elements,viewport,args.abstract_size).save(base['target_abstract'])
            last_corrupted=None
            for trajectory in range(0 if getattr(args,'clean_only',False) else args.trajectories):
                html=tagged
                for step in range(args.max_noise):
                    accepted=False
                    for attempt in range(30):
                        from .tree_edits import FIELDS
                        i=rng.randrange(1,len(tree['nodes']));fields=FIELDS
                        if not fields:continue
                        field=rng.choice(fields)
                        value=rng.choice(ACTION_VALUES[field]);nid=tree['nodes'][i]['id']
                        try:browser.edit_visual_action(html,viewport,nid,field,value)
                        except Exception as exc:
                            if isinstance(exc,ValueError) or 'Explicit ' in str(exc):continue
                            raise
                        candidate_html=browser.page.content();candidate=observe()
                        # Corruption validity is syntactic/executable, not
                        # "farther from target pixels". The teacher will diff
                        # actual declarations later, never negate this delta.
                        if candidate_html==html:continue
                        save_example(f't{trajectory}-s{step}',candidate_html,candidate,[i,field,value])
                        html=candidate_html;accepted=True
                        last_corrupted=local_policy[-1];break
                    if not accepted:break
            if getattr(args,'clean_only',False):last_corrupted=local_policy[0]
            if last_corrupted is None:raise ValueError('No verified corruption; page excluded, not silently clean-only')
            page={**base,'screenshot':base['target_image'],'html':str(work/'reference.html'),
                  'initial_html':last_corrupted['current_html'],'evaluation_target_boxes':target_boxes,
                  'total_visible_nodes':dom['total_visible_nodes'],'selected_nodes':len(ids),
                  'source_screenshot':source.get('screenshot'),'source_pixel_mae':source_mae,
                  'construction':('clean normalization input; no corruption' if getattr(args,'clean_only',False)
                                  else 'controlled same-DOM corruption; not screenshot-generated initial HTML')}
            saved={'page':page,'policy':local_policy,'detection':local_detection,
                   'visual_geometry_contract':geometry_contract};write_json(cache,saved)
        return saved, None, False
    except Exception as error:
        failure={'id':source['id'],'error':str(error)}
        write_json(work/'failure.json',failure)
        return None, failure, False


def build(args):
    if getattr(args,'lazy_train',False) and not getattr(args,'clean_only',False):
        raise ValueError('--lazy-train requires --clean-only; corruption is generated online')
    workers=getattr(args,"workers",1)
    if workers<1:raise ValueError("workers must be positive")
    sources=list(read_jsonl(args.manifest))
    if not sources or min(args.trajectories,args.max_noise)<1:raise ValueError('Need nonempty manifest and positive trajectories/noise')
    sources=[{**r,'source_sha':digest(r['html'])} for r in sources];validate_splits(sources)
    if len({r['id'] for r in sources})!=len(sources):raise ValueError('Duplicate source IDs')
    out=Path(args.out).resolve()
    config={'kind':'visual-data-v7-cached-css-corruptions','contract':CONTRACT,'action_contract':ACTION_CONTRACT,
            'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT,
            'sources':sources,'settings':{k:v for k,v in vars(args).items() if k not in ('out','resume','workers')}}
    if not getattr(args,'clean_only',False):config['settings'].pop('clean_only',None)
    if any(r.get('hierarchy') for r in sources):raise ValueError('Use original parent-preserving HTML, not flat manifests')
    guard_run(out,config,args.resume)
    policy=[];detection=[];pages=[];errors=[]
    cache_hits=0
    tasks=((args,out,index,source) for index,source in enumerate(sources))
    for index,(saved,error,reused) in enumerate(page_map(_render_page,tasks,workers)):
        if error:
            errors.append(error)
        else:
            policy+=saved['policy'];detection+=saved['detection'];pages.append(saved['page'])
            cache_hits+=int(reused)
        print(f'[{index+1}/{len(sources)}] visual examples={len(policy)}, failures={len(errors)}, cache_hits={cache_hits}',flush=True)
        write_json(out/'progress.json',dict(processed=index+1,total=len(sources),
            pages=len(pages),failures=len(errors),cache_hits=cache_hits,workers=workers))
    for split in ('train','val','test'):
        write_jsonl(out/f'policy-{split}.jsonl',[r for r in policy if r['split']==split])
        write_jsonl(out/f'detector-{split}.jsonl',[r for r in detection if r['split']==split])
        write_jsonl(out/f'pages-{split}.jsonl',[r for r in pages if r['split']==split])
    coverage={}
    for split in ('train','val','test'):
        source_views=[r for r in sources if r['split']==split];usable_views=[r for r in pages if r['split']==split]
        selected_pages={r.get('page_id',r['id']) for r in source_views}
        usable_pages={r.get('page_id',r['id']) for r in usable_views}
        coverage[split]={'selected':len(selected_pages),'selected_views':len(source_views),
            'usable':len(usable_pages),'usable_views':len(usable_views),
            'excluded':len(selected_pages-usable_pages)}
        if any(r.get('lazy_contract') for r in usable_views):
            coverage[split].update(status='deferred',pending_pages=len(usable_pages),pending_views=len(usable_views),
                usable=None,usable_views=None,excluded=None)
    write_json(out/'report.json',{'pages':len(pages),'policy_examples':len(policy),'detector_examples':len(detection),
                                'split_coverage':coverage,'errors':errors})
    if not policy:raise ValueError('No usable visual training examples; see report.json')
    return policy

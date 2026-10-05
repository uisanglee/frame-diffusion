"""Browser-produced supervision shared by both visual policy conditions."""
import copy
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .ir import read_jsonl,write_json,write_jsonl
from .html_bridge import HtmlBrowser,embed_placeholder
from .html_feedback import refresh_geometry
from .plans import dom_tree
from .visual import (CONTRACT,ACTION_CONTRACT,ACTION_VALUES,candidate_fields,action_is_legal,
                     annotate,elements,abstract_image)
from .web_experiment import guard_run,digest


def validate_splits(records):
    assignments={}
    for r in records:
        if not r.get('group') or r.get('split') not in ('train','val','test'):
            raise ValueError('Explicit group and train/val/test split required')
        for key in ('group','source_sha'):
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


def build(args):
    sources=list(read_jsonl(args.manifest))
    if not sources or min(args.trajectories,args.max_noise)<1:raise ValueError('Need nonempty manifest and positive trajectories/noise')
    sources=[{**r,'source_sha':digest(r['html'])} for r in sources];validate_splits(sources)
    if len({r['id'] for r in sources})!=len(sources):raise ValueError('Duplicate source IDs')
    out=Path(args.out).resolve()
    config={'kind':'visual-data-v4-best-reverse','contract':CONTRACT,'action_contract':ACTION_CONTRACT,
            'sources':sources,'settings':{k:v for k,v in vars(args).items() if k not in ('out','resume')}}
    guard_run(out,config,args.resume)
    policy=[];detection=[];pages=[];errors=[]
    with HtmlBrowser() as browser:
        for index,source in enumerate(sources):
            work=out/'pages'/hashlib.sha256(source['id'].encode()).hexdigest()[:20];work.mkdir(parents=True,exist_ok=True)
            cache=work/'records.json'
            if args.resume and cache.exists():
                from .ir import read_json
                saved=read_json(cache);policy+=saved['policy'];detection+=saved['detection'];pages.append(saved['page']);continue
            rng=random.Random(args.seed+index);local_policy=[];local_detection=[]
            try:
                # A renderer crash must affect at most one source page. A fresh
                # context also bounds memory retained by large real-world DOMs.
                browser.reset_context()
                viewport=source.get('viewport',[1280,800]);raw=Path(source['html']).read_text()
                asset=Path(source['html']).parent/'rick.jpg'
                if asset.exists():raw=embed_placeholder(raw,asset)
                dom=browser.snapshot(raw,viewport,work/'target.png',args.max_nodes-1)
                tree,target_boxes,_=dom_tree(dom);annotate(browser,tree)
                target_elements=elements(tree,target_boxes,viewport)
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
                base.update(contract=CONTRACT,action_contract=ACTION_CONTRACT,viewport=viewport,target_image=str(work/'target.png'),
                            target_abstract=str(work/'target-abstract.png'),
                            teacher_strategy='best-improving-reverse-v1',target_elements=target_elements)
                abstract_image(elements(tree,target_boxes,viewport),viewport,args.abstract_size).save(base['target_abstract'])
                def observe():
                    boxes=browser.tagged_boxes(ids);boxes[root]=[0,0,*viewport];return boxes
                def distance(boxes):
                    return sum(abs(boxes[k][a]-target_boxes[k][a])/viewport[a%2]
                               for k in ids for a in range(4))/max(1,4*len(ids))
                def best_reverse(current_html,boxes,candidates):
                    """Choose the legal path edit with the largest target-distance decrease.

                    Candidate evaluation is an offline data-construction cost.  It
                    never runs during policy inference.  Reflow is observed in the
                    browser, so edits are ranked by their effect on every element,
                    not only by the declaration that was changed.
                    """
                    before=distance(boxes);best=None;best_distance=before
                    for edit in candidates:
                        if not action_is_legal(tree,boxes,viewport,edit):continue
                        i,field,value=edit;nid=tree['nodes'][i]['id']
                        try:
                            browser.load(current_html,viewport)
                            browser.edit_visual_action(current_html,viewport,nid,field,value)
                            observed=observe();score=distance(observed)
                        except ValueError:
                            continue
                        if score<best_distance-1e-9:
                            best_distance=score;best=list(edit)
                    browser.load(current_html,viewport)
                    if best is None:raise ValueError('No reverse-path edit improves the target distance')
                    return best,before,best_distance
                def save_example(name,current_html,boxes,edit,corruption=None):
                    current_tree=refresh_geometry(tree,boxes)
                    screenshot=work/f'{name}.png';browser.page.screenshot(path=str(screenshot),animations='disabled')
                    browser.screenshots+=1
                    abstract=work/f'{name}-abstract.png'
                    labels=elements(tree,boxes,viewport)
                    abstract_image(labels,viewport,args.abstract_size).save(abstract)
                    html_path=work/f'{name}.html';html_path.write_text(current_html)
                    record={**base,'id':source['id']+'/'+name,'current':current_tree,'current_boxes':copy.deepcopy(boxes),
                        'current_image':str(screenshot),'current_abstract':str(abstract),'current_html':str(html_path),
                        'teacher_edits':[] if edit is None else [edit],
                        'corruption_edit':corruption}
                    local_policy.append(record)
                    local_detection.append({**base,'id':record['id'],'image':str(screenshot),'elements':labels})
                browser.load(tagged,viewport);save_example('clean',tagged,target_boxes,None)
                last_corrupted=None
                for trajectory in range(args.trajectories):
                    html=tagged;boxes=target_boxes;used=set();reverse_path=[]
                    for step in range(args.max_noise):
                        accepted=False
                        for attempt in range(30):
                            i=rng.randrange(1,len(tree['nodes']));fields=candidate_fields(tree,i)
                            if not fields:continue
                            field=rng.choice(fields)
                            if (i,field) in used:continue
                            value=rng.choice(ACTION_VALUES[field]);nid=tree['nodes'][i]['id']
                            try:inverse=browser.edit_visual_action(html,viewport,nid,field,value,return_inverse=True)
                            except ValueError:continue
                            candidate_html=browser.page.content();candidate=observe()
                            if distance(candidate)<=distance(boxes)+1e-7:continue
                            # CSS constraints and reflow can make the nominal
                            # inverse invalid for the actually rendered box
                            # (notably a negative width/height edit on a tiny
                            # element). Never persist supervision that the
                            # policy's legal-action mask will reject.
                            teacher=[i,*inverse]
                            if not action_is_legal(tree,candidate,viewport,teacher):continue
                            # Verify the inverse pixel action in the real browser.
                            browser.edit_visual_action(candidate_html,viewport,nid,*inverse);restored=observe()
                            if max(abs(restored[k][a]-boxes[k][a]) for k in ids for a in range(4))>.15:continue
                            reverse_path.append(teacher)
                            best_teacher,before_distance,after_distance=best_reverse(candidate_html,candidate,reverse_path)
                            browser.load(candidate_html,viewport)
                            save_example(f't{trajectory}-s{step}',candidate_html,candidate,best_teacher,[i,field,value])
                            local_policy[-1]['teacher_distance_before']=before_distance
                            local_policy[-1]['teacher_distance_after']=after_distance
                            local_policy[-1]['teacher_candidate_count']=len(reverse_path)
                            html,boxes=candidate_html,candidate;used.add((i,field));accepted=True
                            last_corrupted=local_policy[-1];break
                        if not accepted:break
                if last_corrupted is None:raise ValueError('No verified corruption; page excluded, not silently clean-only')
                page={**base,'screenshot':base['target_image'],'html':str(work/'reference.html'),
                      'initial_html':last_corrupted['current_html'],'evaluation_target_boxes':target_boxes,
                      'total_visible_nodes':dom['total_visible_nodes'],'selected_nodes':len(ids),
                      'source_screenshot':source.get('screenshot'),'source_pixel_mae':source_mae,
                      'construction':'controlled same-DOM corruption; not screenshot-generated initial HTML'}
                saved={'page':page,'policy':local_policy,'detection':local_detection};write_json(cache,saved)
                pages.append(page);policy+=local_policy;detection+=local_detection
            except Exception as error:errors.append({'id':source['id'],'error':str(error)})
            print(f'[{index+1}/{len(sources)}] visual examples={len(policy)}, failures={len(errors)}',flush=True)
    for split in ('train','val','test'):
        write_jsonl(out/f'policy-{split}.jsonl',[r for r in policy if r['split']==split])
        write_jsonl(out/f'detector-{split}.jsonl',[r for r in detection if r['split']==split])
        write_jsonl(out/f'pages-{split}.jsonl',[r for r in pages if r['split']==split])
    coverage={split:{'selected':sum(r['split']==split for r in sources),
                     'usable':sum(r['split']==split for r in pages)} for split in ('train','val','test')}
    for counts in coverage.values():counts['excluded']=counts['selected']-counts['usable']
    write_json(out/'report.json',{'pages':len(pages),'policy_examples':len(policy),'detector_examples':len(detection),
                                'split_coverage':coverage,'errors':errors})
    if not policy:raise ValueError('No usable visual training examples; see report.json')
    return policy

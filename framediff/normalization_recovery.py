"""Keep the pre-normalization render fixed; fit executable recovery labels.

The bounded teacher search is preparation-only, never inference. A fitted CSS
state is a supervision anchor, NOT a replacement ground-truth screenshot.
"""
import copy
import hashlib
import re
from pathlib import Path

from . import css_owners
from .ir import read_json
from .preparation_workers import write_json
from .tree_edits import FIELDS
from .web_experiment import digest

CONTRACT = 'pre-normalization-render-recovery-v1'


def fit_reference(browser, tree, viewport, owners, target_boxes, target_elements, size, steps):
    from .tree_online import abstraction
    from .visual import semantic_masks
    target = semantic_masks(target_elements, viewport, size)
    ids = [n['id'] for n in tree['nodes'][1:] if n.get('visual_class', 0)>0]
    def measure():
        mask, boxes, _ = abstraction(browser, tree, viewport, size)
        geometry = sum(sum(abs(a-b) / viewport[j%2] for j,(a,b) in enumerate(zip(boxes[k],target_boxes[k])))
                       for k in ids) / max(1,4*len(ids))
        visual = float((mask-target).abs().mean())
        return geometry+visual, boxes, visual
    initial, boxes, initial_visual = measure()
    score=initial;trace=[]
    for _ in range(steps):
        if score<=1e-8:break
        state=css_owners.read(browser,tree)['state'];best=None;best_score=score
        for i,owner in enumerate(owners):
            if owner.get('kind')!='inline' or owner.get('id') not in ids:continue
            key=owner['id'];box=boxes[key];goal=target_boxes[key]
            for field_index,field in enumerate(FIELDS):
                value,priority=state[i][field_index]
                if not re.fullmatch(r'-?(?:\d+(?:\.\d*)?|\.\d+)px',value):continue
                old=float(value[:-2])
                axis=2 if field=='width' else 3 if field=='height' else 0 if field in ('margin-left','margin-right') else 1
                change=goal[axis]-box[axis]
                proposals={old+change,old-change}
                if field in ('width','height'):proposals.add(goal[axis])
                for number in sorted(proposals):
                    if abs(number-old)<1e-4 or (field in ('width','height') and number<=0):continue
                    edit=[i,field,f'{number:.4f}px',priority]
                    try:
                        css_owners.execute(browser,owners,edit)
                        candidate,_,_=measure()
                        if candidate<best_score-1e-8:best_score=candidate;best=edit
                    finally:css_owners.execute(browser,owners,[i,field,value,priority])
        if best is None:break
        css_owners.execute(browser,owners,best)
        score,boxes,_=measure();trace.append(best)
    _,_,final_visual=measure()
    return css_owners.read(browser,tree)['state'],dict(
        strategy='bounded-greedy-geometry-and-abstraction-v1',initial_distance=initial,
        final_distance=score,initial_abstraction_mae=initial_visual,
        final_abstraction_mae=final_visual,edits=trace,max_steps=steps,
        exact_geometry_recovery=all(max(abs(a-b) for a,b in zip(boxes[k],target_boxes[k]))<=1 for k in ids),
        note='Finite search; residual errors are reported, not relabeled as ground truth.')


def prepare_page(task):
    from .html_bridge import HtmlBrowser
    from .explicit_html import convert,feedback_metadata
    from .html_feedback import refresh_geometry
    from .visual import annotate
    from .tree_online import target_pool,OnlineSampler
    args,out,split,index,page_id,row,pages=task
    folder=out/'normalized-pages'/split/hashlib.sha256(page_id.encode()).hexdigest()
    folder.mkdir(parents=True,exist_ok=True)
    source=Path(row['current_html'] if row['id'].endswith('/clean') else row['target_html'])
    signature={'html':digest(source),'target':digest(row['target_image']),
               'tree':row['current'],'boxes':row['current_boxes'],'elements':row['target_elements'],
               'contract':CONTRACT,'fit_steps':args.recovery_fit_steps}
    cache=folder/'record.json';sampler=None;stage='recovery_source'
    try:
        if args.resume and cache.exists():
            saved=read_json(cache)
            if saved['signature']!=signature:raise ValueError('Recovery inputs changed; use a new output directory')
            for path,sha in saved['assets'].items():
                if digest(path)!=sha:raise ValueError('Recovery cache asset changed')
            return saved['row'],saved['page'],None
        normalized=folder/'normalized.html';reused=False
        reuse=getattr(args,'reuse_normalized',None)
        if reuse:
            previous=Path(reuse)/'normalized-pages'/split/folder.name
            stamp=previous/'normalization.json'
            if stamp.exists():
                old=read_json(stamp);path=previous/'normalized.html'
                if old['source_sha']==signature['html'] and old['sha']==digest(path):
                    normalized.write_bytes(path.read_bytes());reused=True
        with HtmlBrowser() as browser:
            stage='normalization'
            if not reused:
                qa=convert(browser,source.read_text(),row['viewport'],folder,mode='sizes',
                           max_pixel_mae=args.max_pixel_mae,max_box_error=args.max_box_error,
                           action_ids=[n['id'] for n in row['current']['nodes'][1:] if n.get('visual_class',1)>0],
                           target_tree=row['current'],observation_size=args.observation_size)
                if not qa['accepted']:
                    if qa.get('failure_stage')!='normalization_preservation':
                        raise ValueError(qa.get('error','Unstable original render'))
                    normalized.write_bytes((folder/'rejected.html').read_bytes())
                write_json(folder/'normalization-diagnostics.json',qa)
            browser.load(normalized.read_text(),row['viewport'])
            tree=copy.deepcopy(row['current']);annotate(browser,tree)
            parsed=css_owners.read(browser,tree)
            if len(parsed['owners'])>args.max_css_owners:raise ValueError('Too many CSS owners')
            stage='recovery_teacher'
            goal,fit=fit_reference(browser,tree,row['viewport'],parsed['owners'],row['current_boxes'],
                                   row['target_elements'],args.observation_size,args.recovery_fit_steps)
            (folder/'fitted-teacher.html').write_text(browser.page.content())
            write_json(folder/'recovery-report.json',fit)
            browser.load(normalized.read_text(),row['viewport'])
            boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
            boxes[tree['nodes'][0]['id']]=[0,0,*row['viewport']]
            feedback=feedback_metadata(browser,tree)
        seed={k:row[k] for k in ('group','split','source_sha','viewport','target_image','target_elements')}
        seed.update({k:row[k] for k in ('page_id','view') if k in row})
        seed.update(id=page_id+'/seed',current=refresh_geometry(tree,boxes),current_boxes=boxes,
                    target_html=str(normalized),reference_html=str(source),recovery_contract=CONTRACT,
                    css_owners=parsed['owners'],target_declaration_state=goal,
                    normalization_contract=CONTRACT,recovery_report=fit,**feedback)
        stage='visible_candidate_inspection'
        pool=target_pool([{**seed,'split':'train'}],cache_dir=out/'.online-target-cache',observation_size=args.observation_size)
        stage='recovery_sample'
        sampler=OnlineSampler(pool,'screenshot',seed=args.seed,max_noise=1)
        new=sampler.generate(index)
        initial=folder/'initial.html';png=folder/'initial.png'
        initial.write_text(new.pop('current_html_text'));png.write_bytes(new['current_image'])
        new.update(split=split,current_html=str(initial),current_image=str(png),
                   normalization_stats={'reused_normalization':reused,'normalized_elements':len(parsed['owners'])-1})
        page={**pages[page_id],'html':str(normalized),'target_html':str(normalized),
              'reference_html':str(source),'screenshot':row['target_image'],'initial_html':str(initial),
              'evaluation_target_boxes':row['current_boxes'],'recovery_contract':CONTRACT}
        assets={str(p):digest(p) for p in (source,Path(row['target_image']),normalized,initial,png)}
        write_json(cache,dict(signature=signature,row=new,page=page,assets=assets))
        return new,page,None
    except Exception as exc:
        failure={'id':page_id,'stage':stage,'error':str(exc)}
        write_json(folder/'failure.json',failure)
        return None,None,failure
    finally:
        if sampler is not None:sampler.close()

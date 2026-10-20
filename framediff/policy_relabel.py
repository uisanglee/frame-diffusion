"""Refresh visual supervision on immutable, already normalized CSS examples."""
import copy
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from . import css_owners
from .html_bridge import HtmlBrowser
from .html_feedback import refresh_geometry
from .ir import read_json, read_jsonl
from .preparation_workers import page_map, write_json, write_jsonl
from .visual import VISUAL_GEOMETRY_CONTRACT, annotate, abstract_image
from .web_experiment import digest, guard_run


def refresh_row(row, out, size=384, clean_only=False):
    """No HTML/image replacement or teacher search; reject inconsistent assets."""
    from .explicit_html import feedback_metadata
    from .tree_online import abstraction, changed, target_pool
    from .tree_edits import repair_path
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=copy.deepcopy(row)
    for key in ('predicted_target_abstract','predicted_target_elements','parser_provenance','parser_sha'):
        result.pop(key,None)
    tree=copy.deepcopy(row['current'])
    with HtmlBrowser() as browser:
        observations={}
        for prefix in (('target',) if clean_only else ('target','current')):
            browser.load(Path(row[prefix+'_html']).read_text(),row['viewport'])
            png,_=browser.stable_capture()
            with Image.open(io.BytesIO(png)) as measured, Image.open(row[prefix+'_image']) as saved:
                if measured.size!=saved.size or measured.convert('RGB').tobytes()!=saved.convert('RGB').tobytes():
                    raise ValueError(f'{prefix}: saved screenshot differs from measured HTML')
            annotate(browser,tree,visual_boxes=True)
            mask,boxes,items=abstraction(browser,tree,row['viewport'],size)
            parsed=css_owners.read(browser,tree)
            if parsed['owners']!=row['css_owners']:
                raise ValueError(f'{prefix}: CSS owners changed')
            field='target_declaration_state' if prefix=='target' else 'declaration_state'
            if parsed['state']!=row[field]:raise ValueError(f'{prefix}: CSS state changed')
            observations[prefix]=(mask,boxes,items)
            if prefix=='current' or clean_only:
                result['current']=refresh_geometry(tree,boxes)
                result['current_boxes']=boxes
                result.update(feedback_metadata(browser,result['current']))
        target=observations['target']
        current=observations.get('current',target)
        if not clean_only:
            if not changed(target[0],current[0]):raise ValueError('Existing corruption has no new abstraction effect')
            teacher=row['replacement_edit']
            if teacher not in repair_path(row['declaration_state'],row['target_declaration_state']):
                raise ValueError('Existing teacher does not repair a changed CSS declaration')
            css_owners.execute(browser,row['css_owners'],teacher)
            after,_,_=abstraction(browser,tree,row['viewport'],size)
            if not changed(current[0],after):raise ValueError('Existing teacher has no new abstraction effect')
    result['target_elements']=target[2]
    result['visual_geometry_contract']=VISUAL_GEOMETRY_CONTRACT
    for prefix,observation in observations.items():
        path=out/f'{prefix}-abstract.png'
        abstract_image(observation[2],row['viewport'],size).save(path)
        result[prefix+'_abstract']=str(path)
    result.setdefault('online',{}).update(abstraction_max_difference=float((target[0]-current[0]).abs().max()))
    # Recheck clean-target eligibility under the new observation definition.
    checked=target_pool([{**result,'split':'train'}],cache_dir=out.parent/'.inspection',observation_size=size)[0]
    result['visible_candidates']=checked['visible_candidates']
    result['relabel_audit']={'pixels_equal':True,'teacher_preserved':True,
                            'target_boxes':target[1]}
    return result


def _task(task):
    row,out,size,resume,legacy_cache=task
    try:
        if row.get('lazy_contract'):
            result=copy.deepcopy(row)
            result['visual_geometry_contract']=VISUAL_GEOMETRY_CONTRACT
            result['lazy_settings']['policy_relabel']={'source':row,'size':size,
                'cache_dir':str(legacy_cache),'code':digest(__file__)}
            return result,None
        signatures={row[k]:digest(row[k]) for k in
                    ('target_html','current_html','target_image','current_image')}
        signature={'row':row,'assets':signatures,'size':size,'code':digest(__file__)}
        record=out/'record.json'
        if resume and record.is_file():
            saved=read_json(record)
            if saved['signature']==signature and all(Path(p).is_file() and digest(p)==sha for p,sha in saved['assets'].items()):
                return saved['row'],None
        result=refresh_row(row,out,size)
        write_json(record,{'signature':signature,'row':result,'assets':{
            result[k]:digest(result[k]) for k in ('target_abstract','current_abstract')}})
        return result,None
    except Exception as exc:
        return None,{'id':row['id'].rsplit('/',1)[0],'sample_id':row['id'],
                     'stage':'visual_relabel','error':f'{type(exc).__name__}: {exc}'}


def relabel(args):
    source=Path(args.rendered).resolve();out=Path(args.out).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('Use a separate output directory')
    if args.workers<1 or args.observation_size<1:raise ValueError('workers and observation-size must be positive')
    files=[source/f'{kind}-{split}.jsonl' for split in ('train','val','test') for kind in ('policy','pages')]
    guard_run(out,{'kind':'policy-visual-relabel-v1','sources':{str(p):digest(p) for p in files},
                   'size':args.observation_size,'code':digest(__file__)},args.resume)
    summary={}
    for split in ('train','val','test'):
        rows=list(read_jsonl(source/f'policy-{split}.jsonl'))
        kept=[];errors=[]
        tasks=((r,out/'records'/hashlib.sha256(r['id'].encode()).hexdigest(),args.observation_size,args.resume,
                source/'.online-target-cache/lazy-train-views') for r in rows)
        for index,(row,error) in enumerate(page_map(_task,tasks,args.workers),1):
            if error:errors.append(error)
            else:kept.append(row)
            if index%50==0 or index==len(rows):
                print(f'{split} relabel [{index}/{len(rows)}], kept={len(kept)}, failures={len(errors)}',flush=True)
        by_page={r['id'].rsplit('/',1)[0]:r for r in kept}
        pages=[]
        for page in read_jsonl(source/f'pages-{split}.jsonl'):
            row=by_page.get(page['id'])
            if row is None:continue
            page={**page,'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT}
            if row.get('lazy_contract'):
                page.update(lazy_settings=row['lazy_settings'])
            else:
                page.update(evaluation_target_boxes=row['relabel_audit']['target_boxes'])
            pages.append(page)
        write_jsonl(out/f'policy-{split}.jsonl',kept)
        write_jsonl(out/f'pages-{split}.jsonl',pages)
        write_jsonl(out/f'rejections-{split}.jsonl',errors)
        summary[split]={'kept':len(kept),'failed':len(errors),'lazy':sum(bool(r.get('lazy_contract')) for r in kept)}
        write_json(out/'relabel-report.json',summary)
        if not kept:raise ValueError(f'Empty relabeled split: {split}; inspect rejections-{split}.jsonl')
    return summary

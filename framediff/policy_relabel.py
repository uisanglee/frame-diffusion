"""Refresh visual supervision on immutable, already normalized CSS examples."""
import copy
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from . import css_owners
from .html_bridge import HtmlBrowser
from .html_feedback import refresh_geometry
from .ir import read_json, read_jsonl
from .preparation_workers import page_map, write_json, write_jsonl
from .visual import VISUAL_GEOMETRY_CONTRACT, annotate, abstract_image
from .web_experiment import digest, guard_run

# Semantic cache revision: text-fragment caches must not be reused as owner boxes.
# Bump when supervision/eligibility semantics change, not for logging-only edits.
CACHE_REVISION='policy-dom-owner-box-v1'


def failure_reason(exc):
    message=str(exc)
    for text,reason in (
        ('saved screenshot differs','screenshot_mismatch'),
        ('CSS owners changed','css_owner_mismatch'),
        ('CSS state changed','css_state_mismatch'),
        ('Existing corruption has no new abstraction effect','corruption_no_visual_effect'),
        ('Existing teacher has no new abstraction effect','teacher_no_visual_effect'),
        ('Existing teacher does not repair','teacher_state_mismatch'),
        ('Empty online target pool','no_eligible_candidate'),
        ('Render stabilization timed out','render_unstable'),
        ('Timeout','timeout'),('timed out','timeout'),
    ):
        if text in message:return reason
    if isinstance(exc,FileNotFoundError):return 'missing_asset'
    return 'other'


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
    row,out,size,resume,legacy_cache,*options=task
    clean_only=bool(options and options[0])
    try:
        if row.get('lazy_contract'):
            result=copy.deepcopy(row)
            result['visual_geometry_contract']=VISUAL_GEOMETRY_CONTRACT
            result['lazy_settings']['policy_relabel']={'source':row,'size':size,
                'cache_dir':str(legacy_cache),'code':CACHE_REVISION}
            return result,None
        keys=('target_html','target_image') if clean_only else ('target_html','current_html','target_image','current_image')
        signatures={row[k]:digest(row[k]) for k in keys}
        signature={'row':row,'assets':signatures,'size':size,'code':CACHE_REVISION}
        if clean_only:signature['clean_only']=True
        record=out/'record.json'
        if resume and record.is_file():
            saved=read_json(record)
            if saved['signature']==signature and all(Path(p).is_file() and digest(p)==sha for p,sha in saved['assets'].items()):
                return saved['row'],None
        result=refresh_row(row,out,size,clean_only=True) if clean_only else refresh_row(row,out,size)
        if clean_only:
            # These describe the obsolete corruption, not the verified clean target.
            for key in ('current_html','current_image','current_abstract','current_html_text','replacement_edit'):
                result.pop(key,None)
            result['declaration_state']=copy.deepcopy(result['target_declaration_state'])
            result['symbolic_distance']=0
            result['relabel_audit']['teacher_preserved']=False
        write_json(record,{'signature':signature,'row':result,'assets':{
            result[k]:digest(result[k]) for k in (('target_abstract',) if clean_only else ('target_abstract','current_abstract'))}})
        return result,None
    except Exception as exc:
        return None,{'id':row['id'].rsplit('/',1)[0],'sample_id':row['id'],
                     'stage':'visual_relabel','reason':failure_reason(exc),
                     'error':f'{type(exc).__name__}: {exc}'}


def relabel(args):
    source=Path(args.rendered).resolve();out=Path(args.out).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('Use a separate output directory')
    if args.workers<1 or args.observation_size<1:raise ValueError('workers and observation-size must be positive')
    files=[source/f'{kind}-{split}.jsonl' for split in ('train','val','test') for kind in ('policy','pages')]
    clean_heldout=getattr(args,'clean_heldout',False)
    config={'kind':'policy-visual-relabel-v1','sources':{str(p):digest(p) for p in files},
            'size':args.observation_size,'code':CACHE_REVISION}
    if clean_heldout:config['clean_heldout']=True
    guard_run(out,config,args.resume)
    summary={}
    for split in ('train','val','test'):
        rows=list(read_jsonl(source/f'policy-{split}.jsonl'))
        clean_only=clean_heldout and split!='train'
        if clean_only:
            unique={}
            for row in rows:unique.setdefault(row['id'].rsplit('/',1)[0],row)
            rows=list(unique.values())
        kept=[];errors=[];reasons=Counter()
        tasks=((r,out/'records'/hashlib.sha256(r['id'].encode()).hexdigest(),args.observation_size,args.resume,
                source/'.online-target-cache/lazy-train-views',clean_only) for r in rows)
        def progress(index,status):
            state={'split':split,'processed':index,'total':len(rows),'kept':len(kept),
                   'failures':len(errors),'failure_reasons':dict(reasons),'status':status}
            write_json(out/f'relabel-progress-{split}.json',state)
            return state
        progress(0,'running')
        # This file describes the current pass; retries are not double-counted.
        with (out/f'rejections-{split}.jsonl').open('w') as rejected:
            for index,(row,error) in enumerate(page_map(_task,tasks,args.workers),1):
                if error:
                    errors.append(error);reasons[error['reason']]+=1
                    rejected.write(json.dumps(error,ensure_ascii=False)+'\n');rejected.flush()
                    print(json.dumps({'relabel_failure':{'split':split,**error},
                                      'failure_reasons':dict(reasons)},ensure_ascii=False),flush=True)
                    progress(index,'running')
                else:kept.append(row)
                if index%50==0 or index==len(rows):
                    progress(index,'running')
                    print(f'{split} relabel [{index}/{len(rows)}], kept={len(kept)}, failures={len(errors)}, reasons={dict(reasons)}',flush=True)
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
                if clean_only:page.pop('initial_html',None)
            pages.append(page)
        write_jsonl(out/f'policy-{split}.jsonl',kept)
        write_jsonl(out/f'pages-{split}.jsonl',pages)
        write_jsonl(out/f'rejections-{split}.jsonl',errors)
        summary[split]={'kept':len(kept),'failed':len(errors),'lazy':sum(bool(r.get('lazy_contract')) for r in kept),
                        'failure_reasons':dict(reasons)}
        write_json(out/'relabel-report.json',summary)
        progress(len(rows),'complete' if kept else 'failed')
        if not kept:raise ValueError(f'Empty relabeled split: {split}; inspect rejections-{split}.jsonl')
    return summary

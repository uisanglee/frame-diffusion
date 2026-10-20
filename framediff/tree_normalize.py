"""Normalize clean HTML once, then seed online training and held-out corruption."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import css_owners
from .explicit_html import feedback_metadata
from .safe_sizes import convert, CONTRACT, COMPATIBLE_IMPLEMENTATION_SHAS
from .html_bridge import HtmlBrowser
from .html_feedback import refresh_geometry
from .ir import read_json, read_jsonl
from .tree_online import ONLINE_CONTRACT, SAMPLING_CONTRACT, OnlineSampler, target_pool
from .visual import annotate, elements, VISUAL_GEOMETRY_CONTRACT, DETECTOR_LABEL_CONTRACT
from .visual_data import validate_splits
from .web_experiment import digest, guard_run
from .preparation_workers import page_map, write_json, write_jsonl


def _guard_normalization_run(out,config,resume):
    """Allow only the audited scroll-fast-path migration to reuse records."""
    path=out/'config.json'
    if not (resume and path.exists()):
        guard_run(out,config,resume);return
    previous=read_json(path)
    if previous==config:return
    old_sha=previous.get('safe_sizes_sha');new_sha=config.get('safe_sizes_sha')
    allowed={new_sha,*COMPATIBLE_IMPLEMENTATION_SHAS}
    def semantic(value):
        value=dict(value)
        value.pop('safe_sizes_sha',None)
        value.pop('compatible_safe_sizes_shas',None)
        return value
    if (semantic(previous)==semantic(config) and old_sha in allowed and new_sha in allowed):
        migrated={**config,'compatible_safe_sizes_shas':sorted(
            set(previous.get('compatible_safe_sizes_shas',[]))|{old_sha,new_sha})}
        write_json(path,migrated)
        print({'normalization_cache_migration':'scroll-fast-path-v1',
               'reused_safe_sizes_sha':old_sha,'current_safe_sizes_sha':new_sha},flush=True)
        return
    guard_run(out,config,resume)



def _prepare_page(task, persist_seed=True):
    args, out, split, index, page_id, row, pages = task
    latest=getattr(args,'latest_visual_labels',False)
    expected_contract=VISUAL_GEOMETRY_CONTRACT if latest else row.get('visual_geometry_contract')
    if row.get('lazy_contract'):
        if split!='train':raise ValueError('Lazy held-out normalization is forbidden')
        settings={**row['lazy_settings'],**{k:getattr(args,k) for k in
            ('max_css_owners','observation_size','max_pixel_mae','max_box_error')}}
        if latest:
            settings.update(visual_geometry_contract=VISUAL_GEOMETRY_CONTRACT,
                            detector_label_contract=DETECTOR_LABEL_CONTRACT)
        contract=CONTRACT
        new={**row,'lazy_settings':settings,'normalization_contract':contract,
             'teacher_strategy':css_owners.CONTRACT}
        if latest:new['visual_geometry_contract']=expected_contract
        return new,{**pages[page_id],**new,'id':page_id},None
    folder=out/'normalized-pages'/split/hashlib.sha256(page_id.encode()).hexdigest()
    folder.mkdir(parents=True,exist_ok=True)
    html_path=Path(row['current_html'] if row['id'].endswith('/clean') else row['target_html'])
    cache=folder/'record.json';sampler=None;stage='source';qa=None
    try:
        signature=digest(html_path)
        if args.resume and cache.exists():
            saved=read_json(cache)
            if saved['source_sha']!=signature:raise ValueError('Source HTML changed; use a new output directory')
            if saved['row'].get('visual_geometry_contract')!=expected_contract:
                raise ValueError('Visual labels changed; use a new normalized output directory')
            for key,sha in saved['assets'].items():
                if digest(key)!=sha:raise ValueError('Normalized cache assets changed')
            new=saved['row'];page=saved['page']
        else:
            # Failed preservation checks never create record.json or
            # normalization.json. Resume retries them while keeping
            # successful pages and their assets unchanged.
            stage='normalization'
            normalized=folder/'normalized.html'
            stamp=folder/'normalization.json'
            with HtmlBrowser() as browser:
                if latest:
                    # Measure the cached tagged HTML; preserve DOM owner IDs and
                    # source assets, but never relabel by changing metadata alone.
                    row=copy.deepcopy(row)
                    browser.load(html_path.read_text(),row['viewport'])
                    browser.stable_capture()
                    annotate(browser,row['current'],visual_boxes=True)
                    row['visual_geometry_contract']=expected_contract
                if args.resume and stamp.exists():
                    status=read_json(stamp)
                    if status['source_sha']!=signature or status['sha']!=digest(normalized):
                        raise ValueError('Normalized HTML changed; use a new output directory')
                else:
                    nodes=row['current']['nodes'][1:]
                    action_ids=[n['id'] for n in nodes if n.get('visual_class',1)>0]
                    qa=convert(browser,html_path.read_text(),row['viewport'],folder,mode='sizes',
                               max_pixel_mae=args.max_pixel_mae,max_box_error=args.max_box_error,
                               action_ids=action_ids,target_tree=row['current'],
                               observation_size=args.observation_size)
                    if not qa['accepted']:
                        stage=qa.get('failure_stage') or 'normalization_preservation'
                        raise ValueError(qa.get('error') or 'Normalization changed rendering; see page report.json')
                    write_json(stamp,{'source_sha':signature,'sha':digest(normalized)})
                qa=read_json(folder/'report.json')
                stats={k:qa.get(k) for k in ('normalized_elements','partial_normalization',
                    'verification_trials','scroll_verification_trials','full_scroll_fallback')}
                browser.load(normalized.read_text(),row['viewport'])
                stage='css_owner_parsing'
                tree=copy.deepcopy(row['current']);annotate(browser,tree)
                boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
                boxes[tree['nodes'][0]['id']]=[0,0,*row['viewport']]
                tree=refresh_geometry(tree,boxes)
                parsed=css_owners.read(browser,tree)
                if len(parsed['owners'])>args.max_css_owners:raise ValueError('Too many normalized CSS owners')
                feedback=feedback_metadata(browser,tree)
                target_elements=elements(tree,boxes,row['viewport'],feedback['current_clip_boxes'])
            seed={k:row[k] for k in ('group','split','source_sha','viewport')}
            for key in ('page_id','view'):
                if key in row:seed[key]=row[key]
            seed.update(id=page_id+'/seed',current=tree,current_boxes=boxes,
                visual_geometry_contract=tree.get('visual_geometry_contract'),
                target_html=str(normalized),target_image=str(folder/'after.png'),
                target_elements=target_elements,css_owners=parsed['owners'],
                target_declaration_state=parsed['state'],normalization_contract=CONTRACT,**feedback)
            # target_pool caches visibility probes; train and evaluation
            # use only this verified normalized clean program.
            stage='visible_candidate_inspection'
            write_json(folder/'inspection-input.json',seed)
            inspection=[]
            try:
                pool=target_pool([{**seed,'split':'train'}],cache_dir=out/'.online-target-cache',
                                 observation_size=args.observation_size,diagnostics=inspection)
            finally:
                if any(r['reason']=='no_editable_css' for r in inspection):
                    from .candidate_diagnostics import inspect_saved_target
                    try:
                        detail=inspect_saved_target(seed,folder)
                        for r in inspection:
                            if r['reason']=='no_editable_css':r['eligibility']=detail
                    except Exception as exc:
                        for r in inspection:r['eligibility_error']=f'{type(exc).__name__}: {exc}'
                write_json(folder/'candidate-inspection.json',inspection)
            stage='seed_corruption'
            sampler=OnlineSampler(pool,'screenshot',seed=args.seed,max_noise=1)
            new=sampler.generate(index)
            initial=folder/'initial.html';png=folder/'initial.png'
            if persist_seed:
                initial.write_text(new.pop('current_html_text'));png.write_bytes(new['current_image'])
                new.update(current_html=str(initial),current_image=str(png))
            else:
                new.pop('current_html_text',None);new.pop('current_image',None)
            new.update(split=split,normalization_stats=stats,
                       normalization_contract=CONTRACT)
            page={k:pages[page_id][k] for k in ('id','group','split','source_sha','viewport')}
            for key in ('page_id','view'):
                if key in pages[page_id]:page[key]=pages[page_id][key]
            page.update(html=str(normalized),target_html=str(normalized),
                visual_geometry_contract=tree.get('visual_geometry_contract'),
                screenshot=str(folder/'after.png'),
                evaluation_target_boxes=boxes,policy_subset=css_owners.CONTRACT,
                normalization_contract=CONTRACT,original_html=str(html_path))
            if persist_seed:
                page['initial_html']=str(initial)
                assets={str(p):digest(p) for p in (normalized,folder/'after.png',initial,png)}
                write_json(cache,{'source_sha':signature,'row':new,'page':page,'assets':assets})
        write_json(folder/'page-status.json',{'id':page_id,'status':'kept',
                   'normalization_stats':new.get('normalization_stats')})
    except Exception as exc:
        failure={'id':page_id,'stage':stage,'error':str(exc)}
        if stage=='visible_candidate_inspection' and hasattr(exc,'diagnostics'):
            failure['inspection_reasons']=[r['reason'] for r in exc.diagnostics]
            failure['inspection_report']=str(folder/'candidate-inspection.json')
        if qa is not None and not qa.get('accepted',True):
            failure['error_type']=qa.get('error_type')
            failure['diagnostics']={'pixel_mae':qa.get('pixel_mae'),
                'max_box_error_px':qa.get('max_box_error_px'),
                'abstraction_mae':qa.get('abstraction_mae'),
                'non_target_geometry_changes':len(qa.get('non_target_geometry_changes',[])),
                'scroll_preservation':qa.get('scroll_preservation'),
                'first_element_failures':qa.get('element_failures',[])[:3]}
        write_json(folder/'page-status.json',{'status':'failed',**failure})
        return None, None, failure
    finally:
        if sampler is not None:sampler.close()
    return new, page, None


def prepare(args):
    workers=getattr(args,'workers',1)
    if workers<1:raise ValueError('workers must be positive')
    source=Path(args.rendered).resolve();out=Path(args.out).resolve()
    upstream_report=read_json(source/'report.json') if (source/'report.json').is_file() else {}
    limit=getattr(args,'limit_pages',0)
    if limit<0:raise ValueError('limit-pages must be nonnegative')
    if source==out or source in out.parents:
        raise ValueError('Use a separate normalized output directory')
    files=[source/f'{kind}-{split}.jsonl' for split in ('train','val','test') for kind in ('policy','pages')]
    config=dict(kind=css_owners.CONTRACT,normalization=CONTRACT,
        normalizer_sha=digest(Path(__file__).with_name('explicit_html.py')),
        safe_sizes_sha=digest(Path(__file__).with_name('safe_sizes.py')),
        scroll_guard_sha=digest(Path(__file__).with_name('box_scroll_guard.py')),
        sources={str(p):digest(p) for p in files},seed=args.seed,limit_pages=limit,
        max_css_owners=args.max_css_owners,observation_size=args.observation_size,
        max_pixel_mae=args.max_pixel_mae,max_box_error=args.max_box_error,
        corruption=ONLINE_CONTRACT,sampling=SAMPLING_CONTRACT)
    if getattr(args,'latest_visual_labels',False):
        config['visual_geometry_contract']=VISUAL_GEOMETRY_CONTRACT
        config['visual_labels_sha']=digest(Path(__file__).with_name('visual.py'))
    _guard_normalization_run(out,config,args.resume)
    report={};all_rows=[]
    # Per-page normalization closes its browser before the corruption sampler
    # starts another synchronous Playwright runtime.
    def prepare_splits():
        for split in ('train','val','test'):
            pages={r['id']:r for r in read_jsonl(source/f'pages-{split}.jsonl')}
            clean={}
            for row in read_jsonl(source/f'policy-{split}.jsonl'):
                page_id=row['id'].rsplit('/',1)[0]
                if row['id'].endswith('/clean') or (page_id not in clean and row.get('target_html')):
                    clean[page_id]=row
            validate_splits(list(clean.values()))
            source_views=len(pages)
            source_page_ids={r.get('page_id',r['id']) for r in pages.values()}
            source_pages=len(source_page_ids)
            selected_pages=upstream_report.get('split_coverage',{}).get(split,{}).get('selected',source_pages)
            if limit:
                clean=dict(list(clean.items())[:limit])
                pages={k:v for k,v in pages.items() if k in clean}
            kept=[];selected=[]
            errors=[{'id':p,'error':'Missing clean HTML record'} for p in pages if p not in clean]
            rejection_path=out/f'rejections-{split}.jsonl'
            write_jsonl(rejection_path,errors)
            reasons=Counter(e['error'] for e in errors)
            tasks=((args,out,split,index,page_id,row,{page_id:pages.get(page_id,{})})
                   for index,(page_id,row) in enumerate(clean.items()))
            for index,(new,page,failure) in enumerate(page_map(_prepare_page,tasks,workers)):
                if failure:
                    errors.append(failure)
                    reasons[failure['stage']+': '+failure['error'].splitlines()[0]]+=1
                    with rejection_path.open('a') as log:
                        log.write(json.dumps(failure,ensure_ascii=False)+'\n')
                    print({'normalization_failure':failure},flush=True)
                else:
                    kept.append(new);selected.append(page)
                if index%50==0 or index==len(clean)-1:
                    print(f'{split} normalized [{index+1}/{len(clean)}], kept={len(kept)}, failures={len(errors)}',flush=True)
                    print({'normalization_top_errors':reasons.most_common(5)},flush=True)
                write_json(out/f'progress-{split}.json',{'processed':index+1,'total':len(clean),
                    'kept':len(kept),'failures':len(errors),'top_errors':reasons.most_common(10)})
            write_jsonl(out/f'policy-{split}.jsonl',kept)
            write_jsonl(out/f'pages-{split}.jsonl',selected)
            write_jsonl(out/f'rejections-{split}.jsonl',errors)
            clean_pages=len({r.get('page_id',r['id'].rsplit('/',1)[0]) for r in clean.values()})
            kept_pages=len({r.get('page_id',r['id'].rsplit('/',1)[0]) for r in kept})
            report[split]={'selected_pages':selected_pages,'source_pages':source_pages,'source_views':source_views,
                'clean_pages':clean_pages,'clean_views':len(clean),'kept':kept_pages,'kept_views':len(kept),
                'rejected':clean_pages-kept_pages,'rejected_views':len(errors),'top_errors':reasons.most_common(10),
                'kept_with_frozen_sizes':sum((r.get('normalization_stats',{}).get('normalized_elements') or 0)>0 for r in kept),
                'kept_without_frozen_sizes':sum((r.get('normalization_stats',{}).get('normalized_elements') or 0)==0 for r in kept)}
            if any(r.get('lazy_contract') for r in kept):
                report[split].update(status='deferred',pending_pages=kept_pages,pending_views=len(kept),
                    verified_pages=0,kept_with_frozen_sizes=0,kept_without_frozen_sizes=0)
            all_rows.extend(kept)
    prepare_splits()
    validate_splits(all_rows)
    write_json(out/'prepare-report.json',report)
    if any(not report[s]['kept'] for s in report):raise ValueError('Empty normalized split; inspect rejections and page reports')

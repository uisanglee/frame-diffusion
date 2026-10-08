"""Normalize clean HTML once, then seed online training and held-out corruption."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import css_owners
from .explicit_html import convert, feedback_metadata
from .html_bridge import HtmlBrowser
from .html_feedback import refresh_geometry
from .ir import read_json, read_jsonl, write_json, write_jsonl
from .tree_online import ONLINE_CONTRACT, SAMPLING_CONTRACT, OnlineSampler, target_pool
from .visual import annotate, elements
from .visual_data import validate_splits
from .web_experiment import digest, guard_run

CONTRACT = 'parent-preserved-targeted-inline-v4'


def prepare(args):
    source=Path(args.rendered).resolve();out=Path(args.out).resolve()
    limit=getattr(args,'limit_pages',0)
    if limit<0:raise ValueError('limit-pages must be nonnegative')
    if source==out or source in out.parents:
        raise ValueError('Use a separate normalized output directory')
    files=[source/f'{kind}-{split}.jsonl' for split in ('train','val','test') for kind in ('policy','pages')]
    guard_run(out,dict(kind=css_owners.CONTRACT,normalization=CONTRACT,
        normalizer_sha=digest(Path(__file__).with_name('explicit_html.py')),
        sources={str(p):digest(p) for p in files},seed=args.seed,limit_pages=limit,
        max_css_owners=args.max_css_owners,observation_size=args.observation_size,
        max_pixel_mae=args.max_pixel_mae,max_box_error=args.max_box_error,
        corruption=ONLINE_CONTRACT,sampling=SAMPLING_CONTRACT),args.resume)
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
            source_pages=len(pages)
            if limit:
                clean=dict(list(clean.items())[:limit])
                pages={k:v for k,v in pages.items() if k in clean}
            kept=[];selected=[]
            errors=[{'id':p,'error':'Missing clean HTML record'} for p in pages if p not in clean]
            rejection_path=out/f'rejections-{split}.jsonl'
            write_jsonl(rejection_path,errors)
            reasons=Counter(e['error'] for e in errors)
            for index,(page_id,row) in enumerate(clean.items()):
                folder=out/'normalized-pages'/split/hashlib.sha256(page_id.encode()).hexdigest()
                folder.mkdir(parents=True,exist_ok=True)
                html_path=Path(row['current_html'] if row['id'].endswith('/clean') else row['target_html'])
                cache=folder/'record.json';sampler=None;stage='source';qa=None
                try:
                    signature=digest(html_path)
                    if args.resume and cache.exists():
                        saved=read_json(cache)
                        if saved['source_sha']!=signature:raise ValueError('Source HTML changed; use a new output directory')
                        for key,sha in saved['assets'].items():
                            if digest(key)!=sha:raise ValueError('Normalized cache assets changed')
                        new=saved['row'];page=saved['page']
                    else:
                        stage='normalization'
                        normalized=folder/'normalized.html'
                        stamp=folder/'normalization.json'
                        with HtmlBrowser() as browser:
                            if args.resume and stamp.exists():
                                status=read_json(stamp)
                                if status['source_sha']!=signature or status['sha']!=digest(normalized):
                                    raise ValueError('Normalized HTML changed; use a new output directory')
                            else:
                                nodes=row['current']['nodes'][1:]
                                action_ids=[n['id'] for n in nodes if n.get('visual_class',1)>0]
                                qa=convert(browser,html_path.read_text(),row['viewport'],folder,mode='sizes',
                                           max_pixel_mae=args.max_pixel_mae,max_box_error=args.max_box_error,
                                           action_ids=action_ids)
                                if not qa['accepted']:raise ValueError('Normalization changed rendering; see page report.json')
                                write_json(stamp,{'source_sha':signature,'sha':digest(normalized)})
                            qa=read_json(folder/'report.json')
                            stats={k:qa.get(k) for k in ('normalized_elements','partial_normalization','verification_trials')}
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
                        seed.update(id=page_id+'/seed',current=tree,current_boxes=boxes,
                            target_html=str(normalized),target_image=str(folder/'after.png'),
                            target_elements=target_elements,css_owners=parsed['owners'],
                            target_declaration_state=parsed['state'],normalization_contract=CONTRACT,**feedback)
                        # target_pool caches visibility probes; train and evaluation
                        # use only this verified normalized clean program.
                        stage='visible_candidate_inspection'
                        pool=target_pool([{**seed,'split':'train'}],cache_dir=out/'.online-target-cache',
                                         observation_size=args.observation_size)
                        stage='seed_corruption'
                        sampler=OnlineSampler(pool,'screenshot',seed=args.seed,max_noise=1)
                        new=sampler.generate(index)
                        initial=folder/'initial.html';png=folder/'initial.png'
                        initial.write_text(new.pop('current_html_text'));png.write_bytes(new['current_image'])
                        new.update(split=split,current_html=str(initial),current_image=str(png),normalization_stats=stats,
                                   normalization_contract=CONTRACT)
                        page={k:pages[page_id][k] for k in ('id','group','split','source_sha','viewport')}
                        page.update(html=str(normalized),target_html=str(normalized),
                            screenshot=str(folder/'after.png'),initial_html=str(initial),
                            evaluation_target_boxes=boxes,policy_subset=css_owners.CONTRACT,
                            normalization_contract=CONTRACT,original_html=str(html_path))
                        assets={str(p):digest(p) for p in (normalized,folder/'after.png',initial,png)}
                        write_json(cache,{'source_sha':signature,'row':new,'page':page,'assets':assets})
                    kept.append(new);selected.append(page)
                    write_json(folder/'page-status.json',{'id':page_id,'status':'kept',
                               'normalization_stats':new.get('normalization_stats')})
                except Exception as exc:
                    failure={'id':page_id,'stage':stage,'error':str(exc)}
                    if stage=='normalization' and qa is not None:
                        failure['diagnostics']={'pixel_mae':qa.get('pixel_mae'),
                            'max_box_error_px':qa.get('max_box_error_px'),
                            'first_element_failures':qa.get('element_failures',[])[:3]}
                    errors.append(failure)
                    reasons[stage+': '+str(exc).splitlines()[0]]+=1
                    with rejection_path.open('a') as log:
                        log.write(json.dumps(failure,ensure_ascii=False)+'\n')
                    write_json(folder/'page-status.json',{'status':'failed',**failure})
                    print({'normalization_failure':failure},flush=True)
                finally:
                    if sampler is not None:sampler.close()
                if index%50==0 or index==len(clean)-1:
                    print(f'{split} normalized [{index+1}/{len(clean)}], kept={len(kept)}, failures={len(errors)}',flush=True)
                    print({'normalization_top_errors':reasons.most_common(5)},flush=True)
                write_json(out/f'progress-{split}.json',{'processed':index+1,'total':len(clean),
                    'kept':len(kept),'failures':len(errors),'top_errors':reasons.most_common(10)})
            write_jsonl(out/f'policy-{split}.jsonl',kept)
            write_jsonl(out/f'pages-{split}.jsonl',selected)
            write_jsonl(out/f'rejections-{split}.jsonl',errors)
            report[split]={'source_pages':source_pages,'clean_pages':len(clean),'kept':len(kept),
                'rejected':len(errors),'top_errors':reasons.most_common(10),
                'kept_with_frozen_sizes':sum((r.get('normalization_stats',{}).get('normalized_elements') or 0)>0 for r in kept),
                'kept_without_frozen_sizes':sum((r.get('normalization_stats',{}).get('normalized_elements') or 0)==0 for r in kept)}
            all_rows.extend(kept)
    prepare_splits()
    validate_splits(all_rows)
    write_json(out/'prepare-report.json',report)
    if any(not report[s]['kept'] for s in report):raise ValueError('Empty normalized split; inspect rejections and page reports')

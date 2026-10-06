"""A read-only view of cached numeric corruption prefixes. No browser needed."""
from pathlib import Path
import copy
import hashlib
import re
from collections import Counter
from .ir import read_json,read_jsonl,write_jsonl,write_json
from .visual import NUMERIC_FIELDS,elements,action_is_legal,CONTRACT,ACTION_CONTRACT
from .html_bridge import HtmlBrowser
from .web_experiment import digest, guard_run


def numeric_rows(rows):
    indexed={r['id']:r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('Duplicate policy IDs')
    kept=[]
    for row in rows:
        prefix, name=row['id'].rsplit('/',1)
        match=re.fullmatch(r't(\d+)-s(\d+)',name)
        if name=='clean':
            if row['teacher_edits']:raise ValueError('Clean row has edits')
        elif match:
            trajectory,step=map(int,match.groups())
            history=[indexed.get(f'{prefix}/t{trajectory}-s{k}') for k in range(step+1)]
            if any(r is None or not r['teacher_edits'] or
                   (r.get('corruption_edit') or r['teacher_edits'][0])[1] not in NUMERIC_FIELDS or
                   any(e[1] not in NUMERIC_FIELDS for e in r['teacher_edits']) for r in history):continue
        else:raise ValueError(f'Unrecognized trajectory ID: {row["id"]}')
        kept.append({**row,'policy_subset':'numeric-prefix-v1'})
    return kept


def prepare(args):
    source=Path(args.rendered).resolve();out=Path(args.out).resolve()
    if out==source or source in out.parents:raise ValueError('Use a separate output directory')
    paths=[source/f'{kind}-{split}.jsonl' for split in ('train','val','test') for kind in ('policy','pages')]
    guard_run(out,{'kind':'numeric-prefix-v1','sources':{str(p):digest(p) for p in paths}},True)
    report={}
    for split in ('train','val','test'):
        rows=list(read_jsonl(source/f'policy-{split}.jsonl'));kept=numeric_rows(rows)
        clean={r['target_image']:r for r in rows if r['id'].endswith('/clean')}
        for row in kept:
            if 'target_elements' not in row:
                target=clean.get(row['target_image'])
                if target is None:raise ValueError('Missing clean cached geometry for target masks')
                row['target_elements']=elements(target['current'],target['current_boxes'],target['viewport'])
        if not kept or not any(r['teacher_edits'] for r in kept):raise ValueError(f'No numeric edits in {split}')
        by_page={}
        for r in kept:
            if r['teacher_edits']:by_page[r['id'].rsplit('/',1)[0]]=r
        pages=[]
        for page in read_jsonl(source/f'pages-{split}.jsonl'):
            if page['id'] in by_page:
                pages.append({**page,'initial_html':by_page[page['id']]['current_html'],
                              'policy_subset':'numeric-prefix-v1'})
        if not pages:raise ValueError(f'No numeric evaluation pages in {split}')
        write_jsonl(out/f'policy-{split}.jsonl',kept)
        write_jsonl(out/f'pages-{split}.jsonl',pages)
        report[split]={'source_rows':len(rows),'kept_rows':len(kept),'pages':len(pages),
                       'properties':dict(Counter(e[1] for r in kept for e in r['teacher_edits']))}
    write_json(out/'subset-report.json',report);print(report,flush=True)


def _page_prefix(row):return row['id'].rsplit('/',1)[0]


def relabel_best_reverse(args):
    """Build improvement distributions using cached HTML and real browser reflow.

    Existing screenshots, abstractions, DOM trees and HTML states are referenced
    in place. Only candidate CSS edits are executed; no assets or corruptions are
    regenerated and no detector is involved.
    """
    source=Path(args.rendered).resolve();out=Path(args.out).resolve()
    if out==source or source in out.parents:raise ValueError('Use a separate relabeled output directory')
    paths=[source/f'{kind}-{split}.jsonl' for split in ('train','val','test')
           for kind in ('policy','pages','detector')]
    config={'kind':'visual-relabel-improvement-distribution-v3','contract':CONTRACT,'action_contract':ACTION_CONTRACT,
            'sources':{str(path):digest(path) for path in paths}}
    guard_run(out,config,args.resume)
    totals={'pages':0,'input_rows':0,'kept_rows':0,'rejected_rows':0,'failed_pages':0}
    with HtmlBrowser() as browser:
        for split in ('train','val','test'):
            rows=list(read_jsonl(source/f'policy-{split}.jsonl'));totals['input_rows']+=len(rows)
            grouped={}
            for row in rows:grouped.setdefault(_page_prefix(row),[]).append(row)
            relabeled=[];successful_pages=set();failures=[]
            cache_root=out/'pages'/split;cache_root.mkdir(parents=True,exist_ok=True)
            for page_index,(page_id,page_rows) in enumerate(grouped.items(),1):
                cache=cache_root/f'{hashlib.sha256(page_id.encode()).hexdigest()[:20]}.json'
                if args.resume and cache.exists():
                    saved=read_json(cache);relabeled+=saved['rows']
                    if saved['usable']:successful_pages.add(page_id)
                    failures+=saved['failures'];continue
                local=[];page_failures=[]
                try:
                    browser.reset_context()
                    clean=[row for row in page_rows if row['id'].endswith('/clean')]
                    if len(clean)!=1:raise ValueError(f'Expected one clean row, found {len(clean)}')
                    clean=clean[0];tree=clean['current'];target_boxes=clean['current_boxes'];viewport=clean['viewport']
                    root=tree['nodes'][0]['id'];ids=[node['id'] for node in tree['nodes'][1:]]
                    target_items=clean.get('target_elements') or elements(tree,target_boxes,viewport)
                    clean_copy={**clean,'teacher_strategy':'improvement-distribution-v1',
                                'target_elements':target_items,'improving_edits':[]}
                    local.append(clean_copy)
                    trajectories={}
                    for row in page_rows:
                        match=re.fullmatch(r't(\d+)-s(\d+)',row['id'].rsplit('/',1)[1])
                        if match:trajectories.setdefault(int(match.group(1)),[]).append((int(match.group(2)),row))
                    def distance(boxes):
                        return sum(abs(boxes[key][axis]-target_boxes[key][axis])/viewport[axis%2]
                                   for key in ids for axis in range(4))/max(1,4*len(ids))
                    for trajectory,states in sorted(trajectories.items()):
                        candidates=[];broken=False
                        for step,row in sorted(states):
                            if broken:continue
                            corruption=row.get('corruption_edit')
                            if corruption and corruption[1] in NUMERIC_FIELDS and isinstance(corruption[2],(int,float)):
                                candidate=[corruption[0],corruption[1],-corruption[2]]
                            elif len(row.get('teacher_edits',[]))==1:
                                candidate=copy.deepcopy(row['teacher_edits'][0])
                            else:
                                page_failures.append({'id':row['id'],'error':'Cannot recover reverse candidate'})
                                broken=True;continue
                            if candidate not in candidates:candidates.append(candidate)
                            before=distance(row['current_boxes']);improving=[]
                            for edit in candidates:
                                if not action_is_legal(row['current'],row['current_boxes'],viewport,edit):continue
                                node,field,value=edit;node_id=row['current']['nodes'][node]['id']
                                try:
                                    browser.edit_visual_action(Path(row['current_html']).read_text(),viewport,
                                                               node_id,field,value)
                                    # The LayoutIR root is the synthetic __viewport__
                                    # node and has no DOM element/data-fd-id.
                                    observed=browser.tagged_boxes(ids);observed[root]=[0,0,*viewport]
                                    score=distance(observed)
                                except Exception:
                                    continue
                                gain=before-score
                                if gain>1e-9:
                                    improving.append({'edit':copy.deepcopy(edit),'distance_after':score,'gain':gain})
                            if not improving:
                                page_failures.append({'id':row['id'],'error':'No legal candidate improves target distance'})
                                broken=True;continue
                            improving.sort(key=lambda item:(-item['gain'],item['edit']))
                            best=improving[0]
                            local.append({**row,'teacher_edits':[best['edit']],
                                'teacher_strategy':'improvement-distribution-v1','target_elements':target_items,
                                'improving_edits':improving,
                                'teacher_distance_before':before,'teacher_distance_after':best['distance_after'],
                                'teacher_candidate_count':len(candidates)})
                    usable=any(row['teacher_edits'] for row in local)
                    if usable:successful_pages.add(page_id)
                except Exception as error:
                    local=[];usable=False;page_failures.append({'id':page_id,'error':str(error)})
                relabeled+=local;failures+=page_failures
                write_json(cache,{'rows':local,'usable':usable,'failures':page_failures})
                if page_index%25==0 or page_index==len(grouped):
                    print(f'[{split} {page_index}/{len(grouped)}] relabeled={len(relabeled)} failures={len(failures)}',flush=True)
            pages=[page for page in read_jsonl(source/f'pages-{split}.jsonl') if page['id'] in successful_pages]
            write_jsonl(out/f'policy-{split}.jsonl',relabeled)
            write_jsonl(out/f'pages-{split}.jsonl',pages)
            write_jsonl(out/f'detector-{split}.jsonl',read_jsonl(source/f'detector-{split}.jsonl'))
            write_json(out/f'relabel-{split}.json',{'pages':len(successful_pages),'rows':len(relabeled),
                                                    'failures':failures})
            totals['pages']+=len(successful_pages);totals['kept_rows']+=len(relabeled)
            totals['rejected_rows']+=len(rows)-len(relabeled);totals['failed_pages']+=len(grouped)-len(successful_pages)
    write_json(out/'relabel-report.json',totals);print(totals,flush=True)
    return totals

"""Nested WebUI sample-size experiments with one domain-disjoint split."""
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from .ir import read_json, read_jsonl
from .preparation_workers import write_json, write_jsonl
from .visual_data import validate_splits
from .web_experiment import digest, guard_run
from .webui_visual import import_webui

SPLITS = ('train', 'val', 'test')


def counts(size, overrides=None):
    if overrides and str(size) in overrides:return overrides[str(size)]
    if size < 10:
        raise ValueError('Sizes must be at least 10')
    return dict(train=size-2*(size//10), val=size//10, test=size//10)


def nested_sources(rows, sizes, seed, overrides=None):
    validate_splits(rows)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source page-view IDs')
    pages={}
    for row in rows:
        page_id=row.get('page_id',row['id'])
        page=pages.setdefault(page_id,{'id':page_id,'split':row['split'],'group':row['group'],'rows':[]})
        if (page['split'],page['group'])!=(row['split'],row['group']):
            raise ValueError('A page or its views cross split/domain assignments')
        page['rows'].append(row)
    # Identical HTML is expected between views of one responsive page, but not
    # between independently counted pages.
    owners={}
    for page in pages.values():
        for sha in {r.get('source_sha') for r in page['rows'] if r.get('source_sha')}:
            if sha in owners and owners[sha]!=page['id']:raise ValueError('Duplicate source HTML across pages')
            owners[sha]=page['id']
    pools={s:sorted((p for p in pages.values() if p['split']==s),
        key=lambda p:hashlib.sha256(f"{seed}:{p['id']}".encode()).hexdigest()) for s in SPLITS}
    result={}
    for size in sorted(set(sizes)):
        requested=counts(size,overrides)
        if any(len(pools[s])<requested[s] for s in SPLITS):
            raise ValueError(f'Insufficient pages for {size}: requested={requested}')
        result[size]=[r for s in SPLITS for page in pools[s][:requested[s]] for r in page['rows']]
    return result


def prepare(args):
    sizes=sorted(set(args.sizes))
    for size in sizes:counts(size)
    out=Path(args.out).resolve()
    explicit=getattr(args,'master_counts',None)
    reference=getattr(args,'allocation_counts',None)
    overrides={}
    if explicit:
        if min(explicit)<1 or sum(explicit)!=sizes[-1]:
            raise ValueError('Master counts must be positive and sum to the largest size')
        overrides[str(sizes[-1])]=dict(zip(SPLITS,explicit))
    if reference and (not explicit or min(reference)<1):
        raise ValueError('Allocation counts require explicit master counts and positive values')
    largest=counts(sizes[-1],overrides)
    requested_views=getattr(args,'views',None);legacy_view=getattr(args,'view',None)
    views=requested_views or ([legacy_view] if legacy_view else ['default_1280-720'])
    config=dict(kind='webui-scaling-811-v2-multiview',root=str(Path(args.root).resolve()),
        sizes=sizes,seed=args.seed,views=views)
    if overrides or reference:config.update(split_counts=overrides,allocation_counts=reference)
    guard_run(out,config,args.resume)
    master=out/'master/source'
    import_webui(SimpleNamespace(root=args.root,out=str(master),views=views,seed=args.seed,
        train_count=largest['train'],val_count=largest['val'],test_count=largest['test'],resume=args.resume,
        allocation_counts=dict(zip(SPLITS,reference)) if reference else None))
    selected=nested_sources(list(read_jsonl(master/'manifest.jsonl')),sizes,args.seed,overrides)
    for size,rows in selected.items():
        folder=out/str(size)/'source'
        write_jsonl(folder/'manifest.jsonl',rows)
        for split in SPLITS:
            write_jsonl(folder/f'manifest-{split}.jsonl',(r for r in rows if r['split']==split))
        write_json(folder/'report.json',{'selected':counts(size,overrides),
            'selected_views':{s:sum(r['split']==s for r in rows) for s in SPLITS},
            'view_distribution':{s:dict(Counter(r.get('view','legacy') for r in rows if r['split']==s)) for s in SPLITS},
            'domains':{s:len({r['group'] for r in rows if r['split']==s}) for s in SPLITS}})
    write_json(out/'split-report.json',{'sizes':{str(n):counts(n,overrides) for n in sizes},
        'common_test_size':sizes[0]//10,'common_test_scale':sizes[0],
        'protocol':'Custom 8:1:1 split by hostname, nested within a single master split; not official WebUI splits',
        'views':views,'seed':args.seed,'train_sampling':'uniform-page-then-uniform-available-view-v1',
        'note':'Sizes count selected unique pages before rendering/normalization exclusions; page views inherit one split.'})


def select_rows(path, hashes):
    return (r for r in read_jsonl(path) if r.get('source_sha') in hashes)


def subset(args):
    """Share master assets; write only scale-specific manifests and coverage."""
    root=Path(args.data).resolve();config=read_json(root/'config.json')
    master=root/'master'
    inputs={str(p):digest(p) for name in ('rendered','labels','css')
            for p in (master/name).glob('*.jsonl')}
    if not inputs:raise ValueError('Prepare master rendered/labels/css data first')
    for size in config['sizes']:
        base=root/str(size)
        rows=list(read_jsonl(base/'source/manifest.jsonl'))
        guard_run(base/'subset',{'kind':'webui-scaling-subset-v1','inputs':inputs,
            'selection':digest(base/'source/manifest.jsonl')},True)
        render_report={};normalize_report={};freeze_report={}
        for split in SPLITS:
            chosen=[r for r in rows if r['split']==split]
            chosen_pages={r.get('page_id',r['id']) for r in chosen}
            hashes={r['source_sha'] for r in chosen};ids={r['id'] for r in chosen}
            for name,kinds in [('rendered',('policy','pages','detector')),
                               ('labels',('policy','pages')),('css',('policy','pages'))]:
                for kind in kinds:
                    write_jsonl(base/name/f'{kind}-{split}.jsonl',
                                select_rows(master/name/f'{kind}-{split}.jsonl',hashes))
                if name!='rendered':
                    rejection=master/name/f'rejections-{split}.jsonl'
                    write_jsonl(base/name/f'rejections-{split}.jsonl',
                        (r for r in read_jsonl(rejection) if r['id'] in ids) if rejection.exists() else [])
            rendered=list(read_jsonl(base/'rendered'/f'pages-{split}.jsonl'))
            normalized=list(read_jsonl(base/'labels'/f'pages-{split}.jsonl'))
            frozen=list(read_jsonl(base/'css'/f'pages-{split}.jsonl'))
            errors=list(read_jsonl(base/'labels'/f'rejections-{split}.jsonl'))
            rendered_pages={r.get('page_id',r['id']) for r in rendered}
            normalized_pages={r.get('page_id',r['id']) for r in normalized}
            frozen_pages={r.get('page_id',r['id']) for r in frozen}
            render_report[split]={'selected':len(chosen_pages),'selected_views':len(chosen),
                'usable':len(rendered_pages),'usable_views':len(rendered),
                'excluded':len(chosen_pages-rendered_pages)}
            normalize_report[split]={'selected_pages':len(chosen_pages),'selected_views':len(chosen),
                'source_pages':len(rendered_pages),'source_views':len(rendered),
                'clean_pages':len(rendered_pages),'clean_views':len(rendered),
                'kept':len(normalized_pages),'kept_views':len(normalized),
                'rejected':len(rendered_pages-normalized_pages),'rejected_views':len(errors),
                'top_errors':Counter(r.get('stage','')+': '+r['error'] for r in errors).most_common(10)}
            freeze_report[split]={'pages':len(frozen_pages),'page_views':len(frozen),
                'failed_pages':len(normalized_pages-frozen_pages),
                'failed_page_views':len(normalized)-len(frozen),
                'rows':sum(1 for _ in read_jsonl(base/'css'/f'policy-{split}.jsonl'))}
            if split=='train' and any(r.get('lazy_contract') for r in rendered):
                render_report[split].update(status='deferred',pending_pages=len(rendered_pages),
                    pending_views=len(rendered),usable=None,usable_views=None,excluded=None)
                normalize_report[split].update(status='deferred',pending_pages=len(normalized_pages),
                    pending_views=len(normalized),kept=None,kept_views=None,rejected=None,rejected_views=None)
                freeze_report[split].update(status='online-only',pages=None,page_views=None)
        write_json(base/'rendered/report.json',{'split_coverage':render_report})
        write_json(base/'labels/prepare-report.json',normalize_report)
        write_json(base/'css/freeze-report.json',freeze_report)
        # The fixed-test evaluator follows this pointer to report all attrition.
        write_json(base/'css/config.json',{'kind':'webui-scaling-css-view-v1',
            'data':{str(base/'labels'/f'policy-{s}.jsonl'):digest(base/'labels'/f'policy-{s}.jsonl') for s in SPLITS}})
        write_json(base/'coverage.json',{'selected':counts(size,config.get('split_counts')),'rendering':render_report,
            'normalization':normalize_report,'fixed_corruption':freeze_report,
            'assets_root':str(master),'warning':'Keep master assets; subset manifests reference them.'})


def summarize(args):
    root=Path(args.data).resolve();runs=Path(args.runs).resolve()
    config=read_json(root/'config.json');reports=[]
    for size in config['sizes']:
        for population in ('own','common'):
            path=runs/str(size)/f'detector-test-{population}.json'
            if not path.exists():continue
            report=read_json(path)
            p=report['precision_iou50'];r=report['recall_iou50']
            reports.append(dict(size=size,test=population,**{k:report.get(k) for k in
                ('n','unique_pages','precision_iou50','recall_iou50','small_recall_iou50','small_n','threshold','by_view','macro_page')},
                f1_iou50=2*p*r/(p+r) if r is not None and p+r else 0.,
                coverage=read_json(root/str(size)/'coverage.json')))
    write_json(runs/'detector-comparison.json',reports)
    lines=['# WebUI detector data scaling','',
           'Class-aware IoU50 precision/recall/F1, not COCO AP. Compare common-test rows for matched evaluation.',
           '', '| Selected pages | Test | Images | Precision | Recall | F1 | Small recall |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for r in reports:
        vals=[r[k] for k in ('precision_iou50','recall_iou50','f1_iou50','small_recall_iou50')]
        lines.append(f"| {r['size']} | {r['test']} | {r['n']} | "+' | '.join('N/A' if v is None else f'{v:.4f}' for v in vals)+' |')
    lines += ['', '## Per-viewport detector results','',
        '| Selected pages | Test | View | Images | Precision | Recall | Small recall |',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in reports:
        for view,metrics in (r.get('by_view') or {}).items():
            vals=[metrics.get(k) for k in ('precision_iou50','recall_iou50','small_recall_iou50')]
            lines.append(f"| {r['size']} | {r['test']} | {view} | {metrics['n']} | "+
                ' | '.join('N/A' if v is None else f'{v:.4f}' for v in vals)+' |')
    runs.mkdir(parents=True,exist_ok=True)
    (runs/'detector-comparison.md').write_text('\n'.join(lines)+'\n')
    policy=[]
    for size in config['sizes']:
        for path in sorted((runs/str(size)).glob('eval-*/summary.json')):
            coverage_path=path.parent/'coverage.json'
            for row in read_json(path):
                policy.append(dict(size=size,evaluation=path.parent.name,**row,
                    coverage=read_json(coverage_path) if coverage_path.exists() else None))
    write_json(runs/'policy-comparison.json',policy)
    lines=['# WebUI policy data scaling','',
           'Own and common tests are separate populations. n counts page × repeat trials.',
           '', '| Selected pages | Evaluation | n | Failed | Initial IoU | Final IoU | Pixel MAE | Seconds |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in policy:
        values=[row.get(k) for k in ('failure_rate','initial_box_iou','final_box_iou','pixel_mae','seconds')]
        lines.append(f"| {row['size']} | {row['evaluation']} | {row['n']} | "+
                     ' | '.join('N/A' if v is None else f'{v:.4f}' for v in values)+' |')
    lines += ['', '## Per-viewport policy results','',
        '| Selected pages | Evaluation | View | n | Failed | Initial IoU | Final IoU | Pixel MAE | Seconds |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for row in policy:
        for view,metrics in (row.get('by_view') or {}).items():
            values=[metrics.get(k) for k in ('failure_rate','initial_box_iou','final_box_iou','pixel_mae','seconds')]
            lines.append(f"| {row['size']} | {row['evaluation']} | {view} | {metrics['n']} | "+
                ' | '.join('N/A' if v is None else f'{v:.4f}' for v in values)+' |')
    (runs/'policy-comparison.md').write_text('\n'.join(lines)+'\n')


def add_parsers(sub):
    p=sub.add_parser('visual-webui-scales-prepare')
    p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--sizes',nargs='+',type=int,default=[7000,70000])
    p.add_argument('--master-counts',nargs=3,type=int,metavar=('TRAIN','VAL','TEST'))
    p.add_argument('--allocation-counts',nargs=3,type=int,metavar=('TRAIN','VAL','TEST'),
                   help='Replay audited allocation quotas; require the resulting counts to equal master-counts')
    p.add_argument('--view',help='Legacy single-view alias; use --views for new experiments')
    p.add_argument('--views',nargs='+');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('visual-webui-scales-subset');p.add_argument('--data',required=True)
    p=sub.add_parser('visual-webui-scales-summary')
    p.add_argument('--data',required=True);p.add_argument('--runs',required=True)


def run(args):
    return {'visual-webui-scales-prepare':prepare,'visual-webui-scales-subset':subset,
            'visual-webui-scales-summary':summarize}[args.command](args)

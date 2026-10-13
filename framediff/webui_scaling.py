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


def counts(size):
    if size < 10 or size % 10:
        raise ValueError('Sizes must be positive multiples of 10')
    return dict(train=size*8//10, val=size//10, test=size//10)


def nested_sources(rows, sizes, seed):
    validate_splits(rows)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source page IDs')
    if len({r['source_sha'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source HTML')
    pools={s:sorted((r for r in rows if r['split']==s),
        key=lambda r:hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()) for s in SPLITS}
    result={}
    for size in sorted(set(sizes)):
        requested=counts(size)
        if any(len(pools[s])<requested[s] for s in SPLITS):
            raise ValueError(f'Insufficient pages for {size}: requested={requested}')
        result[size]=[r for s in SPLITS for r in pools[s][:requested[s]]]
    return result


def prepare(args):
    sizes=sorted(set(args.sizes))
    for size in sizes:counts(size)
    out=Path(args.out).resolve()
    largest=counts(sizes[-1])
    guard_run(out,dict(kind='webui-scaling-811-v1',root=str(Path(args.root).resolve()),
        sizes=sizes,seed=args.seed,view=args.view),args.resume)
    master=out/'master/source'
    import_webui(SimpleNamespace(root=args.root,out=str(master),view=args.view,seed=args.seed,
        train_count=largest['train'],val_count=largest['val'],test_count=largest['test'],resume=args.resume))
    selected=nested_sources(list(read_jsonl(master/'manifest.jsonl')),sizes,args.seed)
    for size,rows in selected.items():
        folder=out/str(size)/'source'
        write_jsonl(folder/'manifest.jsonl',rows)
        for split in SPLITS:
            write_jsonl(folder/f'manifest-{split}.jsonl',(r for r in rows if r['split']==split))
        write_json(folder/'report.json',{'selected':counts(size),
            'domains':{s:len({r['group'] for r in rows if r['split']==s}) for s in SPLITS}})
    write_json(out/'split-report.json',{'sizes':{str(n):counts(n) for n in sizes},
        'common_test_size':sizes[0]//10,'common_test_scale':sizes[0],
        'protocol':'Custom 8:1:1 split by hostname, nested within a single master split; not official WebUI splits',
        'view':args.view,'seed':args.seed,
        'note':'Sizes count selected unique HTML pages before rendering/normalization exclusions.'})


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
            render_report[split]={'selected':len(chosen),'usable':len(rendered),
                                  'excluded':len(chosen)-len(rendered)}
            normalize_report[split]={'selected_pages':len(chosen),'source_pages':len(rendered),
                'clean_pages':len(rendered),'kept':len(normalized),'rejected':len(errors),
                'top_errors':Counter(r.get('stage','')+': '+r['error'] for r in errors).most_common(10)}
            freeze_report[split]={'pages':len(frozen),'failed_pages':len(normalized)-len(frozen),
                'rows':sum(1 for _ in read_jsonl(base/'css'/f'policy-{split}.jsonl'))}
        write_json(base/'rendered/report.json',{'split_coverage':render_report})
        write_json(base/'labels/prepare-report.json',normalize_report)
        write_json(base/'css/freeze-report.json',freeze_report)
        # The fixed-test evaluator follows this pointer to report all attrition.
        write_json(base/'css/config.json',{'kind':'webui-scaling-css-view-v1',
            'data':{str(base/'labels'/f'policy-{s}.jsonl'):digest(base/'labels'/f'policy-{s}.jsonl') for s in SPLITS}})
        write_json(base/'coverage.json',{'selected':counts(size),'rendering':render_report,
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
                ('n','precision_iou50','recall_iou50','small_recall_iou50','small_n','threshold')},
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
    (runs/'policy-comparison.md').write_text('\n'.join(lines)+'\n')


def add_parsers(sub):
    p=sub.add_parser('visual-webui-scales-prepare')
    p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--sizes',nargs='+',type=int,default=[7000,70000,350000])
    p.add_argument('--view',default='default_1280-720');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('visual-webui-scales-subset');p.add_argument('--data',required=True)
    p=sub.add_parser('visual-webui-scales-summary')
    p.add_argument('--data',required=True);p.add_argument('--runs',required=True)


def run(args):
    return {'visual-webui-scales-prepare':prepare,'visual-webui-scales-subset':subset,
            'visual-webui-scales-summary':summarize}[args.command](args)

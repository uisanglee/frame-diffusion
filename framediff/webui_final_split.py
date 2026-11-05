"""Publish page-disjoint 8:1:1 manifests without rebuilding master assets.

Train consumes normalized clean targets. Held-out views additionally need a
fixed corruption; generate only missing held-out examples, then rebalance after
failures. Master manifests and their original splits remain immutable caches.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from .ir import read_json, read_jsonl
from .preparation_workers import timed_page_map, write_json, write_jsonl
from .preparation_progress import PreparationProgress
from .visual_data import validate_splits
from .web_experiment import digest

CONTRACT = 'webui-final-success-page-source-811-v2'
SPLITS = ('train', 'val', 'test')


def allocate(pages, seed):
    ordered = sorted(set(pages), key=lambda p: hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest())
    n = len(ordered)
    if n < 10:
        raise ValueError('At least 10 successful unique pages are required for 8:1:1')
    counts = dict(train=n-2*(n//10), val=n//10, test=n//10)
    assignment = {}; offset = 0
    for split, count in counts.items():
        assignment.update((p, split) for p in ordered[offset:offset+count]); offset += count
    return assignment, counts


def allocate_linked(pages, seed, links):
    """Keep source-equivalent pages together; preserve v1 allocation if unique."""
    pages = set(pages)
    original, quotas = allocate(pages, seed)
    parent = {p:p for p in pages}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    owners = {}
    for page, sha in sorted(links):
        if page not in pages: continue
        if sha in owners:
            a, b = find(page), find(owners[sha])
            parent[max(a,b)] = min(a,b)
        else: owners[sha] = page
    groups = {}
    for page in sorted(pages): groups.setdefault(find(page), []).append(page)
    if len(groups) == len(pages): return original, quotas
    if len(groups) < 3:
        raise ValueError('Fewer than 3 independent source groups; cannot publish leakage-free train/val/test')
    ordered = sorted(groups.values(), key=lambda g:(-len(g), hashlib.sha256(f'{seed}:{g[0]}'.encode()).hexdigest()))
    assignment = {}; counts = dict.fromkeys(SPLITS, 0)
    for index, group in enumerate(ordered):
        empty = [s for s in SPLITS if counts[s] == 0]
        choices = empty if len(ordered)-index == len(empty) else SPLITS
        target = max(choices, key=lambda s: quotas[s]-counts[s])
        assignment.update((p,target) for p in group)
        counts[target] += len(group)
    return assignment, counts


def _heldout_cache(row, master, settings):
    from .tree_online import ONLINE_CONTRACT
    assets = {row[k]:digest(row[k]) for k in ('target_html','target_image')}
    key = hashlib.sha256(json.dumps(dict(row=row, assets=assets, settings=settings, contract=ONLINE_CONTRACT),
                                    sort_keys=True).encode()).hexdigest()
    return master/'final-split-cache'/key


def _heldout(task):
    """Keep a split-independent cache so rebalancing never rerenders a success."""
    row, master, settings, timeout = task
    from .tree_freeze import _inspect_page, _freeze_page
    from .tree_online import ONLINE_CONTRACT
    view = row['id'].rsplit('/', 1)[0]
    cache = _heldout_cache(row, master, settings)
    result = cache/'result.json'
    if result.exists():
        saved = read_json(result)
        for item in saved['rows']:
            for field in ('current_html', 'current_image'):
                if not Path(item[field]).is_file():
                    raise ValueError(f'Missing cached asset: {item[field]}')
        # Increasing the budget retries only timed-out work, never successful
        # assets or ordinary deterministic exclusions.
        if not (saved.get('timeout_s') and timeout > saved['timeout_s']):
            return saved
    pool, error = _inspect_page((row, master/'labels/.online-target-cache', settings['observation_size']))
    new = []
    if pool:
        target = pool[0]
        args = SimpleNamespace(**settings, resume=True)
        old = master/'css'
        old_config = read_json(old/'config.json') if (old/'config.json').exists() else {}
        compatible = (old_config.get('corruption_contract') == ONLINE_CONTRACT and
            all(old_config.get(k) == settings[v] for k,v in
                [('seed','seed'),('max_noise','max_noise'),('samples','samples_per_page'),
                 ('observation_size','observation_size')]))
        folder = hashlib.sha256(view.encode()).hexdigest()
        old_split = next((s for s in ('val','test') if compatible and
                          (old/s/folder/'record.json').exists()), None)
        # Old fixed examples are already valid independent of their split name.
        destination, split = (old, old_split) if old_split else (cache/'frozen', 'val')
        index = int(hashlib.sha256(view.encode()).hexdigest()[:8], 16)
        new, error = _freeze_page((args, destination, split, index, target))
    if not new and not error:
        error = dict(id=view, error='No eligible corruption target')
    saved = dict(id=view, rows=new, error=error)
    write_json(result, saved)
    return saved


def _rows(master, stage, kind):
    return [r for s in SPLITS for r in read_jsonl(master/stage/f'{kind}-{s}.jsonl')]


def publish(root, size=7000, seed=42, workers=4, timeout=180):
    import math
    if workers<1 or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('workers and timeout must be positive')
    root = Path(root).resolve(); master = root/'master'; base = root/str(size)
    source = {r['id']: r for r in read_jsonl(master/'source/manifest.jsonl')}
    tables = {stage: {kind: _rows(master,stage,kind) for kind in kinds}
              for stage,kinds in [('rendered',('pages','policy','detector')), ('labels',('pages','policy'))]}
    pages = {r['id']: r for r in tables['labels']['pages']}
    clean = {r['id'].rsplit('/',1)[0]: r for r in tables['labels']['policy']}
    available = set(source) & set(pages) & set(clean)
    for kind, rows in tables['rendered'].items():
        available &= {r['id'] if kind=='pages' else r['id'].rsplit('/',1)[0] for r in rows}
    if any(not clean[v].get('clean_target_only') or clean[v].get('lazy_contract') for v in available):
        raise ValueError('Prepare materialized --clean-targets-only labels before final splitting')
    page_ids = {v: str(source[v].get('page_id', v)) for v in available}
    # Hash constraints may appear in source, rendered, or normalized manifests.
    links = set()
    def link(row, view):
        if view in available and row.get('source_sha') is not None:
            links.add((page_ids[view], str(row['source_sha'])))
    for view in available: link(source[view], view)
    for kinds in tables.values():
        for kind, rows in kinds.items():
            for row in rows:
                link(row, row['id'] if kind=='pages' else row['id'].rsplit('/',1)[0])
    normalized_count = len(set(page_ids.values()))
    frozen = {}; rejected = {}
    settings = dict(seed=90210,max_noise=4,samples_per_page=1,observation_size=384)
    round_number = 0
    while True:
        assignment, counts = allocate_linked((page_ids[v] for v in available), seed, links)
        validate_splits([{**source[v], 'group':'webui-page:'+page_ids[v],
                          'page_id':page_ids[v], 'split':assignment[page_ids[v]]} for v in available])
        pending = sorted(v for v in available if assignment[page_ids[v]]!='train' and v not in frozen)
        if not pending: break
        round_number += 1
        progress = PreparationProgress(f'heldout-round-{round_number}',len(pending),
                                       master/'final-split-progress.json')
        processed = 0; timeouts = 0
        progress.update(0,kept=0,failures=0,timeouts=0,workers=workers,round=round_number)
        print(json.dumps(dict(final_split='prepare_missing_heldout',views=len(pending),pages=counts)),flush=True)
        failures = set()
        for index, result, error in timed_page_map(_heldout,
                ((clean[v],master,settings,timeout) for v in pending), workers, timeout):
            view=pending[index]
            if error:
                result=dict(id=view,rows=[],timeout_s=timeout,
                            error=dict(id=view,error=error,error_type='TimeoutError'))
                write_json(_heldout_cache(clean[view],master,settings)/'result.json',result)
            if result['rows']: frozen[view] = result['rows']
            else: rejected[view] = result['error']; failures.add(view)
            processed += 1
            timeouts += int(bool(error) or (result.get('error') or {}).get('error_type')=='TimeoutError')
            progress.update(processed,kept=processed-len(failures),failures=len(failures),
                            timeouts=timeouts,workers=workers,round=round_number)
        available -= failures
        # Failures can remove the last usable view of a page. Recompute quotas
        # and fill the held-out sets with other successful pages until stable.

    def move(row, view):
        return {**row, 'source_group':source[view].get('source_group',source[view]['group']),
                'source_split':source[view].get('source_split',source[view]['split']),
                'page_id':page_ids[view], 'group':'webui-page:'+page_ids[view],
                'split':assignment[page_ids[view]], 'split_contract':CONTRACT}

    outputs = {}
    selected = [move(source[v],v) for v in sorted(available)]
    validate_splits(selected)  # Also rejects duplicate source hashes across splits.
    outputs['source/manifest.jsonl'] = selected
    for s in SPLITS: outputs[f'source/manifest-{s}.jsonl'] = [r for r in selected if r['split']==s]
    for stage,kinds in tables.items():
        for kind, rows in kinds.items():
            moved = []
            for row in rows:
                v = row['id'] if kind=='pages' else row['id'].rsplit('/',1)[0]
                if v in available: moved.append(move(row,v))
            validate_splits(moved)
            for s in SPLITS: outputs[f'{stage}/{kind}-{s}.jsonl'] = [r for r in moved if r['split']==s]
    for s in SPLITS:
        views = sorted(v for v in available if assignment[page_ids[v]]==s)
        policy = [move(r,v) for v in views for r in ([clean[v]] if s=='train' else frozen[v])]
        final_pages = [move(pages[v],v) for v in views]
        if s!='train':
            for row in final_pages: row['initial_html'] = frozen[row['id']][0]['current_html']
        outputs[f'css/policy-{s}.jsonl'] = policy
        outputs[f'css/pages-{s}.jsonl'] = final_pages
        outputs[f'css/rejections-{s}.jsonl'] = []
        outputs[f'labels/rejections-{s}.jsonl'] = []
    validate_splits([r for s in SPLITS for r in outputs[f'css/policy-{s}.jsonl']])
    report = dict(contract=CONTRACT,seed=seed,common_test_scale=size,sizes={str(size):counts},
        unique_pages=sum(counts.values()),
        normalized_candidate_pages=normalized_count,
        image_success_pages=len({r.get('page_id',r['id']) for r in source.values()}),
        renderings={s:len(outputs[f'css/pages-{s}.jsonl']) for s in SPLITS},
        protocol='Successful unique pages, page/source-disjoint (not domain-disjoint); all usable views and source-linked pages stay together',
        rounding='Target val=test=floor(N/10); source-linked pages stay together, so actual counts may differ',
        target_counts=allocate((page_ids[v] for v in available), seed)[1],
        heldout_rejections=[rejected[v] for v in sorted(rejected)],assets_root=str(master),
        note='Train uses normalized clean targets; val/test additionally require fixed corruption. No automatic top-up to the directory size.')
    # Back up only published manifests/metadata, never images or HTML assets.
    previous = root/'split-report.json'
    changed = not previous.exists() or read_json(previous)!=report
    if changed and base.exists():
        history = root/'split-history'/str(time.time_ns())
        for path in [root/'config.json',previous,*base.rglob('*.json'),*base.rglob('*.jsonl')]:
            if path.is_file():
                dest=history/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest)
    for path, rows in outputs.items(): write_jsonl(base/path,rows)
    write_json(base/'css/config.json',dict(kind=CONTRACT,data={
        str(base/'labels'/f'policy-{s}.jsonl'):digest(base/'labels'/f'policy-{s}.jsonl') for s in SPLITS}))
    write_json(base/'coverage.json',dict(selected=counts,final_success=report,assets_root=str(master)))
    write_json(base/'rendered/report.json',dict(split_coverage={s:dict(usable=counts[s],
        usable_views=report['renderings'][s]) for s in SPLITS}))
    write_json(base/'labels/prepare-report.json',{s:dict(selected_pages=counts[s],
        source_pages=counts[s],clean_pages=counts[s],kept=counts[s],rejected=0,
        clean_views=report['renderings'][s],kept_views=report['renderings'][s],rejected_views=0,
        protocol=CONTRACT, note='Counts after final screening; original attrition remains in master reports') for s in SPLITS})
    write_json(base/'css/freeze-report.json',{s:dict(pages=counts[s],page_views=report['renderings'][s],
        rows=len(outputs[f'css/policy-{s}.jsonl'])) for s in SPLITS})
    write_json(root/'config.json',dict(kind=CONTRACT,sizes=[size],split_counts={str(size):counts},seed=seed))
    write_json(previous,report)
    print(json.dumps(dict(final_split_complete=True,unique_pages=counts,renderings=report['renderings'])),flush=True)
    return report


from .training_shutdown import cleanup_on_termination


@cleanup_on_termination
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--size',type=int,default=7000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--timeout',type=float,default=180,help='Active seconds per held-out view; completed caches are reused')
    args=p.parse_args()
    if args.workers<1 or args.size<1: p.error('size/workers must be positive')
    import fcntl
    root=Path(args.data);root.mkdir(parents=True,exist_ok=True)
    with (root/'.final-split.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish(root,args.size,args.seed,args.workers,args.timeout)


if __name__=='__main__': main()

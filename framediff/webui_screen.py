"""Select a new page-disjoint scale after bounded online-generation screening."""
import argparse
from collections import Counter
from concurrent.futures import TimeoutError
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from .ir import read_jsonl
from .preparation_workers import write_json, write_jsonl

SPLITS=('train','val','test')
CONTRACT='webui-screened-pages-v1'
_stop=None


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def view_id(row):
    return row['lazy_source']['id'] if row.get('lazy_contract') else row['id'].rsplit('/',1)[0]


class RowIndex:
    """Keep JSONL byte offsets, not hundreds of thousands of DOM trees, in RAM."""
    def __init__(self,path):
        self.path=Path(path);self.offsets={}
        with self.path.open('rb') as stream:
            while True:
                offset=stream.tell();line=stream.readline()
                if not line:break
                if line.strip():self.offsets.setdefault(view_id(json.loads(line)),offset)

    def get(self,key):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[key]);return json.loads(stream.readline())


def page_inventory(path):
    pages={};domains={};hashes={};views=set()
    for row in read_jsonl(path):
        page_id=row.get('page_id',row['id']);split=row['split'];group=row['group']
        if split not in SPLITS or row['id'] in views:raise ValueError('Invalid/duplicate source view')
        views.add(row['id'])
        if domains.setdefault(group,split)!=split:raise ValueError('Source domain crosses splits')
        sha=row['source_sha']
        if hashes.setdefault(sha,page_id)!=page_id:raise ValueError('Source HTML shared by different pages')
        page=pages.setdefault(page_id,dict(id=page_id,split=split,group=group,views=[],hashes={}))
        if (page['split'],page['group'])!=(split,group):raise ValueError('Page crosses splits')
        page['views'].append(row['id'])
        page['hashes'][row['id']]=sha
    return pages


def _initialize(stop):
    global _stop
    _stop=stop
    import torch
    from multiprocessing.util import Finalize
    from .browser import enable_worker_browser,close_worker_browser
    torch.set_num_threads(1);enable_worker_browser()
    Finalize(None,close_worker_browser,exitpriority=10)


def screen_view(policy,detector,options):
    """Exercise screenshot generation, which also performs abstract observations.

    Held-out rows are probed in an isolated sampler; their original split and
    fixed corruption/teacher rows are never modified or used for training.
    """
    from .lazy_webui import TrainViewCache
    from .tree_online import OnlineSampler,target_pool
    from .online_exclusions import TimeoutExclusions
    start=time.monotonic();stage='known_timeout';sampler=None;cache=None
    try:
        exclusions=TimeoutExclusions([policy])
        if exclusions.contains(exclusions.describe(policy)):
            return dict(status='rejected',reason='known_timeout',elapsed_s=0.)
        stage='detector_cache'
        if detector.get('lazy_contract'):
            cache=TrainViewCache(cache_dir=Path(options['shared_cache'])/'lazy-train-views')
            cache.get(detector)
        else:
            if not Path(detector['image']).is_file():raise FileNotFoundError(detector['image'])
        stage='policy_prepare'
        row=copy.deepcopy(policy)
        # target_pool is train-only; this copy is only a screening input.
        row['split']='train'
        if row.get('lazy_contract'):
            if row['lazy_source']['split']!='train':raise ValueError('Lazy held-out input')
            targets=target_pool([row],cache_dir=Path(options['shared_cache'])/'screening',
                                observation_size=options['observation_size'])
        elif row.get('visible_candidates'):
            row['id']=view_id(policy)
            row['corruption_observation_size']=options['observation_size']
            targets=[row]
        else:
            targets=target_pool([row],cache_dir=Path(options['shared_cache'])/'screening-inspection',
                                observation_size=options['observation_size'])
        sampler=OnlineSampler(targets,'screenshot',seed=options['seed'],
                              max_noise=options['max_noise'],attempts=8,stop=_stop)
        stage='online_probe'
        for index in range(options['samples']):
            sample=sampler.generate(index)
            if not sample.get('replacement_edit') or not sample.get('current_image'):
                raise ValueError('Incomplete online sample')
            del sample
        return dict(status='accepted',samples=options['samples'],elapsed_s=time.monotonic()-start)
    except Exception as exc:
        return dict(status='rejected',reason=stage,error=f'{type(exc).__name__}: {exc}',
                    elapsed_s=time.monotonic()-start)
    finally:
        if sampler is not None:sampler.close();sampler.view_cache.close()
        if cache is not None:cache.close()


def result_record(pool,job,timeout):
    try:
        value=job.result(timeout=timeout)
        # A later job can finish before it is consumed. Enforce the same work
        # budget on those completions, excluding shared-cache lock waits.
        if job.active_s>timeout:
            return dict(status='rejected',reason='timeout',active_s=job.active_s)
        return {**value,'active_s':job.active_s}
    except TimeoutError as exc:
        stage=job.stage
        elapsed=job.active_s+(time.monotonic()-getattr(job,'mark',time.monotonic()) if stage=='working' else 0.)
        pool.restart(job,retry=False)
        if stage=='cache_wait':
            # Another run's lock is not evidence that this page is slow.
            raise RuntimeError('Shared cache lock timed out; stop the competing job and resume') from exc
        return dict(status='rejected',reason='timeout',error=str(exc),active_s=elapsed)


def select_split(pages,needed,policy_index,detector_index,folder,pool,options,timeout):
    selected=[];reasons=Counter();attempted=0
    for start in range(0,len(pages),options['workers']):
        batch=pages[start:start+options['workers']];pending={};records={}
        for page in batch:
            for view in page['views']:
                path=folder/(hashlib.sha256(view.encode()).hexdigest()+'.json')
                if path.exists():records[view]=json.loads(path.read_text());continue
                if view not in policy_index.offsets or view not in detector_index.offsets:
                    records[view]=dict(status='rejected',reason='missing_prepared_view')
                    write_json(path,dict(view=view,**records[view]))
            # A known failed/missing view already disqualifies the whole page.
            # Do not spend browser work warming its other untested views.
            if any(records[v]['status']!='accepted' for v in page['views'] if v in records):continue
            for view in page['views']:
                if view in records:continue
                path=folder/(hashlib.sha256(view.encode()).hexdigest()+'.json')
                policy=policy_index.get(view);detector=detector_index.get(view)
                for row in (policy,detector):
                    if row['split']!=page['split'] or row['group']!=page['group']:
                        raise ValueError(f'Manifest split/domain mismatch: {view}')
                    expected=page.get('hashes',{}).get(view)
                    if expected is not None and row.get('source_sha')!=expected:
                        raise ValueError(f'Manifest source hash mismatch: {view}')
                pending[view]=(path,pool.submit(screen_view,policy,detector,options))
        for view,(path,job) in pending.items():
            records[view]=result_record(pool,job,timeout)
            write_json(path,dict(view=view,**records[view]))
            if records[view]['status']!='accepted':
                print(json.dumps(dict(screened_view=view,**records[view])),flush=True)
        for page in batch:
            attempted+=1
            failed=[records[v].get('reason','error') for v in page['views']
                    if v in records and records[v]['status']!='accepted']
            if failed:reasons.update(set(failed))
            else:
                if any(v not in records for v in page['views']):raise RuntimeError('Unscreened view')
                selected.append(page['id'])
            print(json.dumps(dict(split=page['split'],checked_pages=attempted,kept=len(selected),
                                  required=needed,reasons=dict(reasons))),flush=True)
            if len(selected)==needed:return selected,dict(checked_pages=attempted,reasons=dict(reasons))
    raise ValueError(f'Only {len(selected)}/{needed} pages passed; resume records retained in {folder}')


def publish(source,detectors,out,size,chosen,report):
    """Only manifests are copied; all successful HTML/PNG assets stay in place."""
    from .webui_scaling import counts
    selected=set().union(*(set(v) for v in chosen.values()))
    source_rows=[r for r in read_jsonl(source/'master/source/manifest.jsonl')
                 if r.get('page_id',r['id']) in selected]
    base=out/str(size);hashes={s:{r['source_sha'] for r in source_rows if r['split']==s} for s in SPLITS}
    (base/'source').mkdir(parents=True,exist_ok=True)
    write_jsonl(base/'source/manifest.jsonl',source_rows)
    render_report={};normalize_report={};freeze_report={}
    for split in SPLITS:
        views=[r for r in source_rows if r['split']==split];n=len(chosen[split])
        write_jsonl(base/'source'/f'manifest-{split}.jsonl',views)
        for name,kinds in (('rendered',('policy','pages','detector')),('labels',('policy','pages')),
                           ('css',('policy','pages'))):
            (base/name).mkdir(parents=True,exist_ok=True)
            for kind in kinds:
                src=(detectors if kind=='detector' else source/'master'/name)/f'{kind}-{split}.jsonl'
                write_jsonl(base/name/src.name,(r for r in read_jsonl(src) if r['source_sha'] in hashes[split]))
            if name!='rendered':write_jsonl(base/name/f'rejections-{split}.jsonl',[])
        render_report[split]=dict(selected=n,selected_views=len(views),usable=n,usable_views=len(views),excluded=0)
        normalize_report[split]=dict(selected_pages=n,source_pages=n,clean_pages=n,kept=n,
                                     kept_views=len(views),rejected=0,top_errors=[])
        freeze_report[split]=dict(pages=n,page_views=len(views),failed_pages=0,
                                  rows=sum(1 for _ in read_jsonl(base/'css'/f'policy-{split}.jsonl')))
    write_json(base/'rendered/report.json',dict(split_coverage=render_report))
    write_json(base/'labels/prepare-report.json',normalize_report)
    write_json(base/'css/freeze-report.json',freeze_report)
    write_json(base/'css/config.json',dict(kind=CONTRACT,data={str(base/'labels'/f'policy-{s}.jsonl'):
        file_digest(base/'labels'/f'policy-{s}.jsonl') for s in SPLITS}))
    write_json(base/'coverage.json',dict(selected=counts(size),rendering=render_report,
        normalization=normalize_report,fixed_corruption=freeze_report,screening=report,
        assets_root=str(source/'master'),warning='Runtime-screened subset; keep original assets and shared cache.'))
    write_json(out/'config.json',dict(kind=CONTRACT,sizes=[size],seed=report['seed']))
    write_json(out/'split-report.json',dict(sizes={str(size):counts(size)},common_test_scale=size,
        common_test_size=size//10,protocol='Runtime-screened subset within original domain-disjoint splits',
        note='Reject whole pages when any available view fails; fixed held-out corruptions reused.'))
    write_json(out/'screening-report.json',report)
    write_json(out/'READY.json',dict(size=size,counts=counts(size)))


def _run(args):
    from .lazy_webui import preparation_fingerprint
    from .online_workers import OnlineWorkerPool
    from .webui_scaling import counts
    from .web_experiment import guard_run
    source=Path(args.data).resolve();detectors=Path(args.detector_rendered).resolve();out=Path(args.out).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('Use a separate output root')
    if not math.isfinite(args.timeout) or min(args.workers,args.samples,args.max_noise,args.timeout,args.observation_size)<1:
        raise ValueError('Screening settings must be positive')
    requested=counts(args.size)
    shared=Path(args.shared_cache).resolve();os.environ['TUIDE_SHARED_CACHE']=str(shared)
    os.environ['TUIDE_SAVE_DIAGNOSTICS']='0'
    files=[source/'master/source/manifest.jsonl']
    for split in SPLITS:
        files += [source/'master'/name/f'{kind}-{split}.jsonl'
                  for name,kinds in (('rendered',('policy','pages')),('labels',('policy','pages')),
                                     ('css',('policy','pages'))) for kind in kinds]
        files.append(detectors/f'detector-{split}.jsonl')
    print('Checking input manifests (streaming file hashes)...',flush=True)
    config=dict(contract=CONTRACT,inputs={str(p):file_digest(p) for p in files},
        size=args.size,timeout=args.timeout,samples=args.samples,max_noise=args.max_noise,
        observation_size=args.observation_size,seed=args.seed,shared_cache=str(shared),
        preparation=preparation_fingerprint(),screen_code=file_digest(__file__))
    folder=out/'screening';guard_run(folder,config,args.resume)
    if (out/'READY.json').exists():
        print(f'Already complete: {out}',flush=True);return
    pages=page_inventory(files[0]);chosen={};report=dict(seed=args.seed,timeout_s=args.timeout,
        budget_scope='per-view: detector preparation plus policy preparation and all online probes',
        samples_per_view=args.samples,max_noise=args.max_noise,splits={})
    options=dict(samples=args.samples,max_noise=args.max_noise,seed=args.seed,workers=args.workers,
                 observation_size=args.observation_size,shared_cache=str(shared))
    pool=OnlineWorkerPool(args.workers,initializer=_initialize,cache_wait_timeout=args.timeout)
    try:
        for split in SPLITS:
            candidates=sorted((p for p in pages.values() if p['split']==split),
                key=lambda p:hashlib.sha256(f"{args.seed}:{p['id']}".encode()).hexdigest())
            if len(candidates)<requested[split]:raise ValueError(f'Insufficient {split} source pages')
            policy_index=RowIndex(source/'master/css'/f'policy-{split}.jsonl')
            detector_index=RowIndex(detectors/f'detector-{split}.jsonl')
            records=folder/split;records.mkdir(exist_ok=True)
            chosen[split],report['splits'][split]=select_split(candidates,requested[split],policy_index,
                detector_index,records,pool,options,args.timeout)
    finally:pool.shutdown()
    publish(source,detectors,out,args.size,chosen,report)
    print(f'Prepared screened dataset: {out}/{args.size}. Use a NEW training run directory.',flush=True)


def run(args):
    source=Path(args.data).resolve();out=Path(args.out).resolve()
    for src in (source,Path(args.detector_rendered).resolve()):
        if src==out or src in out.parents or out in src.parents:
            raise ValueError('Use a separate output root')
    if out.exists() and not (out/'screening/config.json').exists():
        if any(p.name!='.screening.lock' for p in out.iterdir()):
            raise ValueError('Output contains unrelated files; use a fresh output root')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.screening.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another screening process owns this output') from None
        pidfile=Path(os.environ['SCREEN_PID_FILE']) if os.environ.get('SCREEN_PID_FILE') else None
        try:
            if pidfile:
                pidfile.parent.mkdir(parents=True,exist_ok=True);pidfile.write_text(str(os.getpid())+'\n')
            _run(args)
        finally:
            if pidfile and pidfile.exists() and pidfile.read_text().strip()==str(os.getpid()):pidfile.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,help='Existing full policy dataset root with master/')
    parser.add_argument('--detector-rendered',required=True,help='Full candidate pool detector rendered directory')
    parser.add_argument('--out',required=True)
    parser.add_argument('--size',type=int,default=7000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=180,help='Active seconds per view including all probes')
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--max-noise',type=int,default=4)
    parser.add_argument('--observation-size',type=int,default=384)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--shared-cache',default=os.environ.get('TUIDE_SHARED_CACHE','data/webui-shared-cache'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    def interrupted(*_):raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:run(args)
    except KeyboardInterrupt:raise SystemExit(130)
    finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':main()

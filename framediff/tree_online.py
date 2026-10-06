"""Bounded CPU-browser prefetch of fresh corruption/path states.

Reference: tree-diffusion td/samplers/mutator.py forward_process_with_path.
We sample CSS declarations on fixed real DOMs, not arbitrary TinySVG programs.
Task seeds depend on consumed sample indices, never worker scheduling or GPU.
"""
import copy
import hashlib
import math
import multiprocessing as mp
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing.util import Finalize
from pathlib import Path

from .html_bridge import HtmlBrowser
from .html_feedback import refresh_geometry
from .ir import read_jsonl
from .tree_edits import FIELDS, CONTRACT, current_state, execute, extract, repair_path, value_valid
from .web_experiment import digest

ONLINE_CONTRACT = 'online-css-path-v1'


def target_pool(rows, manifest=None):
    pages = {p['id']:p for p in read_jsonl(manifest)} if manifest else {}
    targets = {}
    for row in rows:
        if row['split'] != 'train': raise ValueError('Online corruption only accepts training pages')
        page_id = row['id'].rsplit('/',1)[0]
        if page_id in targets: continue
        html = row.get('target_html') or pages.get(page_id,{}).get('html')
        if not html:
            raise ValueError(f'Missing clean HTML for {page_id}; pass --online-targets pages-train.jsonl')
        if pages.get(page_id,{}).get('split','train') != 'train':
            raise ValueError('Online target manifest contains a non-training page')
        for key in ('group','source_sha'):
            if key in pages.get(page_id,{}) and pages[page_id][key]!=row[key]:
                raise ValueError(f'Online target identity mismatch: {page_id} {key}')
        target = {k:copy.deepcopy(row[k]) for k in ('group','split','source_sha','viewport','current',
                  'target_image','target_elements')}
        for key in ('predicted_target_elements','parser_provenance','parser_sha'):
            if key in row: target[key] = row[key]
        target.update(id=page_id,target_html=str(Path(html).resolve()))
        targets[page_id] = target
    if not targets: raise ValueError('Empty online target pool')
    # Stable order makes sample indices independent of repeated offline examples.
    return [targets[k] for k in sorted(targets)]


def asset_signatures(targets):
    paths={t[k] for t in targets for k in ('target_html','target_image','predicted_target_elements') if t.get(k)}
    return {path:digest(path) for path in sorted(paths)}


def task_seed(seed, index):
    return int(hashlib.sha256(f'{ONLINE_CONTRACT}:{seed}:{index}'.encode()).hexdigest()[:16],16)


def sample_mutation(state, viewport, rng):
    """Mix edits of existing declarations with insertion of absent overrides.

    Removal of existing declarations makes SET necessary during repair. This
    does not guarantee balanced labels when source pages use only stylesheets.
    """
    existing=[];absent=[]
    for node in range(1,len(state)):
        for j,field in enumerate(FIELDS):
            value,priority=state[node][j]
            if value and not value_valid(value,field): continue  # must be restorable
            (existing if value else absent).append((node,field,value,priority))
    pools=[p for p in (existing,absent) if p]
    if not pools: raise ValueError('No grammar-repairable declarations')
    node,field,old,priority=rng.choice(rng.choice(pools))
    if old and rng.random()<.25: return [node,field,'','']
    axis=viewport[0 if field in ('width','margin-left','margin-right') else 1]
    if rng.random()<.2:
        value=f'{rng.randint(5,95) if field in ("width","height") else rng.randint(-15,15)}%'
    else:
        value=f'{round(rng.uniform(.01,.8)*axis if field in ("width","height") else rng.uniform(-.15,.15)*axis,3):g}px'
    return [node,field,value,'important']


class OnlineSampler:
    def __init__(self, targets, mode, seed=42, max_noise=4, attempts=8, stop=None):
        if mode not in ('abstract','screenshot') or min(max_noise,attempts)<1: raise ValueError('Invalid online configuration')
        self.targets=targets;self.mode=mode;self.seed=seed;self.max_noise=max_noise;self.attempts=attempts;self.stop=stop
        self.by_id={t['id']:t for t in targets}
        self.browser=None

    @lru_cache(maxsize=64)
    def html(self,path): return Path(path).read_text()

    @lru_cache(maxsize=64)
    def clean_state(self,page_id):
        target=self.by_id[page_id]
        return extract(self.browser,self.html(target['target_html']),target['current'])['state']

    def close(self):
        if self.browser is not None:
            self.browser.__exit__(None,None,None);self.browser=None

    def generate(self,index):
        rng=random.Random(task_seed(self.seed,index));errors=[];started=time.perf_counter()
        for attempt in range(self.attempts):
            if self.stop is not None and self.stop.is_set(): raise RuntimeError('Online producer stopped')
            target=rng.choice(self.targets)
            try:
                if self.browser is None: self.browser=HtmlBrowser().__enter__()
                else: self.browser.reset_context()
                browser=self.browser;tree=target['current'];viewport=target['viewport']
                browser.page.set_default_timeout(10000)
                html=self.html(target['target_html'])
                clean=self.clean_state(target['id'])
                browser.load(html,viewport)
                state=current_state(browser,tree);noise=[]
                for _ in range(rng.randint(1,self.max_noise)):
                    if self.stop is not None and self.stop.is_set(): raise RuntimeError('Online producer stopped')
                    mutation=sample_mutation(state,viewport,rng)
                    execute(browser,tree,mutation);state=current_state(browser,tree);noise.append(mutation)
                path=repair_path(state,clean,rng.getrandbits(64))
                if not path: raise ValueError('Corruption returned to clean program')
                # Select an intermediate reverse-path state, not only the
                # fully corrupted endpoint and not history in reverse order.
                depth=rng.randrange(len(path))
                for edit in path[:depth]: execute(browser,tree,edit)
                state=current_state(browser,tree)
                remaining=path[depth:]
                if sorted(remaining)!=sorted(repair_path(state,clean)):
                    raise ValueError('CSS execution changed the symbolic repair path')
                if not remaining: raise ValueError('Sampled clean state has no next edit')
                boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
                boxes[tree['nodes'][0]['id']]=[0,0,*viewport]
                result={**target,'id':target['id']+f'/online-{index}',
                        'current':refresh_geometry(tree,boxes),'current_boxes':boxes,
                        'declaration_state':state,'replacement_edit':remaining[0],
                        'teacher_strategy':CONTRACT,'symbolic_distance':len(remaining)}
                if self.mode=='screenshot':
                    # Bytes travel through a bounded queue, never thousands of
                    # transient files on disk. Target images are never recaptured.
                    result['current_image']=browser.page.screenshot(animations='disabled')
                result['online']={'index':index,'seed':task_seed(self.seed,index),'retries':attempt,
                                  'corruptions':noise,'reverse_prefix':depth,'path_length':len(path),
                                  'screenshot_captures':int(self.mode=='screenshot'),
                                  'seconds':time.perf_counter()-started,'errors':errors}
                return result
            except Exception as exc:
                errors.append({'page':target['id'],'error':str(exc)})
                if self.browser is not None:
                    try: self.close()
                    except Exception: self.browser=None
        raise RuntimeError(f'Online sample {index} failed after {self.attempts} attempts: {errors}')


_sampler=None


def _initialize(targets,mode,seed,max_noise,attempts,stop):
    global _sampler
    import torch
    torch.set_num_threads(1)  # No CUDA/model/detector in workers.
    _sampler=OnlineSampler(targets,mode,seed,max_noise,attempts,stop)
    Finalize(None,_sampler.close,exitpriority=10)


def _generate(index): return _sampler.generate(index)


class OnlineStream:
    """Ordered bounded prefetch; cursor counts CONSUMED, not queued examples."""
    def __init__(self,targets,mode,seed=42,max_noise=4,workers=2,prefetch=4,attempts=8,timeout=180,start=0):
        if min(workers,prefetch,attempts,max_noise,timeout)<1 or not math.isfinite(timeout) or start<0:
            raise ValueError('Invalid online stream configuration')
        self.args=(targets,mode,seed,max_noise,attempts)
        self.workers=workers;self.prefetch=prefetch;self.timeout=timeout;self.cursor=start;self.next_index=start
        self.pending=deque();self.pool=None

    def __enter__(self):
        context=mp.get_context('spawn')  # Never fork a CUDA-initialized parent.
        self.stop=context.Event()
        self.pool=ProcessPoolExecutor(self.workers,mp_context=context,initializer=_initialize,initargs=(*self.args,self.stop))
        self._fill();return self

    def _fill(self):
        while len(self.pending)<self.prefetch:
            self.pending.append(self.pool.submit(_generate,self.next_index));self.next_index+=1

    def take(self,count):
        rows=[]
        for _ in range(count):
            if not self.pool: raise RuntimeError('Online stream is not open')
            try: row=self.pending[0].result(timeout=self.timeout)
            except Exception as exc: raise RuntimeError(f'Online producer failed/stalled at sample {self.cursor}: {exc}') from exc
            if row['online']['index']!=self.cursor: raise RuntimeError('Online sample order changed')
            self.pending.popleft();self.cursor+=1;rows.append(row);self._fill()
        return rows

    def __exit__(self,*args):
        if self.pool is not None:
            self.stop.set()
            for future in self.pending: future.cancel()
            self.pool.shutdown(wait=True,cancel_futures=True);self.pool=None

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

ONLINE_CONTRACT = 'online-css-existing-values-v3'


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
        for key in ('predicted_target_elements','parser_provenance','parser_sha','css_owners'):
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
    """Replace an existing value in place, preserving its owner and priority.

    Sample uniformly over supported, present owner/property pairs. Absent fields
    are not mutation sites: inserting an override would teach its removal.
    Browser probes subsequently reject shadowed or geometrically inert edits.
    """
    existing=[]
    for node in range(1,len(state)):
        for j,field in enumerate(FIELDS):
            value,priority=state[node][j]
            if value and value_valid(value,field):
                existing.append((node,field,value,priority))
    if not existing: raise ValueError('No supported existing CSS values to corrupt')
    node,field,old,priority=rng.choice(existing)
    axis=viewport[0 if field in ('width','margin-left','margin-right') else 1]
    if rng.random()<.2:
        value=f'{rng.randint(5,95) if field in ("width","height") else rng.randint(-15,15)}%'
    else:
        value=f'{round(rng.uniform(.01,.8)*axis if field in ("width","height") else rng.uniform(-.15,.15)*axis,3):g}px'
    # Avoid no-op string replacements; the browser also checks actual effects.
    if value==old:
        value='1px' if old!='1px' else '2px'
    return [node,field,value,priority]


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
        if 'css_owners' in target:
            from .css_owners import read
            return read(self.browser,target['current'],self.html(target['target_html']))['state']
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
                from . import css_owners
                owner_mode='css_owners' in target
                owners=css_owners.read(browser,tree)['owners'] if owner_mode else None
                if owner_mode and owners!=target['css_owners']: raise ValueError('CSS owner identity changed')
                def state_now():
                    return css_owners.read(browser,tree)['state'] if owner_mode else current_state(browser,tree)
                def apply(edit):
                    return css_owners.execute(browser,owners,edit) if owner_mode else execute(browser,tree,edit)
                def geometry():return browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
                clean_boxes=geometry() if owner_mode else None
                state=state_now();noise=[];probe_rejections=0
                for _ in range(rng.randint(1,self.max_noise)):
                    if self.stop is not None and self.stop.is_set(): raise RuntimeError('Online producer stopped')
                    for trial in range(24 if owner_mode else 1):
                        mutation=sample_mutation(state,viewport,rng)
                        if owner_mode:
                            # Preserve cascade priority when mutating an existing
                            # declaration. Do not promote a shadowed rule to winner.
                            old=state[mutation[0]][FIELDS.index(mutation[1])]
                            if mutation[2]: mutation[3]=old[1]
                            before_values=css_owners.computed(browser,owners,mutation);before_boxes=geometry()
                        apply(mutation)
                        if owner_mode and (before_values==css_owners.computed(browser,owners,mutation) or before_boxes==geometry()):
                            apply([mutation[0],mutation[1],*old]);probe_rejections+=1;continue
                        state=state_now();noise.append(mutation);break
                path=repair_path(state,clean,rng.getrandbits(64))
                if not path: raise ValueError('Corruption returned to clean program')
                # Select an intermediate reverse-path state, not only the
                # fully corrupted endpoint and not history in reverse order.
                depth=rng.randrange(len(path))
                for edit in path[:depth]: apply(edit)
                state=state_now()
                remaining=path[depth:]
                if sorted(remaining)!=sorted(repair_path(state,clean)):
                    raise ValueError('CSS execution changed the symbolic repair path')
                if not remaining: raise ValueError('Sampled clean state has no next edit')
                boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
                if owner_mode and boxes==clean_boxes:raise ValueError('Remaining CSS differences have no visible geometry effect')
                boxes[tree['nodes'][0]['id']]=[0,0,*viewport]
                result={**target,'id':target['id']+f'/online-{index}',
                        'current':refresh_geometry(tree,boxes),'current_boxes':boxes,
                        'declaration_state':state,'target_declaration_state':clean,
                        'replacement_edit':remaining[0],
                        'teacher_strategy':css_owners.CONTRACT if owner_mode else CONTRACT,'symbolic_distance':len(remaining),
                        'corruption_contract':ONLINE_CONTRACT}
                if owner_mode:
                    result['css_owners']=owners
                    result['current_html_text']=browser.page.content()
                if self.mode=='screenshot':
                    # Bytes travel through a bounded queue, never thousands of
                    # transient files on disk. Target images are never recaptured.
                    result['current_image']=browser.page.screenshot(animations='disabled')
                result['online']={'index':index,'seed':task_seed(self.seed,index),'retries':attempt,
                                  'corruptions':noise,'reverse_prefix':depth,'path_length':len(path),
                                  'screenshot_captures':int(self.mode=='screenshot'),
                                  'seconds':time.perf_counter()-started,'errors':errors}
                if owner_mode:
                    result['online'].update(probe_rejections=probe_rejections,
                        teacher_owner=owners[remaining[0][0]]['kind'],
                        rule_mutations=sum(owners[e[0]]['kind']=='rule' for e in noise))
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

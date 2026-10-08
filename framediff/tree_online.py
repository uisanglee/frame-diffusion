"""Bounded CPU-browser prefetch of fresh corruption/path states.

Reference: tree-diffusion td/samplers/mutator.py forward_process_with_path.
We sample CSS declarations on fixed real DOMs, not arbitrary TinySVG programs.
Task seeds depend on consumed sample indices, never worker scheduling or GPU.
"""
import copy
import contextlib
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import random
import time
import tempfile
from importlib.metadata import version
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

ONLINE_CONTRACT = 'online-css-visible-values-v6-size-margin'
SAMPLING_CONTRACT = 'property-first-owner-uniform-v1'
# Bump when eligibility semantics change; old syntax-only caches are invalid.
INSPECTION_LOGIC_VERSION = 'visible-candidates-v1-four-probes'
PROBES_PER_SITE = 4
ABSTRACTION_EPS = 1e-6


def mutation_sites(state):
    """Editable, present declarations eligible for value replacement."""
    if any(len(props)!=len(FIELDS) for props in state):
        raise ValueError('CSS field schema changed; prepare six-field size/margin labels')
    return [(node,field,value,priority)
            for node in range(1,len(state))
            for field,(value,priority) in zip(FIELDS,state[node])
            if value and value_valid(value,field)]


def inspection_fingerprint():
    # Invalidate on changes to parsing, loading, or supported CSS values.
    root = Path(__file__).parent
    return {'tree_online.py': INSPECTION_LOGIC_VERSION,
            **{name: digest(root/name) for name in
               ('tree_edits.py','css_owners.py','html_bridge.py','browser.py','visual.py','explicit_html.py')},
            'playwright': version('playwright')}


@contextlib.contextmanager
def inspection_cache_entry(directory, target, html, fingerprint):
    if directory is None:
        yield None
        return
    signature = {'code': fingerprint, 'html': hashlib.sha256(html.encode()).hexdigest(),
                 'viewport': target['viewport'],
                 'ids': [n['id'] for n in target['current']['nodes']],
                 'observation_size':target.get('corruption_observation_size',384),
                 'owners': target.get('css_owners')}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Different scales/modalities may start simultaneously. Only one process
    # inspects a given page; others read its completed atomic cache entry.
    with (directory/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield directory/f'{key}.json'
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def save_inspection(path, state, error=None, **metadata):
    if path is None: return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({'state': state, 'error': error, **metadata}, stream, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def abstraction(browser, tree, viewport, size):
    """Same clipped occupancy+boundary representation used by the policy."""
    from .visual import elements,semantic_masks
    from .explicit_html import current_clip_boxes
    ids=[n['id'] for n in tree['nodes'][1:]]
    boxes=browser.tagged_boxes(ids);boxes[tree['nodes'][0]['id']]=[0,0,*viewport]
    items=elements(tree,boxes,viewport,current_clip_boxes(browser,ids))
    return semantic_masks(items,viewport,size),boxes,items


def changed(a,b):
    return bool((a-b).abs().max().item()>ABSTRACTION_EPS)


def inspect_candidates(browser,target,html):
    """Find witnessed visible edits; finite probing is not exhaustive CSS search."""
    from . import css_owners
    from .visual import annotate
    tree=copy.deepcopy(target['current']);viewport=target['viewport']
    browser.load(html,viewport);annotate(browser,tree)
    owner_mode='css_owners' in target
    if owner_mode:
        parsed=css_owners.read(browser,tree)
        if parsed['owners']!=target['css_owners']:raise ValueError('CSS owner identity changed')
        state=parsed['state']
    else:state=current_state(browser,tree)
    def apply(edit):
        return css_owners.execute(browser,target['css_owners'],edit) if owner_mode else execute(browser,tree,edit)
    before,boxes,items=abstraction(browser,tree,viewport,target.get('corruption_observation_size',384))
    candidates=[]
    for site in mutation_sites(state):
        node,field,value,priority=site
        rng=random.Random(hashlib.sha256(f'{node}:{field}:{value}'.encode()).digest())
        for _ in range(PROBES_PER_SITE):
            edit=sample_mutation(state,viewport,rng,explicit='data-tuide-explicit-id' in html,sites=[site])
            try:
                apply(edit)
                after,new_boxes,_=abstraction(browser,tree,viewport,target.get('corruption_observation_size',384))
                visible=new_boxes!=boxes and changed(before,after)
            except ValueError as exc:
                if 'Explicit ' not in str(exc):raise
                visible=False
            finally:apply([node,field,value,priority])
            if visible:
                candidates.append(edit);break
    return state,dict(visible_candidates=candidates,target_elements=items,
                     visual_classes=[n.get('visual_class',0) for n in tree['nodes']])


def target_pool(rows, manifest=None, cache_dir=None, observation_size=384):
    if observation_size<1:raise ValueError('Invalid corruption observation size')
    if cache_dir is None and manifest:
        cache_dir = Path(manifest).resolve().parent/'.online-target-cache'
    pages = {p['id']:p for p in read_jsonl(manifest)} if manifest else {}
    targets = {}
    seen = set()
    excluded = 0
    inspection_errors = 0
    for row in rows:
        if row.get('hierarchy'):raise ValueError('Flat targets are retired; use original parent-preserving HTML')
        if row['split'] != 'train': raise ValueError('Online corruption only accepts training pages')
        page_id = row['id'].rsplit('/',1)[0]
        if page_id in seen: continue
        seen.add(page_id)
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
        for key in ('predicted_target_elements','parser_provenance','parser_sha','css_owners',
                    'target_declaration_state','normalization_contract'):
            if key in row: target[key] = row[key]
        target.update(id=page_id,target_html=str(Path(html).resolve()),corruption_observation_size=observation_size)
        clean_state=row.get('target_declaration_state')
        # Stylesheet preparation already parsed the clean program. Excluding
        # pages with no legal replacement sites here prevents unlucky batches
        # of permanently unusable pages from terminating online training.
        if clean_state is not None and not mutation_sites(clean_state):
            excluded += 1
            continue
        targets[page_id] = target
    # Even preparsed declarations need execution probes. Cache accepted and
    # excluded pages, including witnessed edits, independently of model scale.
    browser = None
    inspected = 0
    cache_hits = 0
    fingerprint = inspection_fingerprint() if cache_dir is not None else None
    try:
        for page_id, target in list(targets.items()):
            html = Path(target['target_html']).read_text()
            with inspection_cache_entry(cache_dir, target, html, fingerprint) as entry:
                state = None; error = None; cached = False;metadata={}
                if entry is not None and entry.exists():
                    try:
                        saved = json.loads(entry.read_text())
                        state, error = saved.get('state'), saved.get('error')
                        if error is not None:
                            if not isinstance(error,str) or state is not None: raise ValueError('Invalid cached error')
                        else:
                            if not isinstance(state,list): raise ValueError('Invalid cached state')
                            mutation_sites(state)
                            metadata={k:saved[k] for k in ('visible_candidates','target_elements','visual_classes')}
                            if not isinstance(metadata['visible_candidates'],list):raise ValueError('Invalid candidate cache')
                        cached = True
                    except (ValueError, KeyError, TypeError): state = None
                if cached:
                    cache_hits += 1
                else:
                    inspected += 1
                    try:
                        if browser is None: browser = HtmlBrowser().__enter__()
                        else: browser.reset_context()
                        state,metadata=inspect_candidates(browser,target,html)
                    except Exception as exc:
                        error = f'{type(exc).__name__}: {exc}'
                    # Transient browser failures must be retried next run, not
                    # persisted as permanently unusable source pages.
                    if error is None:save_inspection(entry,state,**metadata)
            if error is not None:
                print({'online_target_inspection_error':page_id,'error':error},flush=True)
                del targets[page_id]
                excluded += 1
                inspection_errors += 1
            elif not metadata['visible_candidates']:
                del targets[page_id]
                excluded += 1
            else:
                target['target_declaration_state'] = state
                target.update(metadata)
                for node,label in zip(target['current']['nodes'],metadata['visual_classes']):node['visual_class']=label
            processed = inspected + cache_hits
            if processed == 1 or processed % 100 == 0:
                print({'online_target_inspection': inspected, 'cache_hits': cache_hits,
                       'excluded_no_editable_css': excluded-inspection_errors,
                       'excluded_no_visible_candidate':excluded-inspection_errors,
                       'excluded_inspection_error': inspection_errors}, flush=True)
    finally:
        if browser is not None: browser.__exit__(None,None,None)
    print({'online_target_filter': {'pages': len(seen), 'kept': len(targets),
          'visible_candidates':sum(len(t['visible_candidates']) for t in targets.values()),
          'excluded_no_editable_css': excluded-inspection_errors,
          'excluded_no_visible_candidate':excluded-inspection_errors,
          'excluded_inspection_error': inspection_errors, 'inspected_missing_states': inspected,
          'cache_hits': cache_hits, 'cache_dir': str(cache_dir) if cache_dir is not None else None}}, flush=True)
    if not targets: raise ValueError('Empty online target pool after filtering pages without witnessed abstraction-changing edits')
    # Stable order makes sample indices independent of repeated offline examples.
    return [targets[k] for k in sorted(targets)]


def asset_signatures(targets):
    paths={t[k] for t in targets for k in ('target_html','target_image','predicted_target_elements') if t.get(k)}
    return {path:digest(path) for path in sorted(paths)}


def task_seed(seed, index):
    return int(hashlib.sha256(f'{ONLINE_CONTRACT}:{seed}:{index}'.encode()).hexdigest()[:16],16)


def choose_mutation_site(sites, rng):
    """Uniform property, then uniform owner, then a witness for that owner.

    Group witnesses too: multiple cached values must not increase an owner's
    selection probability. Single-property/site probes retain their RNG order.
    """
    if not sites: raise ValueError('No supported existing CSS values to corrupt')
    fields=list(dict.fromkeys(site[1] for site in sites))
    field=fields[0] if len(fields)==1 else rng.choice(fields)
    candidates=[site for site in sites if site[1]==field]
    owners=list(dict.fromkeys(site[0] for site in candidates))
    owner=owners[0] if len(owners)==1 else rng.choice(owners)
    return rng.choice([site for site in candidates if site[0]==owner])


def sample_mutation(state, viewport, rng, explicit=False, sites=None):
    """Replace an existing value in place, preserving its owner and priority.

    Sample a present property uniformly, then an eligible owner uniformly. Absent fields
    are not mutation sites: inserting an override would teach its removal.
    Browser probes subsequently reject shadowed or geometrically inert edits.
    """
    existing=mutation_sites(state) if sites is None else sites
    if not existing: raise ValueError('No supported existing CSS values to corrupt')
    node,field,old,priority=choose_mutation_site(existing,rng)
    axis=viewport[0 if field in ('width','margin-left','margin-right') else 1]
    if rng.random()<.2 and not explicit:
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
        # The stylesheet preparation pass already parsed and validated this
        # exact clean program. Reuse that authoritative state so pool
        # filtering and mutation sampling cannot disagree after a reparse.
        if 'target_declaration_state' in target:
            return copy.deepcopy(target['target_declaration_state'])
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
                if 'visible_candidates' not in target:
                    inspected,metadata=inspect_candidates(browser,target,html)
                    target.update(metadata,target_declaration_state=inspected)
                    for node,label in zip(tree['nodes'],metadata['visual_classes']):node['visual_class']=label
                    self.clean_state.cache_clear()
                if not target['visible_candidates']:raise ValueError('No visible corruption candidates')
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
                size=target.get('corruption_observation_size',384)
                clean_mask,clean_boxes,_=abstraction(browser,tree,viewport,size)
                frame=clean_mask
                state=state_now();noise=[];probe_rejections=0
                for _ in range(rng.randint(1,self.max_noise)):
                    if self.stop is not None and self.stop.is_set(): raise RuntimeError('Online producer stopped')
                    for trial in range(24):
                        eligible={(e[0],e[1]) for e in target['visible_candidates']}
                        sites=[s for s in mutation_sites(state) if (s[0],s[1]) in eligible]
                        mutation=sample_mutation(state,viewport,rng,explicit='data-tuide-explicit-id' in html,sites=sites)
                        # Witness edits ensure even a narrow responsive range
                        # gets tried; their effect is rechecked in this state.
                        if trial>=12:
                            mutation=list(choose_mutation_site(target['visible_candidates'],rng))
                        old=state[mutation[0]][FIELDS.index(mutation[1])]
                        if owner_mode:
                            # Preserve cascade priority when mutating an existing
                            # declaration. Do not promote a shadowed rule to winner.
                            old=state[mutation[0]][FIELDS.index(mutation[1])]
                            if mutation[2]: mutation[3]=old[1]
                            before_values=css_owners.computed(browser,owners,mutation);before_boxes=geometry()
                        try:apply(mutation)
                        except ValueError as exc:
                            if 'Explicit ' not in str(exc):raise
                            probe_rejections+=1;continue
                        after_mask,after_boxes,_=abstraction(browser,tree,viewport,size)
                        if not changed(frame,after_mask) or (owner_mode and (before_values==css_owners.computed(browser,owners,mutation) or before_boxes==geometry())):
                            apply([mutation[0],mutation[1],*old]);probe_rejections+=1;continue
                        frame=after_mask;state=state_now();noise.append(mutation);break
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
                frame,boxes,_=abstraction(browser,tree,viewport,size)
                if not changed(frame,clean_mask):raise ValueError('Remaining CSS differences have no abstraction effect')
                # Reversing other edits can make a formerly active declaration
                # invisible. Do not emit an unobservable next-action label.
                for teacher in remaining:
                    old=state[teacher[0]][FIELDS.index(teacher[1])]
                    apply(teacher)
                    next_frame,_,_=abstraction(browser,tree,viewport,size)
                    apply([teacher[0],teacher[1],*old])
                    if changed(frame,next_frame):break
                else:raise ValueError('No abstraction-changing reverse label')
                result={**target,'id':target['id']+f'/online-{index}',
                        'current':refresh_geometry(tree,boxes),'current_boxes':boxes,
                        'declaration_state':state,'target_declaration_state':clean,
                        'replacement_edit':teacher,
                        'teacher_strategy':css_owners.CONTRACT if owner_mode else CONTRACT,'symbolic_distance':len(remaining),
                        'corruption_contract':ONLINE_CONTRACT}
                from .explicit_html import feedback_metadata
                result.update(feedback_metadata(browser,tree))
                if owner_mode:
                    result['css_owners']=owners
                    result['current_html_text']=browser.page.content()
                if self.mode=='screenshot':
                    # Bytes travel through a bounded queue, never thousands of
                    # transient files on disk. Target images are never recaptured.
                    from .explicit_html import feedback_screenshot
                    result['current_image']=feedback_screenshot(browser,animations='disabled')
                result['online']={'index':index,'seed':task_seed(self.seed,index),'retries':attempt,
                                  'visible_candidate_count':len(target['visible_candidates']),
                                  'abstraction_max_difference':float((frame-clean_mask).abs().max()),
                                  'corruptions':noise,'reverse_prefix':depth,'path_length':len(path),
                                  'screenshot_captures':int(self.mode=='screenshot'),
                                  'seconds':time.perf_counter()-started,'errors':errors}
                if owner_mode:
                    result['online'].update(probe_rejections=probe_rejections,
                        teacher_owner=owners[teacher[0]]['kind'],
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

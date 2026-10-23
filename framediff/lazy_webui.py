"""Bounded train-view materialization with optional persistent on-demand cache.

Only generated temporary files are evicted. Raw HTML and prepared evaluation
assets are never modified. Persistent entries are content/settings addressed
and atomically committed so independent online workers can safely share them.
"""
import copy
import ast
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
from importlib.metadata import version

CONTRACT = 'lazy-webui-train-view-v1'
DISK_CACHE_CONTRACT = 'lazy-webui-train-view-cache-v2-verified'


def compact_new_cache(work, result):
    """Trim only unpublished generated work; never prune an existing cache."""
    work=work.resolve()
    keep=set()
    def references(value):
        if isinstance(value,dict):
            for item in value.values():references(item)
        elif isinstance(value,list):
            for item in value:references(item)
        elif isinstance(value,str) and value.startswith('/') and len(value)<4096 and '\x00' not in value:
            path=Path(value).resolve()
            if path.is_relative_to(work) and path.is_file():keep.add(path)
    references(result)
    # Detector relabeling uses image.with_suffix('.html') as its source.
    if 'image' in result:
        html=Path(result['image']).with_suffix('.html').resolve()
        if html.is_file():keep.add(html)
    # Keep human-readable outcomes; preparation-only metadata is disposable
    # because result.json contains the complete verified training row.
    reports={'report.json','candidate-inspection.json','page-status.json','failure.json'}
    for path in work.rglob('*'):
        if path.is_file() and path.name in reports:keep.add(path)
    for path in list(work.rglob('*')):
        if path.is_file() and path not in keep:path.unlink()
    # Preserve paths but store byte-identical immutable assets only once.
    seen={}
    for path in sorted(keep):
        if path.suffix.lower() not in ('.html','.png','.jpg','.jpeg','.webp'):continue
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha in seen:
            path.unlink();os.link(seen[sha],path)
        else:seen[sha]=path
    for path in sorted(work.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if path.is_dir() and not any(path.iterdir()):path.rmdir()


@lru_cache(maxsize=1)
def preparation_fingerprint():
    root=Path(__file__).parent
    names=('safe_sizes.py','tree_normalize.py','html_bridge.py','browser.py',
           'box_scroll_guard.py','explicit_html.py','css_owners.py','tree_edits.py',
           'visual.py','visual_data.py')
    # Stream scheduling/logging cannot change a prepared page. Fingerprint only
    # online functions that construct its seed and inspect editable candidates.
    online=ast.parse((root/'tree_online.py').read_text())
    excluded={'OnlineStream','_initialize','_generate','OnlineGenerationError'}
    semantic=[node for node in online.body if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Assign))
              and getattr(node,'name',None) not in excluded
              and not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_sampler' for t in node.targets))]
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    # Optional seed-file retention changes storage only; preserve existing keys.
    if hashes['tree_normalize.py']=='5f2f562d97282db97fcbe96f6ab5cb2bfccfd347519eaeba3973a810bde6315c':
        hashes['tree_normalize.py']='99abfa568147b5581248b90f477b920c26202fd5dffa02f9059135ea5eb5cc7f'
    # Audited storage-only change: screenshot retention does not change the
    # prepared training example. Unknown future edits still invalidate caches.
    if hashes['safe_sizes.py']=='1acceb2dff0b09384551c9c63276ceb3d4807dab1de160075129a97d4532b4a5':
        hashes['safe_sizes.py']='82670e65a792de714d8cb44f27f3514d717cce013a28d59e0d95c56cbaa4fbd2'
    online_sha=hashlib.sha256(''.join(ast.dump(n) for n in semantic).encode()).hexdigest()
    # Instance-local LRU ownership, browser release, and timing-only sampler
    # instrumentation do not change labels or prepared page contents.
    if online_sha in ('e98088a0e5027f27a10c2c038f23bb8d2a24227af7a4e294e5049ed9e7c8cfc4',
                      '6e923210000160149956198f2df1e10d88086ec38b08ff38dbbc34fc7390aa17',
                      '95c26191b15f4c248138edb4380f8156812c757832f393ee5fbe89f74f84710f'):
        online_sha='e598fecb8a9bb423bba5e9cf118b338f89be0bbe3d83195d5512e4e7c2fbf53d'
    return {**hashes,
            'online_preparation':online_sha,
            'playwright':version('playwright')}


class TrainViewCache:
    def __init__(self, capacity=2, cache_dir=None):
        if capacity < 1: raise ValueError('Cache capacity must be positive')
        self.capacity=capacity;self.entries=OrderedDict()
        self.last_profile=None
        self.cache_dir=Path(cache_dir).resolve() if cache_dir else None
        self.legacy_cache_dir=self.cache_dir
        shared=os.environ.get('TUIDE_SHARED_CACHE')
        if shared and self.cache_dir is not None:
            self.cache_dir=Path(shared).resolve()/'lazy-train-views'
        if self.cache_dir is not None:self.cache_dir.mkdir(parents=True,exist_ok=True)

    def close(self):
        for directory,_ in self.entries.values():
            if directory is not None:directory.cleanup()
        self.entries.clear()

    def _key(self,row,policy):
        source=row['lazy_source'];settings=row['lazy_settings']
        payload=(DISK_CACHE_CONTRACT,source['id'],source['source_sha'],
                 repr(sorted(settings.items())),policy,preparation_fingerprint())
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    @staticmethod
    def _relocate(value,old,new):
        if isinstance(value,dict):return {k:TrainViewCache._relocate(v,old,new) for k,v in value.items()}
        if isinstance(value,list):return [TrainViewCache._relocate(v,old,new) for v in value]
        if isinstance(value,str):
            try:return str(new/Path(value).relative_to(old))
            except ValueError:return value
        return value

    @staticmethod
    def _load_disk(folder):
        record=folder/'result.json'
        if not record.is_file():return None
        try:
            saved=json.loads(record.read_text())
            if saved.get('contract')!=DISK_CACHE_CONTRACT:return None
            result=saved['result']
            assets=saved.get('assets')
            if not assets:return None
            for path,sha in assets.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:return None
            for key in ('target_html','target_image','image','current_html','current_image'):
                if key in result and not Path(result[key]).is_file():return None
            return result
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):return None

    def get(self, row, policy=False):
        started=time.perf_counter()
        if self.cache_dir is None:
            result=self._get(row,policy)
            self.last_profile['total_s']=time.perf_counter()-started
            return result
        key=self._key(row,policy)
        lock_started=time.perf_counter()
        with (self.cache_dir/f'{key}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                lock_wait_s=time.perf_counter()-lock_started
                import_started=time.perf_counter()
                if self.legacy_cache_dir and self.legacy_cache_dir!=self.cache_dir:
                    self._import_legacy(key)
                legacy_import_s=time.perf_counter()-import_started
                # A killed writer cannot run its finally block. With this key
                # locked, its old staging directories are safe to reclaim.
                for stale in self.cache_dir.glob(f'.{key}.*'):
                    if stale.is_dir() and not stale.is_symlink():shutil.rmtree(stale)
                result=self._get(row,policy)
                self.last_profile.update(lock_wait_s=lock_wait_s,legacy_import_s=legacy_import_s,
                                         total_s=time.perf_counter()-started)
                return result
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def _import_legacy(self,key):
        """Import a verified completed entry under the destination key lock."""
        destination=self.cache_dir/key
        if destination.exists():return
        source=self.legacy_cache_dir/key
        if self._load_disk(source) is None:return
        stage=Path(tempfile.mkdtemp(prefix=f'.{key}.',dir=self.cache_dir))
        def relocate(value):
            if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
            if isinstance(value,list):return [relocate(v) for v in value]
            return self._relocate(value,source,destination)
        try:
            # Non-JSON files are immutable committed artifacts. Metadata must
            # be separate because its paths change; never edit linked JSON.
            for path in source.rglob('*'):
                target=stage/path.relative_to(source)
                if path.is_symlink():raise ValueError(f'Symlink in legacy cache: {path}')
                if path.is_dir():target.mkdir(parents=True,exist_ok=True)
                elif path.is_file():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if path.suffix=='.json':
                        target.write_text(json.dumps(relocate(json.loads(path.read_text()))))
                    else:os.link(path,target)
            os.rename(stage,destination)
            if self._load_disk(destination) is None:
                shutil.rmtree(destination)
                raise ValueError('Imported shared cache failed validation')
        finally:
            if stage.exists():shutil.rmtree(stage)

    def _get(self, row, policy=False):
        from .visual_data import _render_page
        from .tree_normalize import _prepare_page
        from .web_experiment import digest
        source=row['lazy_source'];settings=row['lazy_settings']
        profile={'status':'unknown'}
        self.last_profile=profile
        phase_started=time.perf_counter()
        if row.get('lazy_contract')!=CONTRACT:raise ValueError('Unknown lazy train contract')
        if source['split']!='train':raise ValueError('Only train views may be lazy')
        if digest(source['html'])!=source['source_sha']:raise ValueError('Lazy source HTML changed')
        profile['source_hash_s']=time.perf_counter()-phase_started
        key=self._key(row,policy)
        if key in self.entries and self.cache_dir is None:
            profile['status']='memory_hit'
            self.entries.move_to_end(key);return copy.deepcopy(self.entries[key][1])
        disk=self.cache_dir/key if self.cache_dir is not None else None
        if disk is not None:
            phase_started=time.perf_counter()
            result=self._load_disk(disk)
            profile['disk_validation_s']=time.perf_counter()-phase_started
            if result is not None:
                profile['status']='disk_hit'
                self.entries[key]=(None,result)
                while len(self.entries)>self.capacity:self.entries.popitem(last=False)
                return copy.deepcopy(result)
            # Under the per-entry lock, discard only this invalid generated
            # entry. A new successful commit must not lose to corrupt data.
            if disk.exists():shutil.rmtree(disk)
        profile['status']='disk_miss' if disk is not None else 'memory_miss'
        while len(self.entries)>=self.capacity:
            _,(directory,_)=self.entries.popitem(last=False)
            if directory is not None:directory.cleanup()
        directory=None
        if disk is None:
            directory=tempfile.TemporaryDirectory(prefix='tuide-train-view-')
            work=Path(directory.name)
        else:
            work=Path(tempfile.mkdtemp(prefix=f'.{key}.',dir=self.cache_dir))
        args=SimpleNamespace(**settings)
        # Old descriptors keep their original owner-box supervision.
        if not hasattr(args,'visual_geometry_contract'):args.visual_geometry_contract=None
        if not hasattr(args,'detector_label_contract'):args.detector_label_contract=None
        index=int(hashlib.sha256(source['id'].encode()).hexdigest()[:8],16)
        try:
            relabel=settings.get('policy_relabel')
            if relabel and policy:
                phase_started=time.perf_counter()
                from .policy_relabel import refresh_row
                previous=TrainViewCache(cache_dir=relabel['cache_dir'])
                try:
                    original=previous.get(relabel['source'],policy=True)
                    profile['parent_cache_status']=previous.last_profile['status']
                    profile['parent_cache_s']=previous.last_profile['total_s']
                    profile['parent_cache_profile']=copy.deepcopy(previous.last_profile)
                    refresh_started=time.perf_counter()
                    result=refresh_row(original,work/'visual-labels',relabel['size'],clean_only=True)
                    profile['label_refresh_s']=time.perf_counter()-refresh_started
                finally:previous.close()
                profile['relabel_s']=time.perf_counter()-phase_started
            else:
                phase_started=time.perf_counter()
                rendered,error,_=_render_page((args,work/'rendered',index,source))
                if error:raise ValueError(f'Lazy render failed: {error}')
                profile['render_s']=time.perf_counter()-phase_started
            if policy and not relabel:
                phase_started=time.perf_counter()
                seed,page,error=_prepare_page((args,work/'labels','train',index,source['id'],
                    rendered['policy'][0],{source['id']:rendered['page']}),
                    persist_seed=os.environ.get('TUIDE_SAVE_DIAGNOSTICS','0')=='1')
                if error:raise ValueError(f'Lazy normalization failed: {error}')
                result=seed
                profile['normalize_s']=time.perf_counter()-phase_started
            elif not policy:result=rendered['detection'][0]
            phase_started=time.perf_counter()
            if os.environ.get('TUIDE_SAVE_DIAGNOSTICS','0')!='1':
                if policy:
                    # OnlineSampler starts from target_html and generates its
                    # own current state. Seed RGB/HTML are not training inputs.
                    for field in ('current_html','current_image','current_abstract',
                                  'initial_html','initial_image','current_html_text'):
                        result.pop(field,None)
                compact_new_cache(work,result)
            if disk is not None:
                relocated=self._relocate(result,work,disk)
                assets={self._relocate(result[k],work,disk):
                        hashlib.sha256(Path(result[k]).read_bytes()).hexdigest()
                        for k in ('target_html','target_image','image','current_html','current_image') if k in result}
                (work/'result.json').write_text(json.dumps(
                    {'contract':DISK_CACHE_CONTRACT,'result':relocated,'assets':assets},ensure_ascii=False))
                os.rename(work,disk)
                result=self._load_disk(disk)
                if result is None:raise RuntimeError('Concurrent lazy train-view cache commit failed')
            self.entries[key]=(directory,result)
            profile['commit_s']=time.perf_counter()-phase_started
            return copy.deepcopy(result)
        except BaseException:
            if directory is not None:directory.cleanup()
            else:shutil.rmtree(work,ignore_errors=True)
            raise


def portable_images(row, keys):
    """Queued batches must not depend on files later evicted by a worker."""
    result=dict(row)
    for key in keys:
        if key in result and isinstance(result[key],str):result[key]=Path(result[key]).read_bytes()
    return result

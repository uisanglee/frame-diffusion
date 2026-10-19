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
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
from importlib.metadata import version

CONTRACT = 'lazy-webui-train-view-v1'
DISK_CACHE_CONTRACT = 'lazy-webui-train-view-cache-v2-verified'


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
    return {**{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
            'online_preparation':hashlib.sha256(''.join(ast.dump(n) for n in semantic).encode()).hexdigest(),
            'playwright':version('playwright')}


class TrainViewCache:
    def __init__(self, capacity=2, cache_dir=None):
        if capacity < 1: raise ValueError('Cache capacity must be positive')
        self.capacity=capacity;self.entries=OrderedDict()
        self.cache_dir=Path(cache_dir).resolve() if cache_dir else None
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
        if self.cache_dir is None:return self._get(row,policy)
        key=self._key(row,policy)
        with (self.cache_dir/f'{key}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                # A killed writer cannot run its finally block. With this key
                # locked, its old staging directories are safe to reclaim.
                for stale in self.cache_dir.glob(f'.{key}.*'):
                    if stale.is_dir() and not stale.is_symlink():shutil.rmtree(stale)
                return self._get(row,policy)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def _get(self, row, policy=False):
        from .visual_data import _render_page
        from .tree_normalize import _prepare_page
        from .web_experiment import digest
        source=row['lazy_source'];settings=row['lazy_settings']
        if row.get('lazy_contract')!=CONTRACT:raise ValueError('Unknown lazy train contract')
        if source['split']!='train':raise ValueError('Only train views may be lazy')
        if digest(source['html'])!=source['source_sha']:raise ValueError('Lazy source HTML changed')
        key=self._key(row,policy)
        if key in self.entries and self.cache_dir is None:
            self.entries.move_to_end(key);return copy.deepcopy(self.entries[key][1])
        disk=self.cache_dir/key if self.cache_dir is not None else None
        if disk is not None:
            result=self._load_disk(disk)
            if result is not None:
                self.entries[key]=(None,result)
                while len(self.entries)>self.capacity:self.entries.popitem(last=False)
                return copy.deepcopy(result)
            # Under the per-entry lock, discard only this invalid generated
            # entry. A new successful commit must not lose to corrupt data.
            if disk.exists():shutil.rmtree(disk)
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
        index=int(hashlib.sha256(source['id'].encode()).hexdigest()[:8],16)
        try:
            rendered,error,_=_render_page((args,work/'rendered',index,source))
            if error:raise ValueError(f'Lazy render failed: {error}')
            if policy:
                seed,page,error=_prepare_page((args,work/'labels','train',index,source['id'],
                    rendered['policy'][0],{source['id']:rendered['page']}))
                if error:raise ValueError(f'Lazy normalization failed: {error}')
                result=seed
            else:result=rendered['detection'][0]
            if disk is not None:
                relocated=self._relocate(result,work,disk)
                assets={str(disk/Path(result[k]).relative_to(work)):
                        hashlib.sha256(Path(result[k]).read_bytes()).hexdigest()
                        for k in ('target_html','target_image','image','current_html','current_image') if k in result}
                (work/'result.json').write_text(json.dumps(
                    {'contract':DISK_CACHE_CONTRACT,'result':relocated,'assets':assets},ensure_ascii=False))
                os.rename(work,disk)
                result=self._load_disk(disk)
                if result is None:raise RuntimeError('Concurrent lazy train-view cache commit failed')
            self.entries[key]=(directory,result)
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

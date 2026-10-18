"""Bounded train-view materialization with optional persistent on-demand cache.

Only generated temporary files are evicted. Raw HTML and prepared evaluation
assets are never modified. Persistent entries are content/settings addressed
and atomically committed so independent online workers can safely share them.
"""
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

CONTRACT = 'lazy-webui-train-view-v1'
DISK_CACHE_CONTRACT = 'lazy-webui-train-view-cache-v1'


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
                 repr(sorted(settings.items())),policy)
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
            for key in ('target_html','target_image'):
                if key in result and not Path(result[key]).is_file():return None
            return result
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):return None

    def get(self, row, policy=False):
        from .visual_data import _render_page
        from .tree_normalize import _prepare_page
        from .web_experiment import digest
        source=row['lazy_source'];settings=row['lazy_settings']
        if row.get('lazy_contract')!=CONTRACT:raise ValueError('Unknown lazy train contract')
        if source['split']!='train':raise ValueError('Only train views may be lazy')
        if digest(source['html'])!=source['source_sha']:raise ValueError('Lazy source HTML changed')
        key=self._key(row,policy)
        if key in self.entries:
            self.entries.move_to_end(key);return copy.deepcopy(self.entries[key][1])
        disk=self.cache_dir/key if self.cache_dir is not None else None
        if disk is not None:
            result=self._load_disk(disk)
            if result is not None:
                self.entries[key]=(None,result)
                while len(self.entries)>self.capacity:self.entries.popitem(last=False)
                return copy.deepcopy(result)
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
                (work/'result.json').write_text(json.dumps(
                    {'contract':DISK_CACHE_CONTRACT,'result':relocated},ensure_ascii=False))
                try:os.rename(work,disk)
                except FileExistsError:shutil.rmtree(work,ignore_errors=True)
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

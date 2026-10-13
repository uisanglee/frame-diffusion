"""Bounded, process-local train-view materialization; held-out data stays fixed.

Only generated temporary files are evicted. Raw HTML and prepared evaluation
assets are never modified. Temporary caches are rebuilt after process restart.
"""
import copy
import hashlib
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

CONTRACT = 'lazy-webui-train-view-v1'


class TrainViewCache:
    def __init__(self, capacity=2):
        if capacity < 1: raise ValueError('Cache capacity must be positive')
        self.capacity=capacity;self.entries=OrderedDict()

    def close(self):
        for directory,_ in self.entries.values():directory.cleanup()
        self.entries.clear()

    def get(self, row, policy=False):
        from .visual_data import _render_page
        from .tree_normalize import _prepare_page
        from .web_experiment import digest
        source=row['lazy_source'];settings=row['lazy_settings']
        if row.get('lazy_contract')!=CONTRACT:raise ValueError('Unknown lazy train contract')
        if source['split']!='train':raise ValueError('Only train views may be lazy')
        if digest(source['html'])!=source['source_sha']:raise ValueError('Lazy source HTML changed')
        key=(source['id'],source['source_sha'],repr(sorted(settings.items())),policy)
        if key in self.entries:
            self.entries.move_to_end(key);return copy.deepcopy(self.entries[key][1])
        while len(self.entries)>=self.capacity:
            _,(directory,_)=self.entries.popitem(last=False);directory.cleanup()
        directory=tempfile.TemporaryDirectory(prefix='tuide-train-view-')
        work=Path(directory.name)
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
            self.entries[key]=(directory,result)
            return copy.deepcopy(result)
        except BaseException:
            directory.cleanup();raise


def portable_images(row, keys):
    """Queued batches must not depend on files later evicted by a worker."""
    result=dict(row)
    for key in keys:
        if key in result and isinstance(result[key],str):result[key]=Path(result[key]).read_bytes()
    return result

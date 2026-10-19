"""Bounded CPU render prefetch; commit RNG only for consumed detector batches."""
import json
import multiprocessing as mp
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.util import Finalize
from pathlib import Path


_cache=None
_pools=None


def initialize(pools,cache_dir):
    global _cache,_pools
    import torch
    from .browser import enable_worker_browser,close_worker_browser
    from .lazy_webui import TrainViewCache
    torch.set_num_threads(1)
    enable_worker_browser()
    _cache=TrainViewCache(cache_dir=cache_dir);_pools=pools
    Finalize(None,_cache.close,exitpriority=10)
    Finalize(None,close_worker_browser,exitpriority=9)


def materialize(task):
    rows,seeds=task;result=[];errors=[]
    from .lazy_webui import portable_images
    from .visual_train import sample_page_views
    for row,seed in zip(rows,seeds):
        rng=random.Random(seed)
        for _ in range(32):
            try:
                result.append(portable_images(_cache.get(row),('image',)) if row.get('lazy_contract') else row)
                break
            except ValueError as exc:
                errors.append({'id':row['id'],'error':str(exc)})
                row=sample_page_views(_pools,1,rng)[0]
        else:raise RuntimeError(f'32 lazy detector preparation failures: {errors}')
    return result,errors


class DetectorStream:
    def __init__(self,pools,rng,batch_size,cache_dir,log,workers=2,prefetch=4,timeout=900):
        if min(workers,prefetch,batch_size,timeout)<=0:raise ValueError('Invalid detector prefetch settings')
        self.pools=pools;self.rng=rng;self.planner=random.Random();self.planner.setstate(rng.getstate())
        self.batch_size=batch_size;self.cache_dir=str(cache_dir);self.log=Path(log)
        self.workers=workers;self.prefetch=prefetch;self.timeout=timeout;self.pending=deque();self.pool=None

    def __enter__(self):
        self.pool=ProcessPoolExecutor(self.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(self.pools,self.cache_dir))
        self.fill();return self

    def fill(self):
        from .visual_train import sample_page_views
        while len(self.pending)<self.prefetch:
            rows=sample_page_views(self.pools,self.batch_size,self.planner)
            seeds=[self.planner.getrandbits(64) for _ in rows]
            self.pending.append((self.pool.submit(materialize,(rows,seeds)),self.planner.getstate()))

    def take(self):
        future,state=self.pending[0]
        rows,errors=future.result(timeout=self.timeout)
        self.pending.popleft();self.rng.setstate(state)
        if errors:
            with self.log.open('a') as file:
                for error in errors:file.write(json.dumps(error)+'\n')
        self.fill();return rows

    def __exit__(self,*args):
        if self.pool is not None:
            from .worker_cleanup import stop_executor
            stop_executor(self.pool,list((getattr(self.pool,'_processes',None) or {}).values()))
            self.pool=None

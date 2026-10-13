"""Bounded page workers and atomic preparation checkpoints."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from pathlib import Path
from multiprocessing.util import Finalize

from .ir import write_json as _write_json, write_jsonl as _write_jsonl


def _initialize_browser_worker():
    from .browser import enable_worker_browser, close_worker_browser
    enable_worker_browser()
    Finalize(None, close_worker_browser, exitpriority=20)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    _write_json(temporary, value)
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    _write_jsonl(temporary, rows)
    temporary.replace(path)


def page_map(function, tasks, workers):
    """Keep input order/seeds stable and at most 2*workers jobs in flight."""
    if workers < 1:
        raise ValueError('workers must be positive')
    if workers == 1:
        from .browser import enable_worker_browser, close_worker_browser
        enable_worker_browser()
        try:
            yield from map(function, tasks)
        finally:
            close_worker_browser()
        return
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_initialize_browser_worker)
    pending = deque()
    tasks = iter(tasks)
    try:
        for _ in range(2 * workers):
            task = next(tasks, None)
            if task is None:
                break
            pending.append(pool.submit(function, task))
        while pending:
            yield pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(function, task))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)

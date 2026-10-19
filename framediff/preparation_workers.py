"""Bounded page workers, atomic checkpoints and scoped process cleanup."""
import multiprocessing
import os
import signal
import subprocess
import time
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


def process_table():
    output=subprocess.check_output(['ps','-axo','pid=,ppid=,lstart=,command='],text=True)
    table={}
    for line in output.splitlines():
        parts=line.split(None,7)
        if len(parts)==8:table[int(parts[0])]=(int(parts[1]),' '.join(parts[2:]))
    return table


def stop_executor(pool,processes):
    # Capture ancestry before any worker dies and its children are reparented.
    # Freeze workers first so queued work cannot launch a new browser while
    # the captured descendants are being reaped.
    try:
        for process in processes:
            if process.is_alive():
                try:os.kill(process.pid,signal.SIGSTOP)
                except ProcessLookupError:pass
        _stop_descendants(processes)
    finally:
        pool.shutdown(wait=False,cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
                try:os.kill(process.pid,signal.SIGCONT)
                except ProcessLookupError:pass
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():process.kill();process.join(timeout=2)


def _stop_descendants(processes):
    table=process_table();roots={p.pid for p in processes if p.pid and p.is_alive()}
    descendants=set();front=roots
    while front:
        front={pid for pid,(parent,_) in table.items() if parent in front}-descendants-roots
        descendants.update(front)
    tracked={pid:table[pid] for pid in descendants}
    def alive():
        current=process_table()
        return {pid for pid,identity in tracked.items() if pid in current and current[pid][1]==identity[1]}
    def send(pids,sig):
        for pid in pids:
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
    send(alive(),signal.SIGTERM)
    deadline=time.monotonic()+2
    while alive() and time.monotonic()<deadline:time.sleep(.05)
    send(alive(),signal.SIGKILL)
    # Give driver/browser parents time to reap their subprocesses.
    deadline=time.monotonic()+2
    while alive() and time.monotonic()<deadline:time.sleep(.05)
    remaining=alive()
    if remaining:raise RuntimeError(f'Worker descendants did not exit: {sorted(remaining)}')

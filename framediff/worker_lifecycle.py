"""A lightweight owner process keeps crashed workers' descendants reclaimable."""
import ctypes
import multiprocessing as mp
import os
import signal
import sys
import time


def _reap(pids):
    for pid in pids:
        try:os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:pass


def _cleanup_children(child):
    from .preparation_workers import process_table
    tracked = {}
    def remaining():
        child.join(timeout=0)
        table = process_table()
        front = {os.getpid()}; found = set()
        while front:
            front = {pid for pid, (parent, _) in table.items() if parent in front} - found
            found.update(front)
        for pid in found:tracked[pid] = table[pid][1]
        # multiprocessing owns the direct worker's wait status. Reap only
        # adopted grandchildren here, so Process.join retains its exit code.
        _reap(set(tracked) - {child.pid})
        table = process_table()
        return {pid for pid, identity in tracked.items()
                if pid in table and table[pid][1] == identity}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + 2
        while True:
            live = remaining()
            if not live:return
            for pid in live:
                try:os.kill(pid, sig)
                except ProcessLookupError:pass
            if time.monotonic() >= deadline:break
            time.sleep(.05)
    if remaining():raise RuntimeError('Online worker descendants did not exit')


def supervise(connection, stop):
    # Linux reparents even detached Chromium processes to this dedicated owner,
    # rather than PID 1, when the rendering worker crashes or is OOM-killed.
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), 'Cannot enable worker subreaper')
    from .online_workers import _run
    parent = os.getppid()
    child = mp.get_context('spawn').Process(target=_run, args=(connection, stop))
    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        child.start()
        connection.close()
        while child.is_alive():
            child.join(timeout=.1)
            if os.getppid() != parent:break
    finally:
        # Ignore repeated requests while reclaiming descendants.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        connection.close()
        if child.pid is not None:_cleanup_children(child)
        if child.pid is not None:child.join(timeout=1)

import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from framediff.online_workers import OnlineWorkerPool


def _partial_result():
    from framediff import cache_lock
    callback=cache_lock._progress
    connection=dict(zip(callback.__code__.co_freevars,
                        [cell.cell_contents for cell in callback.__closure__]))['connection']
    os.write(connection.fileno(),struct.pack('!i',1000000)+b'x')
    time.sleep(60)


def _value():return 42


def test_partial_response_still_times_out_and_restarts():
    pool=OnlineWorkerPool(1)
    try:
        pool.submit(_value).result(timeout=20)  # exclude process startup
        job=pool.submit(_partial_result)
        started=time.monotonic()
        with pytest.raises(TimeoutError):job.result(timeout=.3)
        assert time.monotonic()-started < 3
        old=pool.slots[0]
        pool.restart(job,retry=False)
        assert not old.process.is_alive()
        assert not old.reader.is_alive()
        assert pool.submit(_value).result(timeout=20)==42
    finally:pool.shutdown()


def _crash_with_detached_child(marker):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],
                           start_new_session=True,stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    Path(marker).write_text(str(child.pid))
    os._exit(7)


@pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux child subreaper integration')
def test_crashed_worker_detached_child_reaped_without_touching_other_jobs(tmp_path):
    from framediff.preparation_workers import process_table
    pool=OnlineWorkerPool(2);child=None
    try:
        marker=tmp_path/'child'
        bad=pool.submit(_crash_with_detached_child,str(marker))
        healthy=pool.submit(_value)
        with pytest.raises(RuntimeError,match='exited unexpectedly'):bad.result(timeout=20)
        assert healthy.result(timeout=20)==42
        child=int(marker.read_text())
        deadline=time.monotonic()+6
        while child in process_table() and time.monotonic()<deadline:time.sleep(.05)
        assert child not in process_table()
        assert pool.submit(_value).result(timeout=20)==42
    finally:
        pool.shutdown()
        if child:
            try:os.kill(child,signal.SIGKILL)
            except ProcessLookupError:pass


def test_cleanup_tracks_adopted_children(monkeypatch):
    # A dead worker's detached child is now owned by the supervisor on Linux.
    from framediff import worker_lifecycle as lifecycle, preparation_workers
    from types import SimpleNamespace
    root=os.getpid();live={123456789:(root,'identity')};signals=[]
    monkeypatch.setattr(lifecycle,'_reap',lambda pids:None)
    monkeypatch.setattr(preparation_workers,'process_table',lambda:dict(live))
    def kill(pid,sig):signals.append((pid,sig));live.pop(pid)
    monkeypatch.setattr(lifecycle.os,'kill',kill)
    lifecycle._cleanup_children(SimpleNamespace(pid=123456788,join=lambda **kw:None))
    assert signals==[(123456789,signal.SIGTERM)]

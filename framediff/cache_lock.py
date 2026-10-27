"""Process-local cache wait notifications for the online worker watchdog."""
import contextlib
import fcntl
import time
import hashlib
from functools import lru_cache
from pathlib import Path

_progress = None


def set_progress(callback):
    global _progress
    _progress = callback


def report_online_target(target):
    if _progress: _progress('target', target)


@contextlib.contextmanager
def cache_lock(file):
    waiting = False
    try:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            waiting = True
            if _progress: _progress('cache_wait', str(file.name))
            while True:
                try:
                    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.1)
        if waiting and _progress: _progress('working', str(file.name))
        waiting = False
        yield
    finally:
        if waiting and _progress: _progress('working', str(file.name))
        fcntl.flock(file, fcntl.LOCK_UN)


class UnusableTrainView(ValueError):
    """Versioned, deterministic eligibility failure; transient errors never use this."""


@lru_cache(maxsize=1)
def observation_revision():
    sha=hashlib.sha256(Path(__file__).with_name('online_observation.py').read_bytes()).hexdigest()
    # This snapshot is browser-tested against the previous multi-call collector.
    # Unknown future collector changes must invalidate observations and witnesses.
    return None if sha in (
        '2459133082882061b46ded9a96109d5745a9c3f1ceaae92cef69a57c32913120',
        # Per-observation rectangle/ancestor clip memoization; browser parity
        # includes CSS reflow, overflow changes, and scrolling between calls.
        '9eca87cdbfe7126a6648da73a8c17b6ce88914b462687a228218d50536f84867',
    ) else sha


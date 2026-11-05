"""Cheap, coordinator-only progress reporting; never inspect prepared assets."""
import math
import time
from pathlib import Path

from .preparation_workers import write_json


def _duration(seconds):
    if seconds is None:
        return '--'
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class PreparationProgress:
    """Report completed views, with an approximate ETA at completion boundaries.

    The caller supplies all counters. No background thread, filesystem scan,
    browser work or dataset read is needed. Fast cache hits and slow new work
    share the observed mean, so ETA is an estimate for the current stage only.
    """

    def __init__(self, stage, total, path, *, every=50, interval=30, unit='views'):
        if type(total) is not int or total < 0:
            raise ValueError('total must be a nonnegative integer')
        if type(every) is not int or every < 1:
            raise ValueError('every must be a positive integer')
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('interval must be finite and positive')
        self.stage = stage
        self.total = total
        self.path = Path(path)
        self.every = every
        self.interval = interval
        self.unit = unit
        self.started = time.monotonic()
        self.last_emit = None
        self.last_emitted_count = 0
        self.last_processed = 0

    def update(self, processed, force=False, **details):
        """Write/print a snapshot when due; return it, or None when throttled."""
        if type(processed) is not int or not 0 <= processed <= self.total:
            raise ValueError('processed must be an integer between zero and total')
        if processed < self.last_processed:
            raise ValueError('processed must not decrease')
        now = time.monotonic()
        self.last_processed = processed
        if not (force or self.last_emit is None or processed == self.total or
                processed - self.last_emitted_count >= self.every or
                now - self.last_emit >= self.interval):
            return None
        elapsed = max(0., now - self.started)
        eta = (0. if processed == self.total else
               elapsed * (self.total - processed) / processed if processed else None)
        snapshot = {**details, 'stage': self.stage, 'unit': self.unit,
                    'processed': processed, 'total': self.total,
                    'percent': round(100. * processed / self.total, 2) if self.total else 100.,
                    'elapsed_s': round(elapsed, 2), 'eta_s': round(eta, 2) if eta is not None else None}
        write_json(self.path, snapshot)
        counters = ' '.join(f'{key}={details[key]}' for key in
                            ('kept', 'pages', 'failures', 'timeouts', 'cache_hits', 'workers', 'round') if key in details)
        prefix = f'[prepare:{self.stage}] {processed}/{self.total} {self.unit} ({snapshot["percent"]:.1f}%)'
        print(f'{prefix} {counters} elapsed={_duration(elapsed)} eta={_duration(eta)}', flush=True)
        self.last_emit = now
        self.last_emitted_count = processed
        return snapshot

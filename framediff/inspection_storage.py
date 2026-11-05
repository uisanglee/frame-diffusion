"""Compact training reads of immutable candidate-inspection caches.

Keep existing detailed JSON intact for audits. A small derived sidecar avoids
parsing every probe on subsequent training runs, while source stat changes
always invalidate it. The caller holds the inspection entry lock.
"""
import json
from pathlib import Path

from .preparation_workers import write_json


def read_inspection(path, detailed=False):
    path = Path(path)
    if detailed:
        return json.loads(path.read_text())
    stat = path.stat()
    identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    compact = path.parent / '.training' / path.name
    try:
        cached = json.loads(compact.read_text())
        if cached['source_stat'] == identity:
            return cached['value']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    value = json.loads(path.read_text())
    value.pop('inspection_diagnostics', None)
    # This is optional acceleration: a read-only shared cache remains usable.
    try:
        write_json(compact, dict(source_stat=identity, value=value))
    except OSError:
        pass
    return value

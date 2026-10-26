"""Shared, persistent exclusions for views that repeatedly stall online workers."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


class TimeoutExclusions:
    def __init__(self, targets):
        from .lazy_webui import preparation_fingerprint
        self.fingerprint = preparation_fingerprint()
        legacy = next((t['lazy_cache_dir'] for t in targets if t.get('lazy_cache_dir')), None)
        root = os.environ.get('TUIDE_SHARED_CACHE')
        self.directory = ((Path(root) if root else Path(legacy).parent if legacy
                           else Path('data/webui-shared-cache')) / 'online-timeout-exclusions')

    def describe(self, target):
        source = target.get('lazy_source', target)
        identity = dict(contract='online-timeout-view-v1', page=source['id'],
                        source_sha=source.get('source_sha'), viewport=target.get('viewport'),
                        settings=target.get('lazy_settings'), code=self.fingerprint)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return dict(key=key, page=source['id'], view=target.get('view', 'unknown'))

    def contains(self, target):
        return (self.directory / f"{target['key']}.json").is_file()

    def exclude(self, target, timeout):
        self.directory.mkdir(parents=True, exist_ok=True)
        record = dict(target, reason='repeated_active_timeout', timeout_s=timeout)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, suffix='.tmp', delete=False) as f:
                temporary = Path(f.name)
                json.dump(record, f)
            temporary.replace(self.directory / f"{target['key']}.json")
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)

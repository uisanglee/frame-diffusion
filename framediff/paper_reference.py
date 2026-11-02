"""Locked target observations shared by independently launched initial-seed jobs."""
import fcntl
import hashlib
import json
from pathlib import Path


def shared_reference(browser, html, viewport, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {'html': html, 'viewport': viewport, 'max_nodes': 2048,
                'contract': 'paper-reference-cache-v1'}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = root/key
    with (root/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        folder.mkdir(exist_ok=True)
        png, dom, complete = folder/'A_rendering.png', folder/'A_rendering-dom.json', folder/'complete.json'
        def hashes():
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (png, dom)}
        if complete.exists():
            if hashes() != json.loads(complete.read_text()):
                raise ValueError(f'Shared reference cache changed: {folder}; do not mix evaluation targets')
        else:
            observed = browser.snapshot(html, viewport, png, 2048)
            dom.write_text(json.dumps({'viewport': viewport, 'nodes': observed['nodes'],
                'total_visible_nodes': observed['total_visible_nodes'], 'max_nodes': 2048}))
            pending = folder/'complete.tmp'
            pending.write_text(json.dumps(hashes()))
            pending.replace(complete)
        return str(png), str(dom)

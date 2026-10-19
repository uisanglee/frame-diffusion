"""Audit or hardlink byte-identical assets of completed TUIDE caches.

Stop all writers before --apply. Existing paths/JSON are preserved. Only files
listed with verified SHA256 in successful cache records are eligible. Committed
assets must remain immutable; regenerate by removing the cache entry, never by
overwriting its HTML/images in place. Never delete cache roots after this tool.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(roots, store, apply=False):
    groups=defaultdict(dict);errors=[];records=0
    for root in roots:
        for name in ('record.json','result.json'):
            for record in root.rglob(name):
                try:
                    value=json.loads(record.read_text())
                    assets=value.get('assets')
                    if not isinstance(assets,dict):continue
                    # Only successful normalized/lazy completion records.
                    if not (('row' in value and 'page' in value) or
                            value.get('contract')=='lazy-webui-train-view-cache-v2-verified'):continue
                    records+=1
                    for filename,sha in assets.items():
                        path=Path(filename)
                        if not path.is_absolute() or not any(path.is_relative_to(r) for r in roots):continue
                        if path.suffix.lower() not in ('.html','.png','.jpg','.jpeg','.webp'):continue
                        if path.is_symlink() or not path.is_file():raise ValueError(f'Missing/symlink asset: {path}')
                        if digest(path)!=sha:raise ValueError(f'Checksum mismatch: {path}')
                        groups[sha][str(path)]=path.stat()
                except (OSError,ValueError,TypeError) as exc:
                    errors.append({'record':str(record),'error':str(exc)})
    report={'records':records,'errors':errors,'duplicate_groups':[], 'estimated_reclaimable_bytes':0,
            'applied':False}
    for sha,files in groups.items():
        if len(files)<2:continue
        inodes={(s.st_dev,s.st_ino) for s in files.values()}
        size=next(iter(files.values())).st_size
        report['duplicate_groups'].append({'sha256':sha,'files':list(files),'unique_inodes':len(inodes),'size':size})
        report['estimated_reclaimable_bytes']+=max(0,len(inodes)-1)*size
    if apply:
        if errors:raise ValueError(f'Cache audit has {len(errors)} errors; run without --apply and inspect report')
        store.mkdir(parents=True,exist_ok=True)
        # Preflight before changing any paths. Never silently copy across disks.
        device=store.stat().st_dev
        for group in report['duplicate_groups']:
            for name in group['files']:
                if Path(name).stat().st_dev!=device:raise ValueError('Store and assets must be on the same filesystem')
        with (store/'dedup.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for group in report['duplicate_groups']:
                sha=group['sha256'];paths=[Path(p) for p in group['files']]
                blob=store/sha
                if blob.exists():
                    if blob.is_symlink() or digest(blob)!=sha:raise ValueError(f'Corrupt common blob: {blob}')
                else:os.link(paths[0],blob)
                for path in paths:
                    if digest(path)!=sha:raise ValueError(f'Asset changed during cleanup: {path}')
                    if os.path.samefile(path,blob):continue
                    with tempfile.TemporaryDirectory(prefix='.dedup-',dir=path.parent) as staging:
                        linked=Path(staging)/'asset';os.link(blob,linked);os.replace(linked,path)
            report['applied']=True
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--store',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    roots=[Path(p).resolve() for p in args.roots]
    if any(not p.is_dir() for p in roots):parser.error('All roots must be existing cache directories')
    report=run(roots,Path(args.store).resolve(),args.apply)
    target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='duplicate_groups'},ensure_ascii=False))
    print(f'Duplicate groups: {len(report["duplicate_groups"])}. Report: {target}')


if __name__=='__main__':main()

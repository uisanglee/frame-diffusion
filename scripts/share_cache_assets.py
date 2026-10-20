"""Audit or hardlink byte-identical assets of completed TUIDE caches.

Stop all writers before --apply. Existing paths/JSON are preserved. By default,
only files listed with verified SHA256 in successful cache records are eligible.
--include-lazy-intermediates also hashes rendered assets in completed lazy caches.
Committed
assets must remain immutable; regenerate by removing the cache entry, never by
overwriting its HTML/images in place. Never delete cache roots after this tool.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import time


# Recognition for storage auditing only, not permission to load old training
# caches as current ones. Every eligible asset still requires its saved hash.
CACHE_CONTRACTS = {'lazy-webui-train-view-cache-v1',
                   'lazy-webui-train-view-cache-v2-verified'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def committed_asset(record, path):
    """Resolve only a nested record's own staging path after atomic commit.

    Historical lazy commits relocated result.json, not nested record.json.
    Never search other entries or infer identity from the basename alone.
    The caller must still check the recorded content hash.
    """
    if path.exists() or path.is_symlink():return path
    for entry in record.parents:
        if not re.fullmatch(r'[0-9a-f]{64}',entry.name):continue
        completion=entry/'result.json'
        if not completion.is_file():continue
        if json.loads(completion.read_text()).get('contract') not in CACHE_CONTRACTS:continue
        for index,part in enumerate(path.parts):
            if not re.fullmatch(r'\.'+entry.name+r'\.[A-Za-z0-9_-]+',part):continue
            suffix=Path(*path.parts[index+1:])
            if '..' in suffix.parts:continue
            candidate=entry/suffix
            if candidate.parent==record.parent and candidate.resolve().is_relative_to(entry.resolve()):
                return candidate
    return path


def run(roots, store, apply=False, include_lazy_intermediates=False):
    groups=defaultdict(dict);errors=[];records=0;relocated=[]
    for root in roots:
        for name in ('record.json','result.json'):
            for record in root.rglob(name):
                try:
                    value=json.loads(record.read_text())
                    assets=value.get('assets')
                    if not isinstance(assets,dict):continue
                    # Only successful normalized/lazy completion records.
                    if not (('row' in value and 'page' in value) or
                            value.get('contract') in CACHE_CONTRACTS):continue
                    records+=1
                    for filename,sha in assets.items():
                        path=Path(filename)
                        original=path
                        path=committed_asset(record,path)
                        if not path.is_absolute() or not any(path.is_relative_to(r) for r in roots):continue
                        if path.suffix.lower() not in ('.html','.png','.jpg','.jpeg','.webp'):continue
                        if path.is_symlink() or not path.is_file():raise ValueError(f'Missing/symlink asset: {path}')
                        if digest(path)!=sha:raise ValueError(f'Checksum mismatch: {path}')
                        if path!=original:
                            relocated.append({'record':str(record),'from':str(original),'to':str(path)})
                        groups[sha][str(path)]=path.stat()
                    if (include_lazy_intermediates and record.name=='result.json'
                        and value.get('contract') in CACHE_CONTRACTS
                        and record.parent.parent.name=='lazy-train-views'
                        and re.fullmatch(r'[0-9a-f]{64}',record.parent.name)):
                        # Older successful entries retain unindexed rendering
                        # intermediates. Link equal bytes, preserving every path.
                        for path in (record.parent/'rendered').rglob('*'):
                            if path.suffix.lower() not in ('.html','.png','.jpg','.jpeg','.webp'):continue
                            if path.is_symlink() or not path.resolve().is_relative_to(record.parent.resolve()):
                                raise ValueError(f'Unsafe intermediate asset: {path}')
                            if path.is_file():groups[digest(path)][str(path)]=path.stat()
                    if records%100==0:print({'cache_records_scanned':records},flush=True)
                except (OSError,ValueError,TypeError) as exc:
                    errors.append({'record':str(record),'error':str(exc)})
    report={'records':records,'errors':errors,'resolved_staging_paths':relocated,
            'duplicate_groups':[], 'estimated_reclaimable_bytes':0,
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


def diagnostic_cleanup(roots, apply=False, min_age_hours=24, journal=None, log_every=5000):
    """Remove only known diagnostics, never whole failed/staging directories."""
    protected=set();errors=[];files=set();metadata=set();visited=set();scanned=0
    started=time.monotonic();last_log=started
    def progress(stage,count,force=False,**extra):
        nonlocal last_log
        now=time.monotonic()
        if force or count%log_every==0 or now-last_log>=5:
            print(json.dumps({'cleanup_stage':stage,'processed':count,
                'elapsed_s':round(now-started,1),**extra}),flush=True)
            last_log=now
    progress('discover',0,True)
    def walk_error(exc):errors.append({'record':str(exc.filename),'error':str(exc)})
    for root in roots:
        for directory,dirs,names in os.walk(root,followlinks=False,onerror=walk_error):
            if directory in visited:
                dirs[:]=[];continue
            visited.add(directory)
            scroll_directory=Path(directory).name in ('scroll-before','scroll-after')
            for name in names:
                scanned+=1
                # Avoid stat()/Path construction for unrelated images/assets.
                if name.endswith(('.json','.jsonl')):
                    metadata.add(Path(directory)/name)
                elif (name in ('before.png','reloaded.png','rejected.html') or
                      (scroll_directory and re.fullmatch(r'\d+\.png',name))):
                    path=Path(directory)/name
                    if not path.is_symlink():files.add(path)
                progress('discover',scanned,candidate_names=len(files),metadata_files=len(metadata))
    progress('discover',scanned,True,candidate_names=len(files),metadata_files=len(metadata))
    candidates=[];blocked=0;young=0;unclassified=0
    now=time.time()
    for index,path in enumerate(sorted(files),1):
        progress('classify',index,total=len(files))
        # Restrict to the known normalized page layout. Preserve every JSON,
        # normalized.html, initial.*, after.png and all source/rendered assets.
        folder=next((p for p in path.parents if p.parent.name in ('train','val','test')
                     and p.parent.parent.name=='normalized-pages'
                     and re.fullmatch('[0-9a-f]{64}',p.name)),None)
        if folder is None:continue
        rel=path.relative_to(folder)
        reason=None
        if len(rel.parts)==1 and path.name in ('before.png','reloaded.png'):
            reason='normalization_diagnostic'
        elif len(rel.parts)==2 and rel.parts[0] in ('scroll-before','scroll-after') and re.fullmatch(r'\d+\.png',path.name):
            reason='scroll_diagnostic'
        elif path.name=='rejected.html' and path.parent==folder:
            try:
                failed=json.loads((folder/'report.json').read_text()).get('accepted') is False
            except (OSError,ValueError):failed=False
            if failed and not (folder/'record.json').exists() and not (folder/'normalization.json').exists():
                reason='failed_normalization_candidate'
        if reason is None:
            unclassified+=1;continue
        if str(path) in protected:
            blocked+=1;continue
        st=path.stat()
        if now-st.st_mtime<min_age_hours*3600:
            young+=1;continue
        candidates.append({'path':str(path),'reason':reason,'size':st.st_size,
                           'device':st.st_dev,'inode':st.st_ino,'nlink':st.st_nlink,
                           'blocks':st.st_blocks,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns})
    # Keep only paths matching candidates; no stat/hash for unrelated metadata
    # references. Still parse every JSON/JSONL so bad metadata blocks deletion.
    candidate_paths={c['path'] for c in candidates}
    basenames={Path(p).name for p in candidate_paths}
    staging_aliases=defaultdict(set)
    for name in candidate_paths:
        parts=Path(name).parts
        for index,part in enumerate(parts):
            if re.fullmatch('[0-9a-f]{64}',part):
                staging_aliases[part,parts[index+1:]].add(name)
    def references(value,origin):
        if isinstance(value,dict):
            for key,item in value.items():references(key,origin);references(item,origin)
        elif isinstance(value,list):
            for item in value:references(item,origin)
        elif isinstance(value,str) and len(value)<4096 and '\x00' not in value and value.rsplit('/',1)[-1] in basenames:
            path=Path(value)
            # A migrated shared cache can retain historical metadata rooted in
            # a DIFFERENT old cache directory. Match key+full relative suffix,
            # not just the basename. Extra protection is safe here.
            for index,part in enumerate(path.parts):
                match=re.fullmatch(r'\.([0-9a-f]{64})\.[A-Za-z0-9_-]+',part)
                if match:protected.update(staging_aliases.get((match[1],path.parts[index+1:]),()))
            choices=[os.path.abspath(path)] if path.is_absolute() else [os.path.abspath(origin.parent/path),os.path.abspath(path)]
            # Protect even historical staging references without probing disk
            # for every string; conservative matches can only prevent deletion.
            choices.extend(re.sub(r'/\.([0-9a-f]{64})\.[A-Za-z0-9_-]+/',r'/\1/',p) for p in list(choices))
            protected.update(candidate_paths.intersection(choices))
            if not candidate_paths.intersection(choices):
                for p in choices:
                    try:
                        resolved=str(Path(p).resolve())
                        if resolved in candidate_paths:protected.add(resolved)
                    except (OSError,ValueError):pass
    progress('references',0,True,total=len(metadata))
    for index,path in enumerate(sorted(metadata),1):
        try:
            with path.open() as stream:
                if path.suffix=='.jsonl':
                    for line_number,line in enumerate(stream,1):
                        if line.strip():references(json.loads(line),path)
                        if line_number%10000==0:progress('references',index,file=str(path),lines=line_number)
                else:references(json.load(stream),path)
        except (OSError,ValueError,TypeError) as exc:
            errors.append({'record':str(path),'error':str(exc)})
        progress('references',index,total=len(metadata),protected=len(protected))
    progress('references',len(metadata),True,total=len(metadata),protected=len(protected))
    blocked=len(protected)
    candidates=[c for c in candidates if c['path'] not in protected]
    by_inode=defaultdict(list)
    for c in candidates:by_inode[c['device'],c['inode']].append(c)
    reclaim=sum(group[0]['blocks']*512 for group in by_inode.values() if len(group)==group[0]['nlink'])
    report={'mode':'diagnostic_cleanup','errors':errors,'candidates':candidates,
            'candidate_count':len(candidates),'referenced_preserved':blocked,'recent_preserved':young,
            'unclassified_preserved':unclassified,'logical_bytes':sum(c['size'] for c in candidates),
            'estimated_reclaimable_bytes':reclaim,'applied':False}
    if apply and not errors:
        # Validate the entire plan before deleting any file. Writers must stop.
        progress('verify_delete',0,True,total=len(candidates))
        for index,c in enumerate(candidates,1):
            p=Path(c['path']);st=p.stat()
            if p.is_symlink() or (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=(c['device'],c['inode'],c['size'],c['mtime_ns'],c['ctime_ns']):
                raise ValueError(f'Diagnostic changed during audit: {p}')
            c['sha256']=digest(p)
            after=p.stat()
            if (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError(f'Diagnostic changed while hashing: {p}')
            progress('verify_delete',index,total=len(candidates))
        if journal is not None:
            journal.parent.mkdir(parents=True,exist_ok=True)
            journal.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        for index,c in enumerate(candidates,1):
            Path(c['path']).unlink()
            progress('delete',index,total=len(candidates))
        report['applied']=True
    progress('complete',len(candidates),True,applied=report['applied'],errors=len(errors))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--store',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--cleanup-diagnostics',action='store_true')
    parser.add_argument('--include-lazy-intermediates',action='store_true',
                        help='Also hardlink identical rendered assets in completed lazy caches; preserve all paths')
    parser.add_argument('--min-age-hours',type=float,default=24)
    parser.add_argument('--log-every',type=int,default=5000)
    args=parser.parse_args()
    if args.min_age_hours<0:parser.error('min-age-hours must be nonnegative')
    if args.log_every<1:parser.error('log-every must be positive')
    if args.cleanup_diagnostics and args.include_lazy_intermediates:
        parser.error('Use include-lazy-intermediates with deduplication, not diagnostic cleanup')
    roots=[Path(p).resolve() for p in args.roots]
    if any(not p.is_dir() for p in roots):parser.error('All roots must be existing cache directories')
    report=(diagnostic_cleanup(roots,args.apply,args.min_age_hours,Path(args.report),args.log_every) if args.cleanup_diagnostics
            else run(roots,Path(args.store).resolve(),args.apply,args.include_lazy_intermediates))
    target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('duplicate_groups','resolved_staging_paths','candidates')},ensure_ascii=False))
    print(f'Report: {target}')
    if args.cleanup_diagnostics:
        if args.apply and report['errors']:raise SystemExit('Cleanup blocked: metadata errors; inspect report')
    else:
        print(f'Resolved staging paths: {len(report["resolved_staging_paths"])}')
        print(f'Duplicate groups: {len(report["duplicate_groups"])}')


if __name__=='__main__':main()

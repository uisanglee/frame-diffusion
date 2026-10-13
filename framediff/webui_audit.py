"""Read-only raw corpus/archive audit. Never downloads or extracts source data."""
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .ir import write_json, write_jsonl
from .webui_visual import scan_sources, _assign

VIEWS=['default_1280-720','default_1366-768','default_1536-864',
       'default_1920-1080','iPad-Pro','iPhone-13 Pro']
ARCHIVES=[('webui-70k','train_split_web70k.zip','train'),
          ('webui-val','val_split_webui.zip','val'),
          ('webui-test','test_split_webui.zip','test')]


def requested_member(name,views):
    return any(Path(name).name==view+suffix for view in views
               for suffix in ('-screenshot.webp','-screenshot.png','-html.html','-url.txt'))


def archive_inventory(first, destination, views, out, sevenzip):
    """Stream 7z's member listing; compare paths and uncompressed byte sizes."""
    issues=[];checked=0;counts=Counter();tail=[];block={}
    def consume():
        nonlocal checked
        name=block.get('Path','')
        if not requested_member(name,views) or 'Size' not in block:return
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            issues.append({'reason':'unsafe_archive_member','member':name});return
        path=destination/relative;checked+=1
        expected=int(block['Size'])
        if not path.is_file():reason='missing_extracted_file'
        elif path.stat().st_size!=expected:reason='extracted_size_mismatch'
        else:counts['matched_size']+=1;return
        counts[reason]+=1
        issues.append({'reason':reason,'member':name,'path':str(path),'expected_bytes':expected,
                       'actual_bytes':path.stat().st_size if path.is_file() else None})
    with subprocess.Popen([sevenzip,'l','-slt',str(first)],stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,errors='replace') as proc:
        for line in proc.stdout:
            line=line.rstrip('\n');tail=(tail+[line])[-20:]
            if not line.strip():consume();block={}
            elif ' = ' in line:
                key,value=line.split(' = ',1);block[key]=value
        consume();code=proc.wait()
    write_jsonl(out/'extraction-issues.jsonl',issues)
    return {'status':'listed' if code==0 else 'listing_failed','returncode':code,
            'checked_members':checked,'counts':dict(counts),'issues':len(issues),
            'extraction_status':('unknown' if code or not checked else 'mismatch' if issues else 'all_requested_members_present_same_size'),
            'tail':tail if code else [],'note':'Size agreement is not a content checksum.'}


def remote_inventory(repo, local):
    """Hub metadata only; pagination handled by HfApi. No payload download."""
    from huggingface_hub import HfApi
    rows=[]
    for entry in HfApi().list_repo_tree('biglab/'+repo,repo_type='dataset',recursive=True):
        if not hasattr(entry,'size') or '.zip.' not in entry.path:continue
        path=local/entry.path;size=path.stat().st_size if path.is_file() else None
        rows.append({'file':entry.path,'expected_bytes':entry.size,'actual_bytes':size,
                     'status':'missing' if size is None else 'size_mismatch' if size!=entry.size else 'size_matches'})
    return {'status':'checked' if rows else 'no_archive_metadata','files':rows,
            'reference':'current Hugging Face main revision; may differ from a historical download',
            'counts':dict(Counter(r['status'] for r in rows)),
            'note':'Remote names/sizes only. Run --test-archives for local archive CRC testing.'}


def audit(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    if not root.is_dir():raise ValueError(f'Raw directory does not exist: {root}')
    if (args.check_remote or args.test_archives) and not args.downloads:
        raise ValueError('--check-remote / --test-archives require --downloads')
    views=args.views or VIEWS
    candidates,_,_,scan=scan_sources(root,views,out)
    groups=defaultdict(list)
    for page in candidates:groups[page['group']].append(page)
    allocation={}
    if args.size<10 or args.size%10:raise ValueError('--size must be a positive multiple of 10')
    try:
        _assign(groups,dict(train=args.size*8//10,val=args.size//10,test=args.size//10),args.seed,allocation)
        allocation['status']='sufficient'
    except ValueError as exc:allocation.update(status='insufficient',error=str(exc))
    write_jsonl(out/'selection-unselected.jsonl',allocation.pop('unselected',[]))
    write_json(out/'selection-audit.json',allocation)
    summary={'source':scan,'selection':allocation,'archives':{},
             'note':'Original archive split names are not the new experimental train/val/test assignment.'}
    write_json(out/'audit-summary.json',summary) # Preserve useful results even if archive inspection is interrupted.
    if args.downloads:
        downloads=Path(args.downloads).resolve();sevenzip=shutil.which('7z') or shutil.which('7zz')
        for repo,archive,split in ARCHIVES:
            print({'archive_audit_started':split,'check_remote':args.check_remote,'test_integrity':args.test_archives},flush=True)
            folder=downloads/repo;destination=root/split;first=folder/(archive+'.001')
            section={'first_volume':str(first),'first_volume_present':first.is_file(),
                'local_volumes':[{'name':p.name,'bytes':p.stat().st_size} for p in sorted(folder.glob(archive+'.*')) if p.is_file()],
                'incomplete_files':[str(p) for p in folder.rglob('*.incomplete')],
                'download_metadata':{'status':'not_checked'},'integrity':{'status':'not_tested'}}
            if args.check_remote:
                try:section['download_metadata']=remote_inventory(repo,folder)
                except Exception as exc:section['download_metadata']={'status':'unavailable','error':str(exc)}
            if first.is_file() and sevenzip:
                report_dir=out/split;report_dir.mkdir(parents=True,exist_ok=True)
                try:section['extraction']=archive_inventory(first,destination,views,report_dir,sevenzip)
                except Exception as exc:section['extraction']={'status':'unknown','error':str(exc)}
                if args.test_archives:
                    log=report_dir/'archive-test.log'
                    with log.open('w') as stream:
                        code=subprocess.run([sevenzip,'t',str(first)],stdout=stream,stderr=subprocess.STDOUT).returncode
                    section['integrity']={'status':'passed' if code==0 else 'failed','returncode':code,'log':str(log)}
            else:section['extraction']={'status':'unknown','reason':'missing_first_volume' if not first.is_file() else '7z/7zz_not_installed'}
            summary['archives'][split]=section
            write_json(out/'audit-summary.json',summary)
            print(json.dumps({'archive_audit':split,**section},ensure_ascii=False),flush=True)
    print('=== SOURCE / SELECTION ===',flush=True)
    print(json.dumps({'source':scan,'selection':allocation},ensure_ascii=False,indent=2),flush=True)
    print(f'Audit saved: {out / "audit-summary.json"}',flush=True)
    return summary


def add_parser(sub):
    p=sub.add_parser('visual-webui-audit',help='Audit raw eligibility, download volumes and extraction without rendering')
    p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--downloads');p.add_argument('--views',nargs='+')
    p.add_argument('--size',type=int,default=70000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--check-remote',action='store_true',help='Compare local archive volume names/sizes with Hugging Face metadata')
    p.add_argument('--test-archives',action='store_true',help='Run full 7z CRC tests; reads all archive volumes and can be slow')

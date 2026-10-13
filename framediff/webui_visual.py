"""Domain-disjoint WebUI sampling for visual detector/policy fine-tuning."""
import gzip
import hashlib
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from .ir import write_json,write_jsonl
from .visual import CONTRACT
from .web_experiment import digest,guard_run


TEXT_ROLES={'StaticText','InlineTextBox'}
IMAGE_ROLES={'img','image','graphics-object','canvas'}
CONTROL_ROLES={'button','textbox','searchbox','checkbox','radio','combobox','slider','switch','spinbutton','listbox'}


def _read_gzip(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def _native_elements(ax_path,bb_path,viewport):
    nodes=_read_gzip(ax_path).get('nodes',[]);boxes=_read_gzip(bb_path);w,h=viewport
    result=[];seen=set()
    for node in nodes:
        if node.get('ignored'):continue
        role=node.get('role',{}).get('value','')
        label=1 if role in TEXT_ROLES else 2 if role in IMAGE_ROLES else 3 if role in CONTROL_ROLES else 0
        box=boxes.get(str(node.get('backendDOMNodeId')))
        if not label or not box:continue
        x,y,bw,bh=(float(box[k]) for k in ('x','y','width','height'))
        x1,y1,x2,y2=max(0.,x),max(0.,y),min(float(w),x+bw),min(float(h),y+bh)
        key=(label,round(x1,2),round(y1,2),round(x2,2),round(y2,2))
        if x2<=x1 or y2<=y1 or key in seen:continue
        seen.add(key);result.append({'box':[x1,y1,x2,y2],'label':label})
    return result


def _assign(groups,counts,seed,diagnostics=None):
    rng=random.Random(seed);items=list(groups.items());rng.shuffle(items)
    selected={split:[] for split in counts};assigned={}
    for group,rows in items:
        available=[s for s in counts if len(selected[s])<counts[s]]
        if not available:break
        split=max(available,key=lambda s:(counts[s]-len(selected[s]))/counts[s])
        rng.shuffle(rows);take=min(len(rows),counts[split]-len(selected[split]))
        selected[split].extend(rows[:take]);assigned[group]=split
    missing={s:counts[s]-len(selected[s]) for s in counts if len(selected[s])<counts[s]}
    if diagnostics is not None:
        used={r['id'] for rows in selected.values() for r in rows}
        diagnostics.update(requested=counts,assigned={s:len(rows) for s,rows in selected.items()},
            missing=missing,eligible_unique_pages=sum(len(rows) for rows in groups.values()),
            unselected=[{'page_id':r['id'],'group':g,
                         'reason':'domain_split_quota_exhausted' if g in assigned else 'requested_counts_filled'}
                        for g,rows in groups.items() for r in rows if r['id'] not in used])
    if missing:raise ValueError(f'Not enough domain-disjoint WebUI pages for requested split: {missing}')
    return selected,assigned


def _views(args):
    views=getattr(args,'views',None)
    if views:
        result=list(dict.fromkeys(views))
    else:
        result=[getattr(args,'view','default_1280-720')]
    if not result or any(not isinstance(view,str) or not view for view in result):
        raise ValueError('At least one nonempty WebUI view is required')
    return result


def scan_sources(root, views, audit_dir):
    """The importer and standalone audit use exactly the same eligibility scan."""
    root=Path(root).resolve();audit_dir=Path(audit_dir)
    images=[];events=[];paired_pages=set();paired_views=0
    def reject(reason,unit,**details):
        excluded[reason]+=1;events.append(dict(reason=reason,unit=unit,**details))
    for view in views:
        images.extend(root.rglob(f'{view}-screenshot.webp'));images.extend(root.rglob(f'{view}-screenshot.png'))
    images=sorted(set(images));raw_pages=defaultdict(list);excluded=Counter()
    for index,image in enumerate(images):
        if index%5000==0:
            print({'source_scan_files':index,'total_files':len(images),'view_exclusions_so_far':dict(excluded)},flush=True)
        page_id=image.parent.name
        view=image.name.rsplit('-screenshot.',1)[0];stem=image.parent/view
        html=Path(str(stem)+'-html.html');url_path=Path(str(stem)+'-url.txt')
        ax=Path(str(stem)+'-axtree.json.gz');bb=Path(str(stem)+'-bb.json.gz')
        detail=dict(page_id=page_id,view=view,screenshot=str(image),html=str(html))
        if not html.is_file():reject('missing_html','page_view',**detail);continue
        paired_pages.add(page_id);paired_views+=1
        try:
            source_hash=digest(html)
            url=url_path.read_text(errors='replace').strip() if url_path.exists() else ''
            host=(urlparse(url).hostname or '').lower()
            with Image.open(image) as source:
                viewport=list(source.size);source.verify()
        except (OSError,ValueError,SyntaxError) as exc:
            reject('unreadable_source','page_view',error=str(exc),**detail);continue
        expected=view.removeprefix('default_').split('-')
        if len(expected)==2 and all(v.isdigit() for v in expected) and viewport!=list(map(int,expected)):
            reject('viewport_mismatch','page_view',actual=viewport,expected=list(map(int,expected)),**detail);continue
        # WebUI's timestamp directory is the stable page identity across
        # separately downloaded train/val/test archives and all device views.
        page_id=image.parent.name
        raw_pages[page_id].append({'view':view,'html':str(html),'screenshot':str(image),
            'viewport':viewport,'source_sha':source_hash,'host':host,'ax':ax,'bb':bb})
    candidates=[];seen_hashes={}
    for page_id,page_views in sorted(raw_pages.items()):
        unique={row['view']:row for row in page_views}
        for row in page_views:
            if row is not unique[row['view']]:
                reject('duplicate_page_view','page_view',page_id=page_id,view=row['view'],
                    screenshot=row['screenshot'],retained_screenshot=unique[row['view']]['screenshot'])
        page_views=[unique[view] for view in views if view in unique]
        if not page_views:continue
        hashes={row['source_sha'] for row in page_views}
        duplicate=next((seen_hashes[h] for h in sorted(hashes) if h in seen_hashes),None)
        if duplicate is not None:
            reject('duplicate_html_page','unique_page',page_id=page_id,duplicate_of=duplicate,
                matching_views=[r['view'] for r in page_views if r['source_sha'] in seen_hashes],
                rule='any shared view HTML hash excludes the whole page');continue
        for h in hashes:seen_hashes[h]=page_id
        hosts={row['host'] for row in page_views if row['host']}
        if len(hosts)>1:
            reject('inconsistent_view_host','unique_page',page_id=page_id,
                hosts_by_view={r['view']:r['host'] for r in page_views})
            continue # Do not pretend one of several domains defines a disjoint group.
        host=sorted(hosts)[0] if hosts else ''
        group='webui-domain:'+host if host else 'webui-page:'+page_id
        candidates.append({'id':page_id,'page_id':page_id,'group':group,'views':page_views,
                           'page_sha':hashlib.sha256('\n'.join(sorted(hashes)).encode()).hexdigest()})
    screenshot_pages={p.parent.name for p in images}
    eligible_ids={p['id'] for p in candidates}
    for event in events:
        if event['reason']=='duplicate_html_page':
            event['duplicate_of_eligible_page']=event['duplicate_of'] in eligible_ids
    regions=defaultdict(list)
    for image in images:regions[image.relative_to(root).parts[0]].append(image)
    summary={'root':str(root),'views':views,'screenshot_files':len(images),
        'screenshot_unique_pages':len(screenshot_pages),'html_paired_views':paired_views,
        'html_paired_unique_pages':len(paired_pages),'pages_after_view_checks':len(raw_pages),
        'pages_with_no_usable_view':len(screenshot_pages-set(raw_pages)),
        'eligible_unique_pages':len(candidates),
        'eligible_page_views':sum(len(p['views']) for p in candidates),
        'eligible_views':dict(Counter(r['view'] for p in candidates for r in p['views'])),
        'filesystem_regions':{name:{'screenshot_files':len(paths),'unique_page_ids':len({p.parent.name for p in paths})}
                              for name,paths in regions.items()},
        'excluded_unique_pages':len(screenshot_pages)-len(candidates),
        'exclusions_by_unit':{unit:dict(Counter(e['reason'] for e in events if e['unit']==unit))
                              for unit in ('page_view','unique_page')},
        'note':'View exclusions and page exclusions are different units; do not sum them as lost pages.'}
    write_json(audit_dir/'source-scan.json',summary)
    write_jsonl(audit_dir/'source-exclusions.jsonl',events)
    write_jsonl(audit_dir/'pages-with-no-usable-view.jsonl',
        ({'page_id':p} for p in sorted(screenshot_pages-set(raw_pages))))
    return candidates,images,excluded,summary


def import_webui(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    if min(args.train_count,args.val_count,args.test_count)<1:raise ValueError('All split counts must be positive')
    views=_views(args)
    candidates,images,excluded,scan=scan_sources(root,views,out)
    grouped=defaultdict(list)
    for row in candidates:grouped[row['group']].append(row)
    counts={'train':args.train_count,'val':args.val_count,'test':args.test_count}
    allocation={}
    try:selected,assigned=_assign(grouped,counts,args.seed,allocation)
    except ValueError as exc:
        write_json(out/'selection-failure.json',{'requested':counts,'eligible_unique_pages':len(candidates),
            'eligible_domains':len(grouped),'error':str(exc),'source_scan':str(out/'source-scan.json'),
            'exclusions':str(out/'source-exclusions.jsonl')})
        raise ValueError(f'{exc}; diagnostics: {out / "selection-failure.json"}') from exc
    finally:
        write_jsonl(out/'selection-unselected.jsonl',allocation.pop('unselected',[]))
        write_json(out/'selection-audit.json',allocation)
    sources=[];native_rows=[];selected_pages=[]
    for split,pages in selected.items():
        for page in pages:
            selected_pages.append({k:page[k] for k in ('id','page_id','group','page_sha')}|{'split':split})
            for row in page['views']:
                sample_id=f"{page['page_id']}/{row['view']}"
                source={k:row[k] for k in ('html','screenshot','viewport','source_sha')}
                source.update(id=sample_id,page_id=page['page_id'],view=row['view'],group=page['group'],split=split)
                sources.append(source)
                native=[]
                if row['ax'].exists() and row['bb'].exists():
                    try:native=_native_elements(row['ax'],row['bb'],row['viewport'])
                    except (OSError,EOFError,ValueError,KeyError,TypeError):excluded['invalid_ax_boxes']+=1
                else:excluded['missing_ax_boxes']+=1
                if native:
                    native_rows.append({'id':sample_id,'page_id':page['page_id'],'view':row['view'],
                        'group':page['group'],'split':split,
                        'source_sha':row['source_sha'],'contract':CONTRACT,'viewport':row['viewport'],
                        'image':row['screenshot'],'elements':native,
                        'annotation_scope':'webui-ax-semantic-v1-no-painted-region'})
    config={'kind':'webui-visual-import-v2-multiview','root':str(root),'views':views,'seed':args.seed,
        'counts':counts,'pages':selected_pages,
        'sources':[{k:r[k] for k in ('id','page_id','view','group','source_sha','split')} for r in sources]}
    guard_run(out,config,args.resume)
    order={'train':0,'val':1,'test':2};sources.sort(key=lambda r:(order[r['split']],r['id']))
    native_rows.sort(key=lambda r:(order[r['split']],r['id']))
    write_jsonl(out/'manifest.jsonl',sources)
    write_jsonl(out/'pages.jsonl',selected_pages)
    for split in counts:
        write_jsonl(out/f'manifest-{split}.jsonl',[r for r in sources if r['split']==split])
        write_jsonl(out/f'pages-{split}.jsonl',[r for r in selected_pages if r['split']==split])
        write_jsonl(out/f'native-detector-{split}.jsonl',[r for r in native_rows if r['split']==split])
    report={'selected':{s:sum(r['split']==s for r in selected_pages) for s in counts},
        'selected_views':{s:sum(r['split']==s for r in sources) for s in counts},
        'view_distribution':{s:dict(Counter(r['view'] for r in sources if r['split']==s)) for s in counts},
        'selected_domains':{s:len({r['group'] for r in selected_pages if r['split']==s}) for s in counts},
        'native_detector_pages':{s:sum(r['split']==s for r in native_rows) for s in counts},
        'native_class_boxes':dict(Counter(e['label'] for r in native_rows for e in r['elements'])),
        'available_images':len(images),'eligible_unique_pages':len(candidates),'views':views,'excluded':dict(excluded),
        'note':'Native AX labels cover text/image/control only; painted-region requires browser-rendered labels.'}
    write_json(out/'report.json',report);print(json.dumps(report,indent=2));return sources

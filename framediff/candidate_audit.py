"""Reinspect rejected pages without regenerating normalization or training data."""
import copy
import hashlib
from collections import Counter
from pathlib import Path

from .ir import read_json, read_jsonl, write_json, write_jsonl


def run(args):
    from . import css_owners
    from .explicit_html import feedback_metadata
    from .html_bridge import HtmlBrowser
    from .tree_online import target_pool, EmptyTargetPoolError
    from .visual import annotate, elements

    root=Path(args.normalized).resolve();out=Path(args.out).resolve()
    if out.exists():raise ValueError('Use a new audit output directory')
    if args.limit<0:raise ValueError('limit must be nonnegative')
    rejected=[r for r in read_jsonl(root/f'rejections-{args.split}.jsonl')
              if r.get('stage')=='visible_candidate_inspection']
    if args.limit:rejected=rejected[:args.limit]
    wanted={r['id'] for r in rejected};sources={}
    config=read_json(root/'config.json') if (root/'config.json').exists() else {}
    paths=[Path(args.source)/f'policy-{args.split}.jsonl'] if args.source else [
        Path(p) for p in config.get('sources',{}) if Path(p).name==f'policy-{args.split}.jsonl']
    needs_source=any(not (root/'normalized-pages'/args.split/
        hashlib.sha256(id.encode()).hexdigest()/'inspection-input.json').is_file() for id in wanted)
    for path in paths if needs_source else []:
        for row in read_jsonl(path):
            id=row['id'].rsplit('/',1)[0]
            if id in wanted and (id not in sources or row['id'].endswith('/clean')):sources[id]=row
    out.mkdir(parents=True)
    summary=Counter();probe_counts=Counter();records=[]
    for rejection in rejected:
        id=rejection['id'];key=hashlib.sha256(id.encode()).hexdigest()
        folder=root/'normalized-pages'/args.split/key
        diagnostics=[]
        try:
            if (folder/'inspection-input.json').is_file():
                seed=read_json(folder/'inspection-input.json')
            else:
                # Older runs did not save the seed preceding failed inspection.
                # Reconstruct it from the preserved normalized HTML, never the
                # unnormalized source or a newly generated corruption.
                row=sources[id];html=folder/'normalized.html'
                tree=copy.deepcopy(row['current']);viewport=row['viewport']
                with HtmlBrowser() as browser:
                    browser.load(html.read_text(),viewport);annotate(browser,tree)
                    parsed=css_owners.read(browser,tree)
                    boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
                    boxes[tree['nodes'][0]['id']]=[0,0,*viewport]
                    feedback=feedback_metadata(browser,tree)
                    items=elements(tree,boxes,viewport,feedback['current_clip_boxes'])
                seed={k:row[k] for k in ('group','source_sha','viewport')}
                seed.update(id=id+'/audit',current=tree,target_html=str(html),
                    target_image=str(folder/'after.png'),target_elements=items,
                    css_owners=parsed['owners'],target_declaration_state=parsed['state'])
            try:
                target_pool([{**seed,'split':'train'}],cache_dir=out/'cache',
                            observation_size=args.observation_size,diagnostics=diagnostics)
            except EmptyTargetPoolError:
                pass
            result=diagnostics[0]
        except Exception as exc:
            result=dict(id=id,reason='audit_setup_error',error=f'{type(exc).__name__}: {exc}')
        write_json(out/f'{key}.json',result)
        summary[result['reason']]+=1
        for site in result.get('sites',[]):
            for probe in site['probes']:probe_counts[probe['outcome']]+=1
        records.append(dict(id=id,reason=result['reason'],report=str(out/f'{key}.json')))
        write_jsonl(out/'pages.jsonl',records)
        write_json(out/'summary.json',dict(pages=len(records),reasons=dict(summary),
            probe_outcomes=dict(probe_counts),finite_probe_search=True))
        print(f'[{len(records)}/{len(rejected)}] {id}: {result["reason"]}',flush=True)

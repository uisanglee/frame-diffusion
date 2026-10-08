"""Fixed held-out inline+stylesheet corruptions; preserve original target assets."""
import hashlib
from collections import Counter
from pathlib import Path

from .css_owners import CONTRACT
from .ir import read_json,read_jsonl,write_json,write_jsonl
from .tree_online import ONLINE_CONTRACT,OnlineSampler,target_pool,asset_signatures
from .web_experiment import guard_run,digest


def freeze(args):
    source=Path(args.rendered).resolve();out=Path(args.out).resolve()
    if out==source or source in out.parents:raise ValueError('Use a separate frozen-data output directory')
    if args.samples_per_page<1:raise ValueError('samples-per-page must be positive')
    files=[source/f'{kind}-{split}.jsonl' for split in ('train','val','test') for kind in ('policy','pages')]
    guard_run(out,{'kind':CONTRACT+'-fixed-v1','corruption_contract':ONLINE_CONTRACT,
        'data':{str(p):digest(p) for p in files},
        'seed':args.seed,'max_noise':args.max_noise,'samples':args.samples_per_page,
        'observation_size':getattr(args,'observation_size',384)},args.resume)
    # Online training needs clean targets, not fresh stored train screenshots.
    if any(r.get('teacher_strategy')!=CONTRACT for r in read_jsonl(source/'policy-train.jsonl')):
        raise ValueError('Prepare --stylesheets labels with the current size/margin action contract first')
    for kind in ('policy','pages'):
        write_jsonl(out/f'{kind}-train.jsonl',list(read_jsonl(source/f'{kind}-train.jsonl')))
    report={}
    for split in ('val','test'):
        rows=list(read_jsonl(source/f'policy-{split}.jsonl'))
        if any(r['teacher_strategy']!=CONTRACT for r in rows):raise ValueError('Prepare --stylesheets labels first')
        # target_pool's train guard is for the actual training stream. This
        # command intentionally samples held-out pages and restores their split.
        pool=target_pool([{**r,'split':'train'} for r in rows],cache_dir=source/'.online-target-cache',
                         observation_size=getattr(args,'observation_size',384))
        pages={p['id']:p for p in read_jsonl(source/f'pages-{split}.jsonl')}
        kept=[];selected=[];errors=[];counts=Counter()
        for index,target in enumerate(pool):
            page_dir=out/split/hashlib.sha256(target['id'].encode()).hexdigest()
            cached=page_dir/'record.json';signature=asset_signatures([target])
            if args.resume and cached.exists():
                saved=read_json(cached)
                if saved['signature']!=signature:raise ValueError('Frozen target assets changed; use a new output directory')
                new=saved['rows']
                for row in new:
                    if not Path(row['current_html']).is_file() or not Path(row['current_image']).is_file():
                        raise ValueError('Incomplete frozen page assets')
            else:
                sampler=OnlineSampler([target],'screenshot',seed=args.seed,max_noise=args.max_noise)
                new=[]
                try:
                    page_dir.mkdir(parents=True,exist_ok=True)
                    for sample in range(args.samples_per_page):
                        row=sampler.generate(index*args.samples_per_page+sample)
                        html=page_dir/f'{sample}.html';png=page_dir/f'{sample}.png'
                        html.write_text(row.pop('current_html_text'));png.write_bytes(row['current_image'])
                        row.update(split=split,current_html=str(html),current_image=str(png))
                        new.append(row)
                    write_json(cached,{'signature':signature,'rows':new})
                except Exception as exc:
                    errors.append({'id':target['id'],'error':str(exc)});new=[]
                finally:sampler.close()
            kept.extend(new)
            if new:selected.append({**pages[target['id']],'initial_html':new[0]['current_html']})
            for row in new:
                edit=row['replacement_edit'];owner=row['css_owners'][edit[0]]
                counts[owner['kind']+'/'+('SET' if edit[2] else 'REMOVE')]+=1
                counts['property/'+edit[1]]+=1
            if (index+1)%50==0 or index==len(pool)-1:
                print(f'{split} fixed CSS corruption [{index+1}/{len(pool)}], kept={len(kept)}, failures={len(errors)}',flush=True)
        write_jsonl(out/f'policy-{split}.jsonl',kept);write_jsonl(out/f'pages-{split}.jsonl',selected)
        write_jsonl(out/f'rejections-{split}.jsonl',errors)
        report[split]={'corruption_contract':ONLINE_CONTRACT,'pages':len(selected),
            'rows':len(kept),'failed_pages':len(errors),'teachers':dict(counts)}
        if not kept:raise ValueError(f'Empty frozen {split}; inspect rejection report')
    write_json(out/'freeze-report.json',report)

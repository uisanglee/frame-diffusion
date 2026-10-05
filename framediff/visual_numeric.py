"""A read-only view of cached numeric corruption prefixes. No browser needed."""
from pathlib import Path
import re
from collections import Counter
from .ir import read_jsonl, write_jsonl, write_json
from .visual import NUMERIC_FIELDS,elements
from .web_experiment import digest, guard_run


def numeric_rows(rows):
    indexed={r['id']:r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('Duplicate policy IDs')
    kept=[]
    for row in rows:
        prefix, name=row['id'].rsplit('/',1)
        match=re.fullmatch(r't(\d+)-s(\d+)',name)
        if name=='clean':
            if row['teacher_edits']:raise ValueError('Clean row has edits')
        elif match:
            trajectory,step=map(int,match.groups())
            history=[indexed.get(f'{prefix}/t{trajectory}-s{k}') for k in range(step+1)]
            if any(r is None or not r['teacher_edits'] or
                   (r.get('corruption_edit') or r['teacher_edits'][0])[1] not in NUMERIC_FIELDS or
                   any(e[1] not in NUMERIC_FIELDS for e in r['teacher_edits']) for r in history):continue
        else:raise ValueError(f'Unrecognized trajectory ID: {row["id"]}')
        kept.append({**row,'policy_subset':'numeric-prefix-v1'})
    return kept


def prepare(args):
    source=Path(args.rendered).resolve();out=Path(args.out).resolve()
    if out==source or source in out.parents:raise ValueError('Use a separate output directory')
    paths=[source/f'{kind}-{split}.jsonl' for split in ('train','val','test') for kind in ('policy','pages')]
    guard_run(out,{'kind':'numeric-prefix-v1','sources':{str(p):digest(p) for p in paths}},True)
    report={}
    for split in ('train','val','test'):
        rows=list(read_jsonl(source/f'policy-{split}.jsonl'));kept=numeric_rows(rows)
        clean={r['target_image']:r for r in rows if r['id'].endswith('/clean')}
        for row in kept:
            if 'target_elements' not in row:
                target=clean.get(row['target_image'])
                if target is None:raise ValueError('Missing clean cached geometry for target masks')
                row['target_elements']=elements(target['current'],target['current_boxes'],target['viewport'])
        if not kept or not any(r['teacher_edits'] for r in kept):raise ValueError(f'No numeric edits in {split}')
        by_page={}
        for r in kept:
            if r['teacher_edits']:by_page[r['id'].rsplit('/',1)[0]]=r
        pages=[]
        for page in read_jsonl(source/f'pages-{split}.jsonl'):
            if page['id'] in by_page:
                pages.append({**page,'initial_html':by_page[page['id']]['current_html'],
                              'policy_subset':'numeric-prefix-v1'})
        if not pages:raise ValueError(f'No numeric evaluation pages in {split}')
        write_jsonl(out/f'policy-{split}.jsonl',kept)
        write_jsonl(out/f'pages-{split}.jsonl',pages)
        report[split]={'source_rows':len(rows),'kept_rows':len(kept),'pages':len(pages),
                       'properties':dict(Counter(e[1] for r in kept for e in r['teacher_edits']))}
    write_json(out/'subset-report.json',report);print(report,flush=True)

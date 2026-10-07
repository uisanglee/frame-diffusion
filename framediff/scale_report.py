"""Combine capacity runs without counting the shared baseline three times."""
import argparse
from pathlib import Path

from .ir import read_jsonl, read_json, write_json, write_jsonl
from .paper_report import report


def build(root):
    root=Path(root);combined=[];parameters={};page_sets=[]
    for scale in ('s','m','l'):
        rows=list(read_jsonl(root/scale/'evaluation/metrics.jsonl'))
        policies=[r for r in rows if r['method']=='abstract-policy']
        if not policies:raise ValueError(f'Missing abstract-policy results for {scale}')
        page_sets.append({r['id'] for r in policies})
        for row in rows:
            if row['method']=='abstract-policy':
                combined.append({**row,'method':f'TUIDE-{scale.upper()}'})
            elif scale=='s' and row['method']!='abstract-oracle':combined.append(row)
        summary=read_json(root/scale/'repair/timing-summary.json')
        parameters[f'TUIDE-{scale.upper()}']=summary['model_parameters']['abstract-policy']
    if any(ids!=page_sets[0] for ids in page_sets):raise ValueError('Scale runs must cover identical page/repeat IDs')
    out=root/'comparison';out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'metrics.jsonl',combined)
    write_json(out/'policy-parameters.json',parameters)
    report(combined,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    build(parser.parse_args().root)

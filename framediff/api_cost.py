"""Aggregate measured API token usage and estimate cost from user-supplied rates."""
import argparse
from collections import defaultdict
from pathlib import Path

from .ir import read_jsonl, write_json


def summarize(metrics, input_rate, output_rate):
    trials = defaultdict(list)
    for row in read_jsonl(metrics):
        if row.get('method') not in {'initial', 'self-revision-1', 'abstract-policy'}:
            continue
        page = row.get('page_id', row['id'].split('/repeat-')[0])
        trials[(page, row['method'])].append(row)
    totals = defaultdict(lambda: {'pages': 0, 'api_calls': 0., 'input_tokens': 0., 'output_tokens': 0.})
    for (_, method), rows in trials.items():
        item = totals[method];item['pages'] += 1
        for key, source in [('api_calls', 'vlm_calls'), ('input_tokens', 'input_tokens'), ('output_tokens', 'output_tokens')]:
            values = [float(row[source]) for row in rows if isinstance(row.get(source), (int, float))]
            if values:item[key] += sum(values) / len(values)
    result=[]
    for method in ('initial', 'self-revision-1', 'abstract-policy'):
        if method not in totals:continue
        item=dict(totals[method]);item['method']=method
        item['estimated_usd']=item['input_tokens']*input_rate/1_000_000+item['output_tokens']*output_rate/1_000_000
        result.append(item)
    return result


def main():
    p=argparse.ArgumentParser(description='Estimate API expense from measured experiment token usage')
    p.add_argument('--metrics',required=True);p.add_argument('--out',required=True)
    p.add_argument('--input-usd-per-million',type=float,required=True)
    p.add_argument('--output-usd-per-million',type=float,required=True)
    p.add_argument('--pricing-as-of',required=True)
    p.add_argument('--model',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=summarize(a.metrics,a.input_usd_per_million,a.output_usd_per_million)
    payload={'model':a.model,'pricing_as_of':a.pricing_as_of,
             'rates_usd_per_million':{'input':a.input_usd_per_million,'output':a.output_usd_per_million},
             'methods':rows}
    write_json(out/'api-cost.json',payload)
    lines=[f'# API cost: {a.model}','',f'Pricing snapshot: {a.pricing_as_of}. Estimates use measured API usage.','',
           '| Method | Pages | API calls | Input tokens | Output tokens | Estimated USD |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['pages']} | {row['api_calls']:.0f} | {row['input_tokens']:.0f} | {row['output_tokens']:.0f} | ${row['estimated_usd']:.4f} |")
    (out/'api-cost.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

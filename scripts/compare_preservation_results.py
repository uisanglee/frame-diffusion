#!/usr/bin/env python3
"""Compare saved original, pixel and geometry runs. No generation/evaluation."""
import argparse
import json
from pathlib import Path
import statistics


def load(root):
    result={}
    for path in Path(root).glob('evaluations/*/budget-*/results.jsonl'):
        level=path.parent.parent.name;budget=int(path.parent.name.split('-')[-1])
        config_path=path.parent/'config.json'
        config=json.loads(config_path.read_text()) if config_path.is_file() else {}
        inputs=config.get('inputs',{})
        fingerprints={kind:sorted(value for key,value in inputs.items() if key.endswith(suffix))
                      for kind,suffix in [('checkpoints','.pt'),('manifests','.jsonl')]}
        for line in path.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);key=(level,budget,r['method'])
            identity=(r['id'],r.get('repeat',1))
            if identity in result.setdefault(key,{}):raise ValueError(f'Duplicate trial: {path}: {identity}')
            r['_fingerprints']=fingerprints
            result[key][identity]=r
    return result


def compare(roots,out):
    runs={name:load(root) for name,root in roots.items()}
    lines=['# Preservation comparison', '',
           'Saved results only. Each row keeps all trials, including failures and rollback. Missing runs and cohort mismatches are reported.', '',
           '| Level | Budget | Target | Protection | N | Initial IoU | Final IoU | Delta | Improved % | Mean actions | Rollbacks | Geometry goal % |',
           '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    output=[];warnings=[]
    for key in sorted(set().union(*(set(r) for r in runs.values()))):
        available={n:r[key] for n,r in runs.items() if key in r}
        reference=next(iter(available.values()))
        for name,trials in available.items():
            if set(trials)!=set(reference) or any(abs(trials[id]['initial_box_iou']-reference[id]['initial_box_iou'])>1e-8 for id in set(trials)&set(reference)):
                warnings.append(f'Not paired: {key}/{name}; cohort or initial score differs.');continue
            first=next(iter(trials.values()))['_fingerprints']
            previous=next(iter(reference.values()))['_fingerprints']
            if first!=previous:
                warnings.append(f'Not paired: {key}/{name}; checkpoint or manifest fingerprints differ.');continue
            if not first['checkpoints'] or not first['manifests']:
                warnings.append(f'Fingerprint verification unavailable: {key}/{name}')
            rows=list(trials.values());initial=statistics.mean(r['initial_box_iou'] for r in rows)
            final=statistics.mean(r['final_box_iou'] for r in rows)
            improved=100*statistics.mean(r['final_box_iou']>r['initial_box_iou']+1e-8 for r in rows)
            actions=statistics.mean(r['actions'] for r in rows)
            rollbacks=sum(r.get('preservation_rollbacks',0) for r in rows)
            goals=100*statistics.mean(r.get('geometry_goal_reached',False) for r in rows)
            level,budget,method=key
            lines.append(f'| {level} | {budget} | {method} | {name} | {len(rows)} | {initial:.4f} | {final:.4f} | {final-initial:+.4f} | {improved:.1f} | {actions:.2f} | {rollbacks} | {goals:.1f} |')
            output.append(dict(level=level,budget=budget,method=method,protection=name,n=len(rows),initial=initial,final=final,delta=final-initial,improved=improved,mean_actions=actions,rollbacks=rollbacks,geometry_goal_percent=goals,failed=sum(bool(r.get('failed')) for r in rows)))
        for name in runs.keys()-available.keys():warnings.append(f'Not available yet: {key}/{name}')
    lines+=['','## Coverage notes','',*warnings]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    (out/'comparison.json').write_text(json.dumps(dict(results=output,warnings=warnings),indent=2)+'\n')
    print(out/'comparison.md')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True);p.add_argument('--pixel',required=True)
    p.add_argument('--geometry',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();compare(dict(original=args.original,pixel=args.pixel,geometry=args.geometry),args.out)

if __name__=='__main__':main()

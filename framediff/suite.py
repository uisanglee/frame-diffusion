"""Sequential orchestration keeps a single 24GB GPU available for each job."""
import json
import subprocess
import sys
import statistics
from pathlib import Path
from .ir import write_json

def run_suite(path):
    config=json.loads(Path(path).read_text());root=Path(config['out']);root.mkdir(parents=True,exist_ok=True)
    status=[]
    for seed in config.get('seeds',[42,43,44]):
        for experiment in config['experiments']:
            run=root/f"{experiment['name']}-seed{seed}"
            cmd=[sys.executable,'-m','framediff','train','--train',config['train'],'--val',config['val'],'--out',str(run),'--seed',str(seed),*config.get('train_args',[]),*experiment.get('train_args',[])]
            subprocess.run(cmd,check=True)
            cmd=[sys.executable,'-m','framediff','evaluate','--data',config['test'],'--out',str(run/'eval'),'--checkpoint',str(run/'best.pt'),'--seed',str(seed),*config.get('eval_args',[]),*experiment.get('eval_args',[])]
            subprocess.run(cmd,check=True)
            status.append({'seed':seed,'experiment':experiment['name'],'run':str(run),'complete':True})
            write_json(root/'status.json',status)
    # Keep seed-level results separate; avoid treating repeated seeds as independent pages.
    summaries=[]
    for item in status:
        summaries.append({**item,'summary':json.loads((Path(item['run'])/'eval/summary.json').read_text())})
    write_json(root/'all-seeds.json',summaries)
    aggregate=[]
    for experiment in config['experiments']:
        matching=[s for s in summaries if s['experiment']==experiment['name']]
        methods=sorted({r['method'] for s in matching for r in s['summary']})
        for method in methods:
            values=[r for s in matching for r in s['summary'] if r['method']==method]
            row={'experiment':experiment['name'],'method':method,'seeds':len(values)}
            for metric in ('box_iou','repair_seconds','browser_executions'):
                xs=[v[metric] for v in values];row[metric+'_mean']=statistics.mean(xs)
                row[metric+'_std']=statistics.stdev(xs) if len(xs)>1 else 0.
            aggregate.append(row)
    write_json(root/'aggregate.json',aggregate)

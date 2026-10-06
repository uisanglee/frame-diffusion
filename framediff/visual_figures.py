"""Publication-ready diagnostics for structured visual-policy training."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

from .ir import read_jsonl,write_json


STAGES=(('detector','Detector'),('raw-stage1','Screenshot Stage 1'),('policy-raw','Screenshot Stage 2'),
        ('abstract-stage1','Abstract Stage 1'),('policy-abstract','Abstract Stage 2'))
ACCURACY_KEYS=(('improving_action_rate','Improving action'),('node_accuracy','Node'),
               ('property_accuracy','Property'),('value_accuracy','Value'))


def pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError as error:
        raise RuntimeError("Install figure dependencies with: pip install -e '.[visual-metrics]'") from error


def rows(path):
    path=Path(path)
    return list(read_jsonl(path)) if path.exists() else []


def best_validation(records):
    values=[r for r in records if 'validation_loss' in r]
    return min(values,key=lambda r:r['validation_loss']) if values else None


def save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=300,bbox_inches='tight')


def training_curves(run_root,out):
    plt=pyplot();fig,axes=plt.subplots(2,3,figsize=(13,7));summary={}
    for axis,(folder,title) in zip(axes.flat,STAGES):
        records=rows(Path(run_root)/folder/'train.jsonl')
        train=[r for r in records if 'loss' in r];val=[r for r in records if 'validation_loss' in r]
        if not train:
            axis.set_axis_off();axis.set_title(title+' (missing)');continue
        axis.plot([r['step'] for r in train],[r['loss'] for r in train],label='train NLL/loss',linewidth=1.4)
        if val:
            axis.plot([r['step'] for r in val],[r['validation_loss'] for r in val],marker='o',label='validation',linewidth=1.5)
            best=best_validation(records);axis.scatter([best['step']],[best['validation_loss']],marker='*',s=100,label='best')
            summary[folder]={'best_step':best['step'],'best_validation_loss':best['validation_loss']}
        axis.set(title=title,xlabel='Optimizer step',ylabel='Loss');axis.grid(alpha=.25);axis.legend(fontsize=8)
    axes.flat[-1].set_axis_off();fig.suptitle('TUIDE training convergence');fig.tight_layout();save(fig,out,'training-curves');plt.close(fig)
    write_json(Path(out)/'training-summary.json',summary)


def action_accuracy(run_root,out):
    plt=pyplot();selected=[]
    for folder,title in STAGES[1:]:
        best=best_validation(rows(Path(run_root)/folder/'train.jsonl'))
        if best and 'improving_action_rate' in best:selected.append((folder,title,best))
    if not selected:return
    import numpy as np
    x=np.arange(len(selected));width=.18;fig,axis=plt.subplots(figsize=(10,4.8))
    for index,(key,label) in enumerate(ACCURACY_KEYS):
        axis.bar(x+(index-1.5)*width,[r.get(key) or 0 for _,_,r in selected],width,label=label)
    axis.set_xticks(x,[title for _,title,_ in selected],rotation=12,ha='right')
    axis.set_ylim(0,1);axis.set_ylabel('Validation accuracy');axis.grid(axis='y',alpha=.25);axis.legend(ncol=4)
    fig.tight_layout();save(fig,out,'action-accuracy');plt.close(fig)
    properties=sorted({p for _,_,r in selected for p in r.get('per_property',{})})
    if properties:
        fig,axes=plt.subplots(len(selected),1,figsize=(12,max(3,2.5*len(selected))),sharex=True)
        axes=[axes] if len(selected)==1 else axes
        report={}
        for axis,(folder,title,row) in zip(axes,selected):
            values=[row.get('per_property',{}).get(p,{}).get('accuracy',0) for p in properties]
            counts=[row.get('per_property',{}).get(p,{}).get('n',0) for p in properties]
            axis.bar(properties,values);axis.set_ylim(0,1);axis.set_ylabel('Accuracy');axis.set_title(title);axis.grid(axis='y',alpha=.2)
            for i,(value,count) in enumerate(zip(values,counts)):
                if count:axis.text(i,value+.02,f'n={count}',rotation=90,ha='center',va='bottom',fontsize=6)
            report[folder]={p:row.get('per_property',{}).get(p,{'n':0,'accuracy':None}) for p in properties}
        axes[-1].tick_params(axis='x',rotation=55);fig.tight_layout();save(fig,out,'per-property-accuracy');plt.close(fig)
        write_json(Path(out)/'per-property-summary.json',report)


def detector_curve(run_root,out):
    plt=pyplot();records=[r for r in rows(Path(run_root)/'detector'/'train.jsonl') if 'validation_loss' in r]
    if not records:return
    fig,axis=plt.subplots(figsize=(7,4.5));other=axis.twinx();steps=[r['step'] for r in records]
    axis.plot(steps,[r['validation_loss'] for r in records],color='#444',marker='o',label='validation loss')
    for key,label,color in (('detector_precision_iou50','Precision@0.5','#2864b4'),
                            ('detector_recall_iou50','Recall@0.5','#26946c'),
                            ('detector_small_recall_iou50','Small recall@0.5','#df7832')):
        if any(r.get(key) is not None for r in records):other.plot(steps,[r.get(key,float('nan')) for r in records],label=label,color=color)
    axis.set(xlabel='Optimizer step',ylabel='Validation loss');other.set_ylabel('Detection metric');other.set_ylim(0,1)
    axis.grid(alpha=.25);axis.legend(loc='upper left');other.legend(loc='lower right');fig.tight_layout();save(fig,out,'detector-validation');plt.close(fig)


def denoising_trajectory(evaluation,out):
    plt=pyplot();values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    pattern=re.compile(r'(.+)-r\d+-trace\.json$')
    for path in Path(evaluation).glob('pages/*/*-trace.json'):
        match=pattern.match(path.name)
        if not match:continue
        method=match.group(1);trace=json.loads(path.read_text())
        for state in trace.get('history',[]):
            for metric,value in state.get('diagnostic_metrics',{}).items():
                if isinstance(value,(int,float)):values[method][int(state['step'])][metric].append(float(value))
    if not values:return
    summary={};fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for method,steps in sorted(values.items()):
        summary[method]={}
        for metric,axis in (('box_iou',axes[0]),('relation_accuracy',axes[1])):
            points=[(step,statistics.mean(group[metric])) for step,group in sorted(steps.items()) if group.get(metric)]
            if points:axis.plot([p[0] for p in points],[p[1] for p in points],marker='o',label=method)
        for step,group in sorted(steps.items()):
            summary[method][str(step)]={metric:statistics.mean(items) for metric,items in group.items()}
            summary[method][str(step)]['n']=max(len(items) for items in group.values())
    for axis,title in zip(axes,('Geometry IoU','Relation accuracy')):
        axis.set(xlabel='Denoising step',ylabel=title);axis.set_ylim(0,1);axis.grid(alpha=.25);axis.legend()
    fig.tight_layout();save(fig,out,'denoising-trajectory');plt.close(fig);write_json(Path(out)/'trajectory-summary.json',summary)


def generate(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    training_curves(args.run_root,out);detector_curve(args.run_root,out);action_accuracy(args.run_root,out)
    if args.evaluation:denoising_trajectory(args.evaluation,out)
    return out

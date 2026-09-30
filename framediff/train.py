from __future__ import annotations
import contextlib
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .data import corrupt,differences,assert_disjoint
from .ir import read_jsonl,write_json
from .model import ModelConfig,EditDenoiser,encode,collate,action_index,flat_logits

def select_device(value='auto'):
    if value!='auto': return value
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def sample(record,rng,cfg,max_noise,clean_probability=.15,objective='edits',online=True):
    if 'teacher_edits' in record:
        from .plans import encode_plan
        from .ir import FIELDS
        if cfg.conditioning!='plan':raise ValueError('Browser trajectory data requires plan conditioning')
        feature=encode_plan(record['current'],record['plan'],cfg.max_nodes,record['current_boxes'])
        for j,field in enumerate(FIELDS):
            if field not in ('width','height','dx','dy'):feature['legal'][:,j,:]=False
        feature['legal'][0]=False
        return feature,record['teacher_edits'],record
    clean=record.get('clean')
    if clean is None:raise ValueError("Supervised training requires clean executable IR; raw frame observations alone are not edit labels")
    if rng.random()<clean_probability:
        current=clean
    elif online:
        current=corrupt(clean,rng,rng.randint(1,max_noise),mode=record.get('noise_mode','mixed'))
    else:current=record['current']
    diff=differences(current,clean)
    if cfg.conditioning=='plan':
        from .plans import oracle_plan,encode_plan,plan_metrics
        from .ir import execute,apply_edit
        # Fixed clean plan across the trajectory; real VLM plans may be supplied explicitly.
        obs=record['observations'][0]
        plan=record.get('plan') or oracle_plan(clean,obs['target'],obs['viewport'])
        feature=encode_plan(current,plan,cfg.max_nodes)
        score=float(feature['plan_loss'])
        if score<1e-6:diff=[]
        else:
            improving=[e for e in diff if plan_metrics(plan,execute(apply_edit(current,e),plan['viewport']))['plan_loss']<score-1e-9]
            if improving:diff=improving
    else:feature=encode(current,record['observations'],cfg.max_nodes)
    return feature,diff,record

def loss_for(model,samples,device,objective):
    batch=collate([s[0] for s in samples],device);out=model(batch)
    n=batch['props'].shape[1]
    if objective=='boxes':
        # Diagnostic one-shot box baseline only. No implied CSS or responsive support.
        g=batch['geometry'][:,0]
        mask=batch['mask'] & (g[:,:,12]>0)
        loss=torch.nn.functional.smooth_l1_loss(out['boxes'][mask],g[:,:,:4][mask]+g[:,:,8:12][mask])
        return loss,{'loss':float(loss.detach())}
    logits=flat_logits(out);logp=torch.log_softmax(logits,dim=-1)
    losses=[]
    for b,(_,diff,_) in enumerate(samples):
        valid=[action_index(e,n) for e in diff] if diff else [action_index(None,n)]
        losses.append(-torch.logsumexp(logp[b,valid],dim=0))
    policy=torch.stack(losses).mean()
    # Train value on current observable geometric error, not privileged tree distance.
    g=batch['geometry'];vmask=g[:,:,:,12]*g[:,:,:,15]*batch['mask'][:,None,:]
    target=(g[:,:,:,8:12].abs().mean(-1)*vmask).sum((1,2))/vmask.sum((1,2)).clamp_min(1)
    if model.cfg.conditioning=='plan':target=batch['plan_loss']
    value=torch.nn.functional.smooth_l1_loss(out['value'],target)
    loss=policy+.1*value
    return loss,{'loss':float(loss.detach()),'policy':float(policy.detach()),'value':float(value.detach())}

def train(args):
    device=select_device(args.device);seed_all(args.seed)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    records=list(read_jsonl(args.train));val=list(read_jsonl(args.val))
    if not records or not val:raise ValueError("Both train and validation sets must be nonempty")
    assert_disjoint(records,val)
    cfg=ModelConfig(args.hidden,args.layers,args.heads,args.dropout,not args.no_tree_bias,args.max_nodes,getattr(args,'conditioning','boxes'))
    if cfg.conditioning=='plan' and args.objective!='edits':raise ValueError('Plan conditioning requires edits objective')
    model=EditDenoiser(cfg).to(device)
    training_groups={r['group'] for r in records}
    if getattr(args,'init_checkpoint',None):
        if args.resume:raise ValueError('Choose init-checkpoint OR resume')
        ck=torch.load(args.init_checkpoint,map_location=device,weights_only=True)
        if asdict(ModelConfig(**ck['config']))!=asdict(cfg):raise ValueError('Initialization configuration mismatch')
        model.load_state_dict(ck['model']);training_groups.update(ck.get('training_groups',[]))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    start=0;best=float('inf');rng=random.Random(args.seed)
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=True)
        if asdict(ModelConfig(**ck['config']))!=asdict(cfg) or ck['objective']!=args.objective:raise ValueError("Resume configuration mismatch")
        training_groups.update(ck.get('training_groups',[]))
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start=ck['step'];best=ck['best'];rng.setstate(ck['rng']);torch.set_rng_state(ck['torch_rng'].cpu())
        if device.startswith('cuda') and ck.get('cuda_rng'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
    if training_groups & {r['group'] for r in val}:raise ValueError('Validation overlaps historical checkpoint training groups')
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    fixed_rng=random.Random(90210)
    val_samples=[sample(r,fixed_rng,cfg,args.max_noise,online=False) for r in val[:args.val_samples]]
    manifest={**vars(args),'device_actual':device,'torch':torch.__version__,'parameters':sum(p.numel() for p in model.parameters()),
              'train_sha256':hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
              'val_sha256':hashlib.sha256(Path(args.val).read_bytes()).hexdigest(),'config':asdict(cfg)}
    manifest.pop('func',None)
    write_json(output/'manifest.json',manifest)
    def amp():
        return torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') and args.bf16 else contextlib.nullcontext()
    t0=time.perf_counter()
    for step in range(start+1,args.steps+1):
        model.train();opt.zero_grad(set_to_none=True);stats={}
        for _ in range(args.accumulation):
            samples=[sample(rng.choice(records),rng,cfg,args.max_noise,objective=args.objective,online=not args.fixed_pairs) for _ in range(args.batch_size)]
            with amp():loss,stats=loss_for(model,samples,device,args.objective)
            (loss/args.accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step%args.log_every==0 or step==args.steps:
            row={'step':step,**stats,'elapsed_s':time.perf_counter()-t0}
            with (output/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            model.eval();losses=[]
            with torch.no_grad():
                for offset in range(0,len(val_samples),args.batch_size):
                    with amp():loss,_=loss_for(model,val_samples[offset:offset+args.batch_size],device,args.objective)
                    losses.append((float(loss),len(val_samples[offset:offset+args.batch_size])))
            validation=sum(a*b for a,b in losses)/sum(b for _,b in losses)
            improved=validation<best;best=min(best,validation)
            ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'config':asdict(cfg),'step':step,'best':best,
                'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                'objective':args.objective,'training_groups':sorted(training_groups)}
            torch.save(ck,output/'last.pt')
            if improved:torch.save(ck,output/'best.pt')
            print(json.dumps({'step':step,'validation_loss':validation,'best':best}),flush=True)
    return manifest

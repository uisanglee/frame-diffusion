"""Train the screenshot parser and the two image-conditioned edit policies."""
import contextlib
from dataclasses import asdict
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .ir import read_jsonl,write_json,write_jsonl
from .train import select_device,seed_all
from .visual import CONTRACT,CLASSES,VisualConfig,VisualPolicy,current_features,visual_batch,image_tensor,action_index,abstract_image
from .visual_data import validate_splits
from .web_experiment import digest,guard_run


def detector(config,pretrained=True):
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    backbone=resnet_fpn_backbone(backbone_name='resnet50',weights=ResNet50_Weights.DEFAULT if pretrained else None,
                                trainable_layers=3)
    return FasterRCNN(backbone,num_classes=len(CLASSES),min_size=config['min_size'],max_size=config['max_size'],
        box_detections_per_img=config['max_detections'])


def load_detector(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('kind')!='visual-detector-v1' or ck['config']['contract']!=CONTRACT:raise ValueError('Detector checkpoint contract mismatch')
    net=detector(ck['config'],False);net.load_state_dict(ck['model'])
    return net.to(device).eval(),ck


def detector_input(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255


def detect(net,image,threshold=.4):
    device=next(net.parameters()).device
    with torch.inference_mode():prediction=net([image.to(device)])[0]
    return [{'box':box,'label':int(label),'score':float(score)}
            for box,label,score in zip(prediction['boxes'].cpu().tolist(),prediction['labels'].cpu().tolist(),prediction['scores'].cpu().tolist())
            if score>=threshold and 0<label<len(CLASSES)]


def detection_loss(net,rows,device):
    images=[detector_input(r['image']).to(device) for r in rows]
    labels=[{'boxes':torch.tensor([e['box'] for e in r['elements']],dtype=torch.float32,device=device).reshape(-1,4),
             'labels':torch.tensor([e['label'] for e in r['elements']],dtype=torch.int64,device=device)} for r in rows]
    losses=net(images,labels)
    return sum(losses.values())


def policy_loss(net,rows,device,rng,prediction_probability):
    cfg=net.cfg
    batch=visual_batch([current_features(r['current'],r['current_boxes'],r['viewport'],cfg.max_nodes) for r in rows],device)
    target=[];current=[]
    for row in rows:
        if cfg.mode=='screenshot':tp,cp=row['target_image'],row['current_image']
        else:
            use_prediction=rng.random()<prediction_probability
            if use_prediction and not row.get('predicted_target_abstract'):raise ValueError('Predicted targets missing; run visual-cache-targets first')
            tp=row['predicted_target_abstract'] if use_prediction else row['target_abstract']
            cp=row['current_abstract']
        target.append(image_tensor(tp,cfg.size));current.append(image_tensor(cp,cfg.size))
    logits=net(batch,torch.stack(target).to(device),torch.stack(current).to(device))
    logp=F.log_softmax(logits,dim=-1);n=batch['mask'].shape[1];losses=[]
    for i,row in enumerate(rows):
        valid=[action_index(e,n) for e in row['teacher_edits']] or [action_index(None,n)]
        if not torch.isfinite(logits[i,valid]).all():raise ValueError('Illegal training edit')
        losses.append(-torch.logsumexp(logp[i,valid],0))
    return torch.stack(losses).mean()


def train(args):
    device=select_device(args.device);seed_all(args.seed)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    train_rows=list(read_jsonl(args.train));val_rows=list(read_jsonl(args.val))
    if not train_rows or not val_rows:raise ValueError('Nonempty train and validation data required')
    if any(r['split']!='train' for r in train_rows) or any(r['split']!='val' for r in val_rows):raise ValueError('Use explicit train/val splits')
    validate_splits(train_rows+val_rows)
    if any(r.get('contract')!=CONTRACT for r in train_rows+val_rows):raise ValueError('Dataset contract mismatch')
    if not 0<=args.prediction_probability<=1:raise ValueError('Prediction probability must be in [0,1]')
    is_detector=args.command=='visual-train-detector'
    if is_detector:
        cfg={'contract':CONTRACT,'min_size':args.min_size,'max_size':args.max_size,'max_detections':args.max_detections}
        if cfg['min_size']<32 or cfg['max_size']<cfg['min_size'] or cfg['max_detections']<1:raise ValueError('Invalid detector dimensions')
        kind='visual-detector-v1'
    else:
        cfg=asdict(VisualConfig(mode=args.mode,size=args.size,hidden=args.hidden,layers=args.layers,
                                heads=args.heads,max_nodes=args.max_nodes,token_grid=args.token_grid));kind='visual-policy-v1'
    if args.resume and args.init_checkpoint:raise ValueError('Choose resume or init-checkpoint')
    data_signature={'train':digest(args.train),'val':digest(args.val)}
    out=Path(args.out).resolve()
    run_settings={k:v for k,v in vars(args).items() if k not in ('out','resume','steps','init_checkpoint')}
    guard_run(out,{'kind':kind,'config':cfg,'data':data_signature,'settings':run_settings},bool(args.resume))
    ck=torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=True) if args.resume or args.init_checkpoint else None
    if ck and (ck.get('kind')!=kind or ck['config']!=cfg):raise ValueError('Checkpoint architecture/mode mismatch')
    net=(detector(cfg,not args.no_pretrained and ck is None) if is_detector else
         VisualPolicy(VisualConfig(**cfg),not args.no_pretrained and ck is None)).to(device)
    groups={r['group'] for r in train_rows};source_hashes={r['source_sha'] for r in train_rows}
    if ck:
        net.load_state_dict(ck['model']);groups.update(ck.get('training_groups',[]));source_hashes.update(ck.get('training_hashes',[]))
    if groups & {r['group'] for r in val_rows} or source_hashes & {r['source_sha'] for r in val_rows}:raise ValueError('Historical training/validation leakage')
    # Frozen parser provenance follows the policy into downstream evaluation.
    parser_groups=set(ck.get('parser_training_groups',[])) if ck else set()
    parser_hashes=set(ck.get('parser_training_hashes',[])) if ck else set()
    if not is_detector and args.mode=='abstract' and args.prediction_probability>0:
        provenance_cache={}
        for r in train_rows+val_rows:
            if not r.get('predicted_target_abstract'):raise ValueError('All rows require cached predicted target abstractions')
            provenance=r
            if r.get('parser_provenance'):
                from .ir import read_json
                path=r['parser_provenance']
                if path not in provenance_cache:provenance_cache[path]=read_json(path)
                provenance=provenance_cache[path]
            parser_groups.update(provenance.get('parser_training_groups',[]));parser_hashes.update(provenance.get('parser_training_hashes',[]))
        if parser_groups & {r['group'] for r in val_rows} or parser_hashes & {r['source_sha'] for r in val_rows}:raise ValueError('Parser leaked validation pages')
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01)
    rng=random.Random(args.seed);start=0;best=float('inf')
    if args.resume:
        if ck['data_signature']!=data_signature:raise ValueError('Resume dataset changed; use init-checkpoint')
        opt.load_state_dict(ck['optimizer']);start=ck['step'];best=ck['best'];rng.setstate(ck['rng'])
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.startswith('cuda') and ck.get('cuda_rng'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
    write_json(out/'manifest.json',{'kind':kind,'config':cfg,'settings':vars(args),'data':data_signature,
        'parameters':sum(p.numel() for p in net.parameters()),'pretrained_backbone':not args.no_pretrained,
        'torch':str(torch.__version__)})
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') and args.bf16 and not is_detector else contextlib.nullcontext()
    def loss(rows,rr):
        return detection_loss(net,rows,device) if is_detector else policy_loss(net,rows,device,rr,args.prediction_probability)
    started=time.perf_counter()
    for step in range(start+1,args.steps+1):
        net.train();opt.zero_grad(set_to_none=True);total=0.
        for _ in range(args.accumulation):
            batch=[rng.choice(train_rows) for _ in range(args.batch_size)]
            with amp():value=loss(batch,rng)
            if not torch.isfinite(value):raise RuntimeError('Non-finite training loss')
            (value/args.accumulation).backward();total+=float(value.detach())/args.accumulation
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step%args.log_every==0 or step==args.steps:
            row={'step':step,'loss':total,'elapsed_s':time.perf_counter()-started}
            with (out/'train.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps(row)+'\n')
            print(row,flush=True)
        if step%args.eval_every==0 or step==args.steps:
            # Torchvision detection losses require train mode. FrozenBatchNorm
            # avoids changing running statistics; fork_rng makes RPN sampling repeatable.
            net.train(is_detector);values=[];vrng=random.Random(90210)
            selected=val_rows[:args.val_samples]
            with torch.random.fork_rng(),torch.no_grad():
                torch.manual_seed(90210)
                for offset in range(0,len(selected),args.batch_size):
                    rows=selected[offset:offset+args.batch_size]
                    with amp():value=loss(rows,vrng)
                    values.append((float(value),len(rows)))
            validation=sum(v*n for v,n in values)/sum(n for _,n in values)
            if not math_isfinite(validation):raise RuntimeError('Non-finite validation loss')
            improved=validation<best;best=min(best,validation)
            checkpoint={'kind':kind,'config':cfg,'model':net.state_dict(),'optimizer':opt.state_dict(),
                'step':step,'best':best,'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                'training_groups':sorted(groups),'training_hashes':sorted(source_hashes),
                'parser_training_groups':sorted(parser_groups),'parser_training_hashes':sorted(parser_hashes),
                'data_signature':data_signature}
            torch.save(checkpoint,out/'last.pt')
            if improved:torch.save(checkpoint,out/'best.pt')
            print({'step':step,'validation_loss':validation,'best':best},flush=True)


def math_isfinite(value):
    import math
    return math.isfinite(value)


def cache_targets(args):
    device=select_device(args.device)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    model,ck=load_detector(args.checkpoint,device);rows=list(read_jsonl(args.data))
    out=Path(args.out).resolve();guard_run(out,{'kind':'visual-cache-v1','data':digest(args.data),
        'checkpoint':digest(args.checkpoint),'threshold':args.threshold,'size':args.size},args.resume)
    provenance=out/'parser-provenance.json';parser_sha=digest(args.checkpoint)
    write_json(provenance,{'parser_sha':parser_sha,'parser_training_groups':ck['training_groups'],'parser_training_hashes':ck['training_hashes']})
    seen={};result=[]
    for r in rows:
        if r['split']!='train' and (r['group'] in ck['training_groups'] or r['source_sha'] in ck['training_hashes']):
            raise ValueError('Parser training overlaps held-out target')
        key=digest(r['target_image']);path=out/f'{key}.png'
        if key not in seen:
            if not (args.resume and path.exists()):
                predictions=detect(model,detector_input(r['target_image']),args.threshold)
                abstract_image(predictions,r['viewport'],args.size).save(path)
                write_json(out/f'{key}.json',predictions)
            seen[key]=str(path)
        result.append({**r,'predicted_target_abstract':seen[key],'parser_sha':parser_sha,'parser_provenance':str(provenance)})
    write_jsonl(out/'data.jsonl',result)
    return result


def evaluate_detector(args):
    from scipy.optimize import linear_sum_assignment
    from torchvision.ops import box_iou
    device=select_device(args.device)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    net,ck=load_detector(args.checkpoint,device);rows=list(read_jsonl(args.data))
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('No detector evaluation samples')
    tp=predicted=truth=small_tp=small_n=0;results=[]
    for r in rows:
        if r['group'] in ck['training_groups'] or r['source_sha'] in ck['training_hashes']:raise ValueError('Detector evaluation leakage')
        pred=detect(net,detector_input(r['image']),args.threshold);target=r['elements']
        hits=set();count=0
        if pred and target:
            overlaps=box_iou(torch.tensor([e['box'] for e in pred]),torch.tensor([e['box'] for e in target])).numpy()
            valid=np.array([[p['label']==t['label'] for t in target] for p in pred]) & (overlaps>=.5)
            # Maximize number of threshold-valid, class-matched pairs first.
            ii,jj=linear_sum_assignment(-(valid*10+overlaps*valid))
            hits={int(j) for i,j in zip(ii,jj) if valid[i,j]};count=len(hits)
        small={i for i,e in enumerate(target) if (e['box'][2]-e['box'][0])*(e['box'][3]-e['box'][1])<32**2}
        tp+=count;predicted+=len(pred);truth+=len(target);small_tp+=len(hits&small);small_n+=len(small)
        results.append({'id':r['id'],'matched_iou50':count,'predictions':len(pred),'targets':len(target),'small_targets':len(small)})
    report={'n':len(rows),'precision_iou50':tp/predicted if predicted else 0.,'recall_iou50':tp/truth if truth else None,
        'small_recall_iou50':small_tp/small_n if small_n else None,'small_n':small_n,'threshold':args.threshold,
        'note':'Class-aware matching at IoU 0.5, not COCO AP','records':results}
    write_json(args.out,report);return report

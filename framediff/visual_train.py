"""Detector training and frozen target predictions; policies use tree_policy.py."""
import contextlib
import io
from collections import defaultdict
from pathlib import Path
import random
import math
import time

import numpy as np
from PIL import Image
import torch

from .ir import read_jsonl,write_json,write_jsonl
from .train import select_device,seed_all
from .visual import CONTRACT,CLASSES,abstract_image
from .visual_data import validate_splits
from .web_experiment import digest,guard_run
from .training_shutdown import cleanup_on_termination


def guard_detector_resume(out, config, resume):
    """Allow microbatch changes only at the same effective batch size."""
    import copy
    import json
    current=copy.deepcopy(config)
    settings=current['settings']
    settings['validation_batch_size']=settings['batch_size']
    path=out/'config.json'
    if resume and path.exists():
        saved=json.loads(path.read_text())
        previous=saved['settings']
        settings['validation_batch_size']=previous.get('validation_batch_size',previous['batch_size'])
        comparable=copy.deepcopy(current)
        comparable['settings'].pop('validation_batch_size')
        old=copy.deepcopy(saved)
        old['settings'].pop('validation_batch_size',None)
        if previous['batch_size']*previous['accumulation']!=settings['batch_size']*settings['accumulation']:
            raise ValueError('Detector resume requires unchanged batch_size * accumulation')
        for key in ('batch_size','accumulation'):
            comparable['settings'][key]=previous[key]
        if comparable!=old:
            raise ValueError('Resume configuration/input changed; use a new output directory')
        guard_run(out,saved,True)
        write_json(path,current)
    else:
        guard_run(out,current,resume)
    return settings['validation_batch_size']


def detector(config,pretrained=True):
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    backbone=resnet_fpn_backbone(backbone_name='resnet50',weights=ResNet50_Weights.DEFAULT if pretrained else None,
                                trainable_layers=3)
    net=FasterRCNN(backbone,num_classes=len(CLASSES),min_size=config['min_size'],max_size=config['max_size'],
        box_detections_per_img=config['max_detections'])
    from .detector_support import BoundedRegionProposalNetwork
    net.rpn=BoundedRegionProposalNetwork.from_rpn(net.rpn)
    return net


def load_detector(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('kind')!='visual-detector-v1' or ck['config']['contract']!=CONTRACT:raise ValueError('Detector checkpoint contract mismatch')
    net=detector(ck['config'],False);net.load_state_dict(ck['model'])
    return net.to(device).eval(),ck


def detector_input(path):
    with Image.open(io.BytesIO(path) if isinstance(path,bytes) else path) as image:
        return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255


def detect(net,image,threshold=.4):
    return detect_batch(net,[image],threshold)[0]


def detect_batch(net,images,threshold=.4):
    device=next(net.parameters()).device
    with torch.inference_mode():predictions=net([image.to(device) for image in images])
    return [[{'box':box,'label':int(label),'score':float(score)}
            for box,label,score in zip(prediction['boxes'].cpu().tolist(),prediction['labels'].cpu().tolist(),prediction['scores'].cpu().tolist())
            if score>=threshold and 0<label<len(CLASSES)] for prediction in predictions]


def detection_losses(net,rows,device):
    from .detector_support import prepare_detector_batch
    rows=[r if '_detector_pixels' in r else prepare_detector_batch([r])[0] for r in rows]
    images=[];labels=[]
    for row in rows:
        pixels=row['_detector_pixels'];target=row['_detector_target']
        images.append(pixels.to(device,non_blocking=True).float()/255)
        labels.append({k:v.to(device,non_blocking=True) for k,v in target.items()})
    return net(images,labels)


def detection_loss(net,rows,device):return sum(detection_losses(net,rows,device).values())


def page_view_pools(rows):
    """Index once, outside the optimizer loop (O(corpus size))."""
    pages=defaultdict(list)
    for row in rows:
        page=row.get('page_id') or (row.get('id') or row['source_sha']).rsplit('/',1)[0]
        pages[page].append(row)
    return tuple(pages[key] for key in sorted(pages))


def sample_page_views(pools,batch_size,rng):
    """O(batch size): uniform page, then uniform available view."""
    return [rng.choice(rng.choice(pools)) for _ in range(batch_size)]


def detector_quality(net,rows,device,threshold=.4,workers=0,prefetch=2):
    from .detector_support import Aggregate, compare, detector_batches, label_coverage
    net.eval();bucket=Aggregate()
    for batch in detector_batches(rows,1,workers,prefetch,str(device).startswith('cuda')):
        row=batch[0]
        pixels=row.get('_detector_pixels')
        image=(pixels.to(device,non_blocking=True).float()/255 if pixels is not None else detector_input(row['image']))
        pred=detect(net,image,threshold)
        target=row['elements']
        bucket.add(row,pred,target,compare(pred,target))
    report=bucket.report()
    return {'detector_precision_iou50':report['precision_iou50'],
            'detector_recall_iou50':report['recall_iou50'],
            'detector_small_recall_iou50':report['small_recall_iou50'],
            'detector_predictions':report['predictions'],'detector_targets':report['targets'],
            'detector_small_n':report['small_n'],'detector_threshold':threshold,
            'detector_by_class':report['by_class'],'detector_by_view':report['by_view'],
            'detector_by_device':report['by_device'],'detector_prediction_errors':report['prediction_errors'],
            'detector_label_coverage':label_coverage(rows)}


@cleanup_on_termination
def train(args):
    with contextlib.ExitStack() as stack:
        return _train(args,stack)


def _train(args,stack):
    patience=args.early_stop_patience;min_delta=args.early_stop_min_delta
    if patience<0 or not math.isfinite(min_delta) or min_delta<0:
        raise ValueError('Early-stop patience and min-delta must be finite and nonnegative')
    device=select_device(args.device);seed_all(args.seed)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    train_rows=list(read_jsonl(args.train));val_rows=list(read_jsonl(args.val))
    if not train_rows or not val_rows:raise ValueError('Nonempty train and validation data required')
    if any(r['split']!='train' for r in train_rows) or any(r['split']!='val' for r in val_rows):raise ValueError('Use explicit train/val splits')
    validate_splits(train_rows+val_rows)
    if any(r.get('contract')!=CONTRACT for r in train_rows+val_rows):raise ValueError('Dataset contract mismatch')
    if not 0<=args.prediction_probability<=1:raise ValueError('Prediction probability must be in [0,1]')
    loader_workers=getattr(args,'loader_workers',2);loader_prefetch=getattr(args,'loader_prefetch',2)
    if loader_workers<0 or loader_prefetch<1:raise ValueError('Invalid detector loader settings')
    cfg={'contract':CONTRACT,'min_size':args.min_size,'max_size':args.max_size,'max_detections':args.max_detections}
    from .visual import label_contract
    geometry_contract=label_contract(train_rows+val_rows)
    if geometry_contract:cfg['visual_geometry_contract']=geometry_contract
    coverage={r.get('detector_label_contract') for r in train_rows+val_rows}
    if len(coverage)>1:raise ValueError('Mixed detector label coverage versions; regenerate matching train/val labels')
    if next(iter(coverage)):cfg['detector_label_contract']=next(iter(coverage))
    if cfg['min_size']<32 or cfg['max_size']<cfg['min_size'] or cfg['max_detections']<1:raise ValueError('Invalid detector dimensions')
    kind='visual-detector-v1'
    if args.resume and args.init_checkpoint:raise ValueError('Choose resume or init-checkpoint')
    data_signature={'train':digest(args.train),'val':digest(args.val)}
    out=Path(args.out).resolve()
    run_settings={k:v for k,v in vars(args).items() if k not in
                  ('out','resume','steps','init_checkpoint','render_workers','render_prefetch','render_timeout',
                   'loader_workers','loader_prefetch')}
    run_settings['validation_sampling']='page-round-robin-v1'
    run_config={'kind':kind,'config':cfg,'data':data_signature,'settings':run_settings}
    validation_batch_size=guard_detector_resume(out,run_config,bool(args.resume))
    ck=torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=True) if args.resume or args.init_checkpoint else None
    if ck and (ck.get('kind')!=kind or ck['config']!=cfg):raise ValueError('Checkpoint architecture/mode mismatch')
    if args.resume and ck.get('early_stopping',{}).get('stopped',False):
        print(f'Early stopping already completed at step {ck["step"]}; keeping best.pt and skipping stage.',flush=True)
        return
    net=detector(cfg,not args.no_pretrained and ck is None).to(device)
    groups={r['group'] for r in train_rows};source_hashes={r['source_sha'] for r in train_rows}
    if ck:
        net.load_state_dict(ck['model'])
        groups.update(ck.get('training_groups',[]));source_hashes.update(ck.get('training_hashes',[]))
    if groups & {r['group'] for r in val_rows} or source_hashes & {r['source_sha'] for r in val_rows}:raise ValueError('Historical training/validation leakage')
    # Preserve historical provenance when resuming an existing detector.
    parser_groups=set(ck.get('parser_training_groups',[])) if ck else set()
    parser_hashes=set(ck.get('parser_training_hashes',[])) if ck else set()
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01)
    view_pools=page_view_pools(train_rows)
    from .lazy_webui import TrainViewCache,portable_images
    lazy_cache=TrainViewCache(cache_dir=Path(args.train).resolve().parent/'.lazy-detector-cache')
    stack.callback(lazy_cache.close)
    rng=random.Random(args.seed);start=0;best=float('inf')
    early={'reference_loss':float('inf'),'bad_validations':0,'stopped':False}
    if args.resume:
        if ck['data_signature']!=data_signature:raise ValueError('Resume dataset changed; use init-checkpoint')
        opt.load_state_dict(ck['optimizer']);start=ck['step'];best=ck['best'];rng.setstate(ck['rng'])
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.startswith('cuda') and ck.get('cuda_rng'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
        early=ck.get('early_stopping',{'reference_loss':best,'bad_validations':0,'stopped':False})
    del ck
    write_json(out/'manifest.json',{'kind':kind,'config':cfg,'settings':vars(args),'data':data_signature,
        'parameters':sum(p.numel() for p in net.parameters()),'pretrained_backbone':not args.no_pretrained,
        'torch':str(torch.__version__)})
    started=time.perf_counter()
    render_stream=None
    if any(row.get('lazy_contract') for row in train_rows) and start<args.steps and not early['stopped']:
        from .detector_support import DetectorStream
        render_stream=stack.enter_context(DetectorStream(view_pools,rng,args.batch_size,
            Path(args.train).resolve().parent/'.lazy-detector-cache',out/'lazy-rejections.jsonl',
            getattr(args,'render_workers',2),getattr(args,'render_prefetch',4),getattr(args,'render_timeout',900.),
            decode=bool(loader_workers)))
    elif loader_workers and start<args.steps and not early['stopped']:
        from .detector_support import DetectorLoader
        render_stream=DetectorLoader(view_pools,rng,args.batch_size,loader_workers,loader_prefetch,
                                     device.startswith('cuda'))
        stack.callback(render_stream.close)
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for step in range(start+1,args.steps+1):
        net.train();opt.zero_grad(set_to_none=True);total=0.
        for _ in range(args.accumulation):
            batch=render_stream.take() if render_stream else sample_page_views(view_pools,args.batch_size,rng)
            if render_stream is None:
                materialized=[]
                for row in batch:
                    if row.get('lazy_contract'):
                        for attempt in range(32):
                            try:
                                row=portable_images(lazy_cache.get(row),('image',));break
                            except ValueError as exc:
                                with (out/'lazy-rejections.jsonl').open('a') as log:
                                    import json
                                    log.write(json.dumps({'id':row['id'],'error':str(exc)})+'\n')
                                row=sample_page_views(view_pools,1,rng)[0]
                        else:raise RuntimeError('32 consecutive lazy detector preparation failures; inspect lazy-rejections.jsonl')
                    materialized.append(row)
                batch=materialized
            value=detection_loss(net,batch,device)
            if not torch.isfinite(value):raise RuntimeError('Non-finite training loss')
            (value/args.accumulation).backward()
            total+=value.detach()/args.accumulation
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step%args.log_every==0 or step==args.steps:
            elapsed=time.perf_counter()-started
            row={'step':step,'loss':float(total),'elapsed_s':elapsed}
            if device.startswith('cuda'):
                row['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(device)
            row['images_per_second']=(step-start)*args.batch_size*args.accumulation/max(elapsed,1e-9)
            with (out/'train.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps(row)+'\n')
            print(row,flush=True)
        if step%args.eval_every==0 or step==args.steps:
            # Torchvision detection losses require train mode. FrozenBatchNorm
            # avoids changing running statistics; fork_rng makes RPN sampling repeatable.
            net.train();values=[];components={}
            selected=validation_sample(val_rows,args.val_samples)
            with torch.random.fork_rng(),torch.no_grad():
                torch.manual_seed(90210)
                from .detector_support import detector_batches
                batches=detector_batches(selected,validation_batch_size,loader_workers,loader_prefetch,device.startswith('cuda'))
                for rows in batches:
                    parts=detection_losses(net,rows,device);value=sum(parts.values())
                    for key,part in parts.items():components[key]=components.get(key,0.)+float(part)*len(rows)
                    values.append((float(value),len(rows)))
            validation=sum(v*n for v,n in values)/sum(n for _,n in values)
            if not math.isfinite(validation):raise RuntimeError('Non-finite validation loss')
            eval_row={'step':step,'validation_loss':validation,'validation_n':len(selected),
                      'validation_pages':len({r.get('page_id',r.get('target_image',r['source_sha'])) for r in selected}),
                      'validation_available_n':len(val_rows),'validation_complete':len(selected)==len(val_rows)}
            eval_row['validation_components']={k:v/len(selected) for k,v in sorted(components.items())}
            with torch.inference_mode():eval_row.update(detector_quality(net,selected,device,args.metric_threshold,
                                                                        loader_workers,loader_prefetch))
            improved=validation<best;best=min(best,validation)
            early=update_early_stopping(early,validation,patience,min_delta)
            eval_row['best']=best;eval_row['improved']=improved
            eval_row['early_stopping']=early
            checkpoint={'kind':kind,'config':cfg,'model':net.state_dict(),'optimizer':opt.state_dict(),
                'step':step,'best':best,'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                'training_groups':sorted(groups),'training_hashes':sorted(source_hashes),
                'parser_training_groups':sorted(parser_groups),'parser_training_hashes':sorted(parser_hashes),
                'data_signature':data_signature,'early_stopping':early}
            torch.save(checkpoint,out/'last.pt')
            if improved:torch.save(checkpoint,out/'best.pt')
            with (out/'train.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps(eval_row)+'\n')
            print(eval_row,flush=True)
            if early['stopped']:
                print(f'Early stopping at step {step}: {patience} validations without sufficient improvement; best loss={best}.',flush=True)
                break


def update_early_stopping(state,validation,patience,min_delta):
    improved=validation<state['reference_loss']-min_delta
    bad=0 if improved else state['bad_validations']+1
    return {'reference_loss':validation if improved else state['reference_loss'],
            'bad_validations':bad,'stopped':bool(patience and bad>=patience)}


def validation_sample(rows,limit,seed=90210):
    """Fixed round-robin sample across pages, not the first few trajectories."""
    if limit<0:raise ValueError('Validation limit must be nonnegative; 0 means all')
    if limit==0 or limit>=len(rows):return list(rows)
    rng=random.Random(seed);pages=defaultdict(list)
    for row in rows:pages[row.get('page_id',row.get('target_image',row['source_sha']))].append(row)
    groups=list(pages.values());rng.shuffle(groups)
    for group in groups:rng.shuffle(group)
    selected=[];depth=0
    while len(selected)<limit:
        for group in groups:
            if depth<len(group):selected.append(group[depth])
            if len(selected)==limit:break
        depth+=1
    return selected


def cache_targets(args):
    from .ir import read_json
    device=select_device(args.device)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    rows=list(read_jsonl(args.data))
    lazy=bool(rows) and all(r.get('lazy_contract') for r in rows)
    if lazy:
        model=None;ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    else:model,ck=load_detector(args.checkpoint,device)
    from .visual import label_contract
    if label_contract(rows)!=ck.get('config',{}).get('visual_geometry_contract'):
        raise ValueError('Detector and target visual label versions differ')
    out=Path(args.out).resolve();guard_run(out,{'kind':'visual-cache-v1','data':digest(args.data),
        'checkpoint':digest(args.checkpoint),'threshold':args.threshold,'size':args.size},args.resume)
    provenance=out/'parser-provenance.json';parser_sha=digest(args.checkpoint)
    reuse=Path(args.reuse_cache).resolve() if getattr(args,'reuse_cache',None) else None
    if reuse:
        config=read_json(reuse/'config.json')
        if any(config.get(k)!=v for k,v in dict(kind='visual-cache-v1',checkpoint=parser_sha,
                                              threshold=args.threshold,size=args.size).items()):
            raise ValueError('Reuse cache detector/threshold/size mismatch')
    write_json(provenance,{'parser_sha':parser_sha,'parser_training_groups':ck['training_groups'],
                          'parser_training_hashes':ck['training_hashes'],
                          'visual_geometry_contract':ck.get('config',{}).get('visual_geometry_contract')})
    seen={};result=[]
    for r in rows:
        if r['split']!='train' and (r['group'] in ck['training_groups'] or r['source_sha'] in ck['training_hashes']):
            raise ValueError('Parser training overlaps held-out target')
        if r.get('lazy_contract'):
            result.append({**r,'parser_provenance':str(provenance),'lazy_detector':{
                'checkpoint':str(Path(args.checkpoint).resolve()),'sha':parser_sha,'threshold':args.threshold}})
            continue
        key=digest(r['target_image']);path=out/f'{key}.png'
        if key not in seen:
            if reuse and (reuse/f'{key}.png').exists() and (reuse/f'{key}.json').exists():
                # References preserve existing detector outputs without a GPU
                # forward pass or duplicated image files.
                path=reuse/f'{key}.png'
            cached=path.exists() and path.with_suffix('.json').exists()
            if not (cached and (args.resume or path.parent==reuse)):
                predictions=detect(model,detector_input(r['target_image']),args.threshold)
                abstract_image(predictions,r['viewport'],args.size).save(path)
                write_json(out/f'{key}.json',predictions)
            seen[key]=str(path)
        result.append({**r,'predicted_target_abstract':seen[key],
            'predicted_target_elements':str(Path(seen[key]).with_suffix('.json')),
            'parser_sha':parser_sha,'parser_provenance':str(provenance)})
    write_jsonl(out/'data.jsonl',result)
    return result


def evaluate_detector(args):
    from .detector_support import Aggregate, compare, visualize, label_coverage
    thresholds=sorted(set([args.threshold]+list(getattr(args,'thresholds',[.3,.4,.5,.6,.7]))))
    if any(not 0 <= t <= 1 for t in thresholds):raise ValueError('thresholds must be in [0,1]')
    if args.limit < 0 or getattr(args,'visualize_limit',30) < 0:raise ValueError('limits must be nonnegative')
    device=select_device(args.device)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    net,ck=load_detector(args.checkpoint,device);rows=list(read_jsonl(args.data))
    from .visual import label_contract
    if label_contract(rows)!=ck.get('config',{}).get('visual_geometry_contract'):
        raise ValueError('Detector and evaluation visual label versions differ; use matching data/checkpoint')
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('No detector evaluation samples')
    # Check the full selected manifest before producing any evaluation artifacts.
    for row in rows:
        if row['group'] in ck['training_groups'] or row['source_sha'] in ck['training_hashes']:
            raise ValueError('Detector evaluation leakage')
    # Do not silently truncate a requested low-score sweep at torchvision's internal floor.
    net.roi_heads.score_thresh=min(net.roi_heads.score_thresh,min(thresholds))
    buckets={t:Aggregate() for t in thresholds}
    for index,row in enumerate(rows):
        predictions=detect(net,detector_input(row['image']),min(thresholds))
        for threshold,bucket in buckets.items():
            pred=[e for e in predictions if e['score']>=threshold]
            detail=compare(pred,row['elements'])
            bucket.add(row,pred,row['elements'],detail)
            bucket.records[-1]['label_audit']=row.get('label_audit')
            if threshold==args.threshold and getattr(args,'visualize_dir',None) and index<getattr(args,'visualize_limit',30):
                path=Path(args.visualize_dir)/f'{index:06d}.png'
                visualize(path,row['image'],pred,row['elements'],detail,row.get('label_audit'))
                bucket.records[-1]['visualization']=str(path)
        if (index+1)%50==0 or index+1==len(rows):
            print({'detector_diagnostics':index+1,'total':len(rows)},flush=True)
    primary=buckets[args.threshold]
    report={**primary.report(),'threshold':args.threshold,'records':primary.records,
            'label_coverage':label_coverage(rows),
            'training_label_contract':ck.get('config',{}).get('detector_label_contract'),
            'visual_geometry_contract':ck.get('config',{}).get('visual_geometry_contract'),
            'threshold_sweep':{str(t):bucket.report() for t,bucket in buckets.items()},
            'checkpoint':str(args.checkpoint),'data':str(args.data),
            'note':'Class-aware maximum-cardinality one-to-one IoU50 matching, not COCO AP. '
                   'Small means original-image box area < 32^2. Sweep uses the same post-NMS predictions '
                   'and checkpoint max_detections cap. Tune thresholds on validation, not test. '
                   'False positives are relative to supplied DOM labels, not proof that no real visual element exists. '
                   'Inspect label_coverage before attributing unmatched predictions to model error.',
            'error_rules':{'duplicate':'Unmatched prediction with same-class IoU >= 0.5',
                           'class_confusion':'Otherwise, different-class IoU >= 0.5',
                           'localization':'Otherwise, same-class 0.1 <= IoU < 0.5',
                           'unmatched':'None of the above; geometric heuristic, not a proven cause'}}
    write_json(args.out,report)
    return report

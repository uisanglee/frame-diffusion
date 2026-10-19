"""Detector training/cache and historical flat/hierarchical baseline support.

The current replacement policy is trained exclusively in tree_policy.py.
"""
import contextlib
import io
from collections import defaultdict
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
from .visual import (CONTRACT,ACTION_CONTRACT,CLASSES,VisualConfig,VisualPolicy,
                     current_features,visual_batch,image_tensor,action_index,decode_action,abstract_image,
                     action_is_legal,NUMERIC_FIELDS,NORMALIZED_DELTAS,semantic_masks,elements)
from .visual_data import validate_splits
from .web_experiment import digest,guard_run


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
    return FasterRCNN(backbone,num_classes=len(CLASSES),min_size=config['min_size'],max_size=config['max_size'],
        box_detections_per_img=config['max_detections'])


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


def policy_batch(net,rows,device,rng,prediction_probability):
    cfg=net.cfg
    if cfg.policy_head=='autoregressive':
        raise ValueError('Delta/gain/set-loss training retired; use visual-tree-prepare and visual-tree-train')
    batch=visual_batch([current_features(r['current'],r['current_boxes'],r['viewport'],cfg.max_nodes) for r in rows],device)
    target=[];current=[]
    for row in rows:
        if cfg.semantic:
            predicted=rng.random()<prediction_probability
            if predicted:
                from .ir import read_json
                path=row.get('predicted_target_elements')
                if not path:raise ValueError('Missing detector JSON; run visual-cache-targets with the new version')
                items=read_json(path)
            else:
                if 'target_elements' not in row:raise ValueError('Missing target elements; run visual-subset-numeric on cached rendered data')
                items=row['target_elements']
            from .visual import semantic_channels
            boundary=semantic_channels(cfg)==8
            target.append(semantic_masks(items,row['viewport'],cfg.size,boundaries=boundary))
            current.append(semantic_masks(elements(row['current'],row['current_boxes'],row['viewport'],row.get('current_clip_boxes')),row['viewport'],cfg.size,boundaries=boundary))
            continue
        if cfg.mode=='screenshot':tp,cp=row['target_image'],row['current_image']
        else:
            use_prediction=rng.random()<prediction_probability
            if use_prediction and not row.get('predicted_target_abstract'):raise ValueError('Predicted targets missing; run visual-cache-targets first')
            tp=row['predicted_target_abstract'] if use_prediction else row['target_abstract']
            cp=row['current_abstract']
        target.append(image_tensor(tp,cfg.size));current.append(image_tensor(cp,cfg.size))
    logits,components=net(batch,torch.stack(target).to(device),torch.stack(current).to(device),
                          return_components=True)
    logp=F.log_softmax(logits,dim=-1);n=batch['mask'].shape[1];losses=[];labels=[]
    factor={'operation':[],'node':[],'property':[],'delta':[]}
    for i,row in enumerate(rows):
        valid=[action_index(e,n) for e in row['teacher_edits']] or [action_index(None,n)]
        illegal=[e for e in row['teacher_edits'] if not action_is_legal(
            row['current'],row['current_boxes'],row['viewport'],e)]
        if illegal:raise ValueError(f'Illegal teacher edit: row={row.get("id")} edits={illegal}')
        if not torch.isfinite(logits[i,valid]).all():
            values=logits[i,valid].detach().float().cpu().tolist()
            raise RuntimeError(f'Non-finite policy logits: row={row.get("id")} target_logits={values}')
        labels.append(valid)
        if net.cfg.policy_head=='hierarchical' and net.cfg.training_scheme=='balanced-v1':
            is_edit=bool(row['teacher_edits']);factor['operation'].append(-components['operation'][i,int(is_edit)])
            if is_edit:
                if len(row['teacher_edits'])!=1:raise ValueError('Balanced hierarchy requires one teacher edit per row')
                node,field,value=row['teacher_edits'][0]
                prop=NUMERIC_FIELDS.index(field);delta=NORMALIZED_DELTAS.index(value)
                factor['node'].append(-components['node'][i,node])
                factor['property'].append(-components['property'][i,node,prop])
                factor['delta'].append(-components['delta'][i,node,prop,delta])
        else:losses.append(-torch.logsumexp(logp[i,valid],0))
    if net.cfg.policy_head=='hierarchical' and net.cfg.training_scheme=='balanced-v1':
        parts={key:torch.stack(values).mean() for key,values in factor.items() if values};loss=sum(parts.values())
    else:parts={'joint':torch.stack(losses).mean()};loss=parts['joint']
    return loss,logits,labels,n,parts


def policy_loss(net,rows,device,rng,prediction_probability):
    return policy_batch(net,rows,device,rng,prediction_probability)[0]


def policy_training_pools(rows,include_stop=True):
    """Balanced operation/property pools; assets and row contents remain unchanged."""
    stop=[r for r in rows if not r['teacher_edits']]
    edits={field:[] for field in NUMERIC_FIELDS}
    for row in rows:
        if row['teacher_edits']:
            if len(row['teacher_edits'])!=1 or row['teacher_edits'][0][1] not in edits:
                raise ValueError('Balanced numeric sampler requires one numeric teacher edit')
            edits[row['teacher_edits'][0][1]].append(row)
    missing=[field for field,items in edits.items() if not items]
    if (include_stop and not stop) or missing:
        raise ValueError(f'Balanced sampler needs requested operations and every numeric property; missing={missing}')
    return {'stop':stop,'edit':edits} if include_stop else {'edit':edits}


def sample_policy_rows(pools,batch_size,rng):
    if 'stop' not in pools:
        fields=list(NUMERIC_FIELDS)
        return [rng.choice(pools['edit'][rng.choice(fields)]) for _ in range(batch_size)]
    stop_count=batch_size//2
    if batch_size%2:stop_count+=rng.randrange(2)
    result=[rng.choice(pools['stop']) for _ in range(stop_count)]
    fields=list(NUMERIC_FIELDS)
    for _ in range(batch_size-stop_count):
        field=rng.choice(fields);result.append(rng.choice(pools['edit'][field]))
    rng.shuffle(result);return result


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


def policy_validation_metrics(net,rows,device,rng,prediction_probability,batch_size,amp):
    totals={'n':0,'correct':0,'stop_n':0,'stop_correct':0,'non_stop_n':0,'non_stop_correct':0,
            'node_correct':0,'property_correct':0,'value_correct':0,'predicted_stop':0,'false_stop':0,
            'edit_only_correct':0,'property_given_node_n':0,'property_given_node_correct':0,
            'delta_given_node_property_n':0,'delta_given_node_property_correct':0,
            'selected_gain':0.,'positive_count':0}
    per_property={};operation_scores=[];operation_targets=[]
    for offset in range(0,len(rows),batch_size):
        batch_rows=rows[offset:offset+batch_size]
        with amp():_,logits,labels,n,_=policy_batch(net,batch_rows,device,rng,prediction_probability)
        predictions=net.select(logits).cpu().tolist()
        edit_predictions=logits[:,:-1].argmax(-1).cpu().tolist()
        scores=torch.sigmoid(logits[:,-1].float()-torch.logsumexp(logits[:,:-1].float(),-1)).cpu().tolist()
        for row,prediction,valid,edit_prediction,score in zip(batch_rows,predictions,labels,edit_predictions,scores):
            predicted=decode_action(prediction,n);targets=[decode_action(label,n) for label in valid];target=targets[0]
            operation_scores.append(score);operation_targets.append(target is None)
            totals['n']+=1;totals['correct']+=prediction in valid
            totals['positive_count']+=len(valid)
            totals['predicted_stop']+=predicted is None
            if target is None:
                totals['stop_n']+=1;totals['stop_correct']+=predicted is None
                continue
            totals['non_stop_n']+=1;totals['non_stop_correct']+=prediction in valid
            totals['false_stop']+=predicted is None
            totals['edit_only_correct']+=edit_prediction in valid
            fields={item[1] for item in targets}
            for field in fields:
                entry=per_property.setdefault(field,{'n':0,'correct':0})
                entry['n']+=1;entry['correct']+=prediction in valid
            if predicted is not None:
                node_ok=any(predicted[0]==item[0] for item in targets)
                totals['node_correct']+=node_ok
                totals['property_correct']+=predicted[1] in fields
                totals['value_correct']+=any(predicted[2]==item[2] for item in targets)
                if node_ok:
                    totals['property_given_node_n']+=1
                    property_ok=predicted[1] in fields
                    totals['property_given_node_correct']+=property_ok
                    if property_ok:
                        totals['delta_given_node_property_n']+=1
                        totals['delta_given_node_property_correct']+=any(predicted[2]==item[2] for item in targets)
    def div(a,b):return totals[a]/totals[b] if totals[b] else None
    edit_recall=1-div('false_stop','non_stop_n') if totals['non_stop_n'] else None
    balanced=(div('stop_correct','stop_n')+edit_recall)/2 if totals['stop_n'] and totals['non_stop_n'] else None
    candidates=[i/100 for i in range(5,100,5)]
    def threshold_score(threshold):
        if not totals['stop_n'] or not totals['non_stop_n']:return None
        stop_hits=sum(score>=threshold for score,target in zip(operation_scores,operation_targets) if target)
        edit_hits=sum(score<threshold for score,target in zip(operation_scores,operation_targets) if not target)
        return (stop_hits/totals['stop_n']+edit_hits/totals['non_stop_n'])/2
    recommended=(max(candidates,key=lambda t:(threshold_score(t),-abs(t-.5)))
                 if totals['stop_n'] and totals['non_stop_n'] else .5)
    result={'node_accuracy':div('node_correct','non_stop_n'),
            'predicted_stop_rate':div('predicted_stop','n'),'false_stop_rate':div('false_stop','non_stop_n'),
            'edit_recall':edit_recall,'operation_balanced_accuracy':balanced,
            'recommended_stop_threshold':recommended,
            'calibrated_operation_balanced_accuracy':threshold_score(recommended),
            'edit_only_joint_accuracy':div('edit_only_correct','non_stop_n'),
            'property_accuracy':div('property_correct','non_stop_n'),'value_accuracy':div('value_correct','non_stop_n'),
            'property_accuracy_given_correct_node':div('property_given_node_correct','property_given_node_n'),
            'delta_accuracy_given_correct_node_property':div('delta_given_node_property_correct','delta_given_node_property_n'),
            'stop_accuracy':div('stop_correct','stop_n'),'non_stop_accuracy':div('non_stop_correct','non_stop_n'),
            'validation_n':totals['n'],'stop_n':totals['stop_n'],'non_stop_n':totals['non_stop_n'],
            'per_property':{field:{'n':v['n'],'accuracy':v['correct']/v['n']} for field,v in sorted(per_property.items())}}
    result['action_accuracy']=div('correct','n')
    return result


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


def train(args):
    with contextlib.ExitStack() as stack:
        return _train(args,stack)


def _train(args,stack):
    patience=args.early_stop_patience;min_delta=args.early_stop_min_delta
    if patience<0 or not math_isfinite(min_delta) or min_delta<0:
        raise ValueError('Early-stop patience and min-delta must be finite and nonnegative')
    device=select_device(args.device);seed_all(args.seed)
    if device=='cpu':torch.set_num_threads(args.cpu_threads)
    train_rows=list(read_jsonl(args.train));val_rows=list(read_jsonl(args.val))
    if not train_rows or not val_rows:raise ValueError('Nonempty train and validation data required')
    if any(r['split']!='train' for r in train_rows) or any(r['split']!='val' for r in val_rows):raise ValueError('Use explicit train/val splits')
    validate_splits(train_rows+val_rows)
    if any(r.get('contract')!=CONTRACT for r in train_rows+val_rows):raise ValueError('Dataset contract mismatch')
    if not 0<=args.prediction_probability<=1:raise ValueError('Prediction probability must be in [0,1]')
    is_detector=args.command=='visual-train-detector'
    loader_workers=getattr(args,'loader_workers',2);loader_prefetch=getattr(args,'loader_prefetch',2)
    if is_detector and (loader_workers<0 or loader_prefetch<1):raise ValueError('Invalid detector loader settings')
    legality_report=None
    if not is_detector:
        train_rows,train_rejected=filter_legal_policy_rows(train_rows)
        val_rows,val_rejected=filter_legal_policy_rows(val_rows)
        legality_report={'train_rejected':len(train_rejected),'val_rejected':len(val_rejected),
                         'train_examples':train_rejected[:10],'val_examples':val_rejected[:10]}
        if train_rejected or val_rejected:
            print({'policy_legality_filter':legality_report},flush=True)
        if not train_rows or not val_rows:raise ValueError('No legal policy train/validation rows remain')
    if is_detector:
        cfg={'contract':CONTRACT,'min_size':args.min_size,'max_size':args.max_size,'max_detections':args.max_detections}
        from .visual import label_contract
        geometry_contract=label_contract(train_rows+val_rows)
        if geometry_contract:cfg['visual_geometry_contract']=geometry_contract
        coverage={r.get('detector_label_contract') for r in train_rows+val_rows}
        if len(coverage)>1:raise ValueError('Mixed detector label coverage versions; regenerate matching train/val labels')
        if next(iter(coverage)):cfg['detector_label_contract']=next(iter(coverage))
        if cfg['min_size']<32 or cfg['max_size']<cfg['min_size'] or cfg['max_detections']<1:raise ValueError('Invalid detector dimensions')
        kind='visual-detector-v1'
    else:
        if any(r.get('action_contract')!=ACTION_CONTRACT for r in train_rows+val_rows):
            raise ValueError('Policy data uses a legacy action contract; rebuild visual data')
        cfg=asdict(VisualConfig(mode=args.mode,size=args.size,hidden=args.hidden,layers=args.layers,
                                heads=args.heads,max_nodes=args.max_nodes,token_grid=args.token_grid,
                                policy_head=getattr(args,'policy_head','flat'),numeric_only=getattr(args,'numeric_only',False),
                                decoding=getattr(args,'decoding','joint'),
                                observation_contract=('semantic-mask-pair-diff-v1' if args.mode=='abstract' and
                                    getattr(args,'policy_head','flat')=='autoregressive' else 'rgb-pair-diff-v1'),
                                training_scheme='balanced-v1' if getattr(args,'balanced_policy',False) else 'legacy'))
        from .visual import label_contract
        geometry_contract=label_contract(train_rows+val_rows)
        if geometry_contract:cfg['visual_geometry_contract']=geometry_contract
        else:cfg.pop('visual_geometry_contract')
        if cfg['policy_head']=='autoregressive':
            raise ValueError('Delta/gain/set-loss training retired; use visual-tree-prepare and visual-tree-train')
        kind='visual-policy-v3'
        if cfg['numeric_only'] and any(r.get('policy_subset')!='numeric-prefix-v1' for r in train_rows+val_rows):
            raise ValueError('Use visual-subset-numeric first: remove categorical corruption ancestry, not just labels')
    if args.resume and args.init_checkpoint:raise ValueError('Choose resume or init-checkpoint')
    data_signature={'train':digest(args.train),'val':digest(args.val)}
    out=Path(args.out).resolve()
    run_settings={k:v for k,v in vars(args).items() if k not in
                  ('out','resume','steps','init_checkpoint','render_workers','render_prefetch','render_timeout',
                   'loader_workers','loader_prefetch')}
    run_settings['validation_sampling']='page-round-robin-v1'
    run_config={'kind':kind,'config':cfg,'data':data_signature,'settings':run_settings}
    if is_detector:
        validation_batch_size=guard_detector_resume(out,run_config,bool(args.resume))
    else:
        guard_run(out,run_config,bool(args.resume))
    ck=torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=True) if args.resume or args.init_checkpoint else None
    legacy_policy_init=False
    if ck:
        checkpoint_config=ck['config']
        if not is_detector and ck.get('kind') in ('visual-policy-v3','visual-policy-v4','visual-policy-v5-no-stop',
                                                  'visual-policy-v6-semantic','visual-policy-v7-improvement',
                                                  'visual-policy-v8-tree-path'):
            checkpoint_config=asdict(VisualConfig(**checkpoint_config))
            if checkpoint_config.get('visual_geometry_contract') is None:checkpoint_config.pop('visual_geometry_contract')
        if ck.get('kind')==kind and checkpoint_config==cfg:pass
        elif (args.init_checkpoint and not is_detector and ck.get('kind')=='visual-policy-v1' and
              cfg['policy_head']=='flat' and not cfg['numeric_only'] and cfg['decoding']=='joint' and
              ck.get('config')=={k:v for k,v in cfg.items() if k not in ('action_contract','policy_head','numeric_only','decoding','training_scheme')}):
            # Reuse the image/tree encoder and transformer, but deliberately
            # reinitialize the expanded structured-action head.
            legacy_policy_init=True
        else:raise ValueError('Checkpoint architecture/mode mismatch')
    if args.resume and ck.get('early_stopping',{}).get('stopped',False):
        print(f'Early stopping already completed at step {ck["step"]}; keeping best.pt and skipping stage.',flush=True)
        return
    net=(detector(cfg,not args.no_pretrained and ck is None) if is_detector else
         VisualPolicy(VisualConfig(**cfg),not args.no_pretrained and ck is None)).to(device)
    groups={r['group'] for r in train_rows};source_hashes={r['source_sha'] for r in train_rows}
    if ck:
        if legacy_policy_init:
            reusable={k:v for k,v in ck['model'].items() if not k.startswith('action.')}
            missing,unexpected=net.load_state_dict(reusable,strict=False)
            if set(missing)!={'action.weight','action.bias'} or unexpected:
                raise ValueError('Legacy policy migration found unexpected parameters')
        else:net.load_state_dict(ck['model'])
        net.stop_threshold=float(ck.get('stop_threshold',.5))
        groups.update(ck.get('training_groups',[]));source_hashes.update(ck.get('training_hashes',[]))
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
    policy_pools=(policy_training_pools(train_rows,cfg['policy_head']!='autoregressive') if not is_detector and
                  cfg['training_scheme']=='balanced-v1' else None)
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
        'policy_legality_filter':legality_report,
        'parameters':sum(p.numel() for p in net.parameters()),'pretrained_backbone':not args.no_pretrained,
        'torch':str(torch.__version__)})
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') and args.bf16 and not is_detector else contextlib.nullcontext()
    started=time.perf_counter()
    render_stream=None
    if is_detector and any(row.get('lazy_contract') for row in train_rows) and start<args.steps and not early['stopped']:
        from .detector_support import DetectorStream
        render_stream=stack.enter_context(DetectorStream(view_pools,rng,args.batch_size,
            Path(args.train).resolve().parent/'.lazy-detector-cache',out/'lazy-rejections.jsonl',
            getattr(args,'render_workers',2),getattr(args,'render_prefetch',4),getattr(args,'render_timeout',900.),
            decode=bool(loader_workers)))
    elif is_detector and loader_workers and start<args.steps and not early['stopped']:
        from .detector_support import DetectorLoader
        render_stream=DetectorLoader(view_pools,rng,args.batch_size,loader_workers,loader_prefetch,
                                     device.startswith('cuda'))
        stack.callback(render_stream.close)
    for step in range(start+1,args.steps+1):
        step_started=time.perf_counter()
        net.train();opt.zero_grad(set_to_none=True);total=0.;step_parts={};data_wait=0.
        for _ in range(args.accumulation):
            wait_started=time.perf_counter()
            batch=render_stream.take() if render_stream else (sample_policy_rows(policy_pools,args.batch_size,rng) if policy_pools else
                   sample_page_views(view_pools,args.batch_size,rng))
            if is_detector and render_stream is None:
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
            data_wait+=time.perf_counter()-wait_started
            with amp():
                if is_detector:value=detection_loss(net,batch,device);parts={}
                elif cfg['training_scheme']=='balanced-v1':
                    value,_,_,_,parts=policy_batch(net,batch,device,rng,args.prediction_probability)
                else:value=policy_loss(net,batch,device,rng,args.prediction_probability);parts={}
            if not torch.isfinite(value):raise RuntimeError('Non-finite training loss')
            (value/args.accumulation).backward()
            total+=(value.detach() if is_detector else float(value.detach()))/args.accumulation
            for key,part in parts.items():step_parts[key]=step_parts.get(key,0.)+float(part.detach())/args.accumulation
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step%args.log_every==0 or step==args.steps:
            row={'step':step,'loss':float(total),'elapsed_s':time.perf_counter()-started}
            if device.startswith('cuda'):
                row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                           cuda_reserved_bytes=torch.cuda.memory_reserved(device))
            if is_detector:
                row.update(data_wait_s=data_wait,step_seconds=time.perf_counter()-step_started)
                row['images_per_second']=args.batch_size*args.accumulation/row['step_seconds']
            row.update({f'{key}_loss':value for key,value in sorted(step_parts.items())})
            with (out/'train.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps(row)+'\n')
            print(row,flush=True)
        if step%args.eval_every==0 or step==args.steps:
            # Torchvision detection losses require train mode. FrozenBatchNorm
            # avoids changing running statistics; fork_rng makes RPN sampling repeatable.
            net.train(is_detector);values=[];components={};component_counts={};vrng=random.Random(90210)
            selected=validation_sample(val_rows,args.val_samples)
            with torch.random.fork_rng(),torch.no_grad():
                torch.manual_seed(90210)
                from .detector_support import detector_batches
                batches=(detector_batches(selected,validation_batch_size,loader_workers,loader_prefetch,device.startswith('cuda'))
                         if is_detector else (selected[i:i+args.batch_size] for i in range(0,len(selected),args.batch_size)))
                for rows in batches:
                    with amp():
                        if is_detector:
                            parts=detection_losses(net,rows,device);value=sum(parts.values())
                            for key,part in parts.items():components[key]=components.get(key,0.)+float(part)*len(rows)
                        elif cfg['training_scheme']=='balanced-v1':
                            value,_,_,_,parts=policy_batch(net,rows,device,vrng,args.prediction_probability)
                            edits=sum(bool(r['teacher_edits']) for r in rows)
                            for key,part in parts.items():
                                count=len(rows) if key in ('operation','joint') else edits
                                if count:
                                    components[key]=components.get(key,0.)+float(part)*count
                                    component_counts[key]=component_counts.get(key,0)+count
                        else:value=policy_loss(net,rows,device,vrng,args.prediction_probability)
                    values.append((float(value),len(rows)))
            if not is_detector and cfg['training_scheme']=='balanced-v1' and cfg['policy_head']=='hierarchical':
                validation=sum(components[key]/component_counts[key] for key in components)
            else:validation=sum(v*n for v,n in values)/sum(n for _,n in values)
            if not math_isfinite(validation):raise RuntimeError('Non-finite validation loss')
            eval_row={'step':step,'validation_loss':validation,'validation_n':len(selected),
                      'validation_pages':len({r.get('page_id',r.get('target_image',r['source_sha'])) for r in selected}),
                      'validation_available_n':len(val_rows),'validation_complete':len(selected)==len(val_rows)}
            if is_detector:
                eval_row['validation_components']={k:v/len(selected) for k,v in sorted(components.items())}
                with torch.inference_mode():eval_row.update(detector_quality(net,selected,device,args.metric_threshold,
                                                                            loader_workers,loader_prefetch))
            else:
                eval_row['validation_components']={key:components[key]/component_counts[key]
                                                   for key in sorted(components)}
                net.eval()
                with torch.no_grad():
                    metrics=policy_validation_metrics(net,selected,device,random.Random(90210),
                        args.prediction_probability,args.batch_size,amp)
                    eval_row.update(metrics)
                if cfg['training_scheme']=='balanced-v1' and cfg['policy_head']!='autoregressive':
                    net.stop_threshold=metrics['recommended_stop_threshold']
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
            if cfg.get('policy_head')!='autoregressive':
                checkpoint['stop_threshold']=getattr(net,'stop_threshold',.5)
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


def math_isfinite(value):
    import math
    return math.isfinite(value)


def filter_legal_policy_rows(rows):
    """Drop only examples whose teacher edit contradicts the action grammar/mask.

    Existing rendered corpora can therefore be reused after this invariant was
    tightened. Clean STOP rows remain valid. Rejected IDs and edits are returned
    for an explicit, auditable training manifest.
    """
    kept=[];rejected=[]
    for row in rows:
        supervised=list(row.get('teacher_edits',[]))+[
            item.get('edit') for item in row.get('improving_edits',[]) if isinstance(item,dict) and item.get('edit')]
        supervised+=list(row.get('tree_path_edits',[]))
        illegal=[edit for edit in supervised if not action_is_legal(
            row['current'],row['current_boxes'],row['viewport'],edit)]
        if illegal:rejected.append({'id':row.get('id'),'edits':illegal})
        else:kept.append(row)
    return kept,rejected


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

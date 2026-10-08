"""Image-conditioned replacement decoder and ordinary teacher-forced token CE.

No image reward in labels/loss, no set loss, no learned STOP, no box oracle at
inference. Grammar constraints belong to decoding; success belongs to execution.
"""
import contextlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .ir import read_json, read_jsonl, write_json
from .model import Block, ModelConfig, encode, collate
from .train import seed_all, select_device
from .tree_edits import CONTRACT, FIELDS, EditTokenizer, apply_state, extract, repair_path
from .visual import SpatialEncoder, elements, semantic_masks, image_tensor,BOUNDARY_CONTRACT,SEMANTIC_CHANNELS
from .visual_data import validate_splits
from .web_experiment import digest, guard_run

KIND = 'visual-policy-css-tree-v1'

POLICY_SCALES = {
    's': {'hidden': 128, 'layers': 3, 'heads': 4},
    'm': {'hidden': 256, 'layers': 6, 'heads': 8},
    'l': {'hidden': 512, 'layers': 8, 'heads': 8},
}


def resolve_policy_scale(args):
    """Resolve a named capacity before checkpoint/config validation."""
    scale = getattr(args, 'policy_scale', None)
    if scale:
        for key, value in POLICY_SCALES[scale].items():
            setattr(args, key, value)


@dataclass
class TreeConfig:
    mode: str = 'abstract'
    size: int = 384
    hidden: int = 128
    layers: int = 3
    heads: int = 4
    max_nodes: int = 128
    token_grid: int = 12
    policy_head: str = 'replacement'
    numeric_only: bool = True
    action_contract: str = CONTRACT
    stylesheets: bool = False
    # False keeps historical checkpoints loadable. New stylesheet training opts
    # in explicitly; inference then cannot add absent declarations/overrides.
    existing_values_only: bool = False
    abstraction_contract: str = BOUNDARY_CONTRACT

    def __post_init__(self):
        if self.abstraction_contract!=BOUNDARY_CONTRACT:
            raise ValueError('Retrain abstract policy for occupancy+boundary inputs')
        from .css_owners import CONTRACT as OWNER_CONTRACT
        if self.mode not in ('abstract','screenshot') or self.action_contract != (OWNER_CONTRACT if self.stylesheets else CONTRACT) or self.policy_head != 'replacement':
            raise ValueError('Replacement policy contract mismatch')
        if self.size < 32 or min(self.hidden,self.layers,self.heads,self.max_nodes,self.token_grid)<1 or self.hidden%self.heads:
            raise ValueError('Invalid model dimensions')

    @property
    def semantic(self): return self.mode == 'abstract'

    @property
    def observation_contract(self): return self.abstraction_contract if self.semantic else 'rgb-pair-diff-v1'


class TreePolicy(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__(); self.cfg=cfg; self.tokenizer=EditTokenizer(cfg.max_nodes)
        self.vision=SpatialEncoder(cfg.hidden,cfg.token_grid,pretrained,3*SEMANTIC_CHANNELS if cfg.semantic else 9)
        self.token=nn.Embedding(len(self.tokenizer.tokens),cfg.hidden)
        self.byte=nn.Embedding(257,cfg.hidden,padding_idx=0)
        self.css_encoder=nn.GRU(cfg.hidden,cfg.hidden,batch_first=True)
        self.css_projection=nn.Linear(len(FIELDS)*cfg.hidden,cfg.hidden)
        if cfg.stylesheets:self.owner_type=nn.Embedding(3,cfg.hidden)
        self.geometry=nn.Linear(4,cfg.hidden)
        self.node=nn.Embedding(cfg.max_nodes,cfg.hidden)
        self.blocks=nn.ModuleList(Block(ModelConfig(hidden=cfg.hidden,layers=cfg.layers,heads=cfg.heads,dropout=0.)) for _ in range(cfg.layers))
        self.cross=nn.MultiheadAttention(cfg.hidden,cfg.heads,batch_first=True)
        self.position=nn.Embedding(self.tokenizer.max_length+1,cfg.hidden)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(cfg.hidden,cfg.heads,cfg.hidden*4,dropout=0.,batch_first=True),cfg.layers)
        self.output=nn.Linear(cfg.hidden,len(self.tokenizer.tokens))

    def encode_pair(self,target,current):
        return self.vision(torch.cat([target,current,(target-current).abs()],1))

    def memory(self,batch,target,current,states,encoded_pair=None):
        from torchvision.ops import roi_align
        spatial,feature=self.encode_pair(target,current) if encoded_pair is None else encoded_pair
        b,n=batch['mask'].shape;device=target.device
        declarations=torch.zeros(b,n,len(FIELDS),64,dtype=torch.long,device=device)
        for i,state in enumerate(states):
            if len(state)>self.cfg.max_nodes: raise ValueError('Too many DOM nodes')
            for j,props in enumerate(state):
                if len(props)!=len(FIELDS):raise ValueError('CSS field schema changed; prepare six-field size/margin labels')
                for k,(value,priority) in enumerate(props):
                    # Current CSS is context, not target CSS. Unsupported unchanged
                    # values remain visible as bytes; only replacements use grammar.
                    encoded=(value+'!'+priority).encode()[:64]
                    declarations[i,j,k,:len(encoded)]=torch.tensor([v+1 for v in encoded],device=device)
        # Preserve character order (12px != 21px) and field identity (width != height).
        flat=declarations.flatten(0,2);lengths=(flat!=0).sum(-1).clamp_min(1)
        encoded,_=self.css_encoder(self.byte(flat))
        embedded=encoded[torch.arange(flat.shape[0],device=device),lengths-1]
        css=self.css_projection(embedded.reshape(b,n,-1))
        if self.cfg.stylesheets:
            # Current selector/path context and inline/rule type, not target CSS.
            context=batch['owner_text'];flat=context.flatten(0,1)
            lengths=(flat!=0).sum(-1).clamp_min(1)
            text,_=self.css_encoder(self.byte(flat))
            css=css+text[torch.arange(b*n,device=device),lengths-1].reshape(b,n,-1)+self.owner_type(batch['owner_type'])
        local=roi_align(feature,[r*feature.shape[-1] for r in batch['roi']],output_size=3,aligned=True).mean((-1,-2)).reshape(b,n,-1)
        x=self.node(torch.arange(n,device=device))[None]+css+self.geometry(batch['roi'])+local
        x=x+self.cross(x,spatial,spatial,need_weights=False)[0]
        for block in self.blocks: x=block(x,batch['relation'],batch['mask'])
        memory=torch.cat([x,spatial],1)
        padding=torch.cat([~batch['mask'],torch.zeros(b,spatial.shape[1],dtype=torch.bool,device=device)],1)
        return memory,padding

    def logits(self,memory,padding,tokens):
        length=tokens.shape[1]
        x=self.token(tokens)+self.position(torch.arange(length,device=tokens.device))[None]
        mask=torch.triu(torch.ones(length,length,device=tokens.device,dtype=torch.bool),1)
        return self.output(self.decoder(x,memory,tgt_mask=mask,memory_key_padding_mask=padding))

    def loss(self,batch,target,current,states,edits):
        sequences=[self.tokenizer.encode(edit,len(state)) for edit,state in zip(edits,states)]
        labels=torch.full((len(edits),max(map(len,sequences))),-100,dtype=torch.long,device=target.device)
        for i,seq in enumerate(sequences): labels[i,:len(seq)]=torch.tensor(seq,device=target.device)
        inputs=labels.clamp_min(0).roll(1,1);inputs[:,0]=self.tokenizer.ids['BOS']
        memory,padding=self.memory(batch,target,current,states)
        logits=self.logits(memory,padding,inputs)
        # The reference trains one replacement token sequence, not a reward
        # weighted distribution or -log(sum probability of possible actions).
        loss=F.cross_entropy(logits.flatten(0,1).float(),labels.flatten(),ignore_index=-100)
        return loss,int((labels!=-100).sum())

    @torch.no_grad()
    def predict(self,batch,target,current,states,encoded_pair=None):
        if len(states)!=1: raise ValueError('Rollout decoder takes one page at a time')
        memory,padding=self.memory(batch,target,current,states,encoded_pair);prefix=[]
        for _ in range(self.tokenizer.max_length):
            allowed=self.tokenizer.allowed(prefix,len(states[0]),
                state=states[0] if self.cfg.existing_values_only else None,
                size_limits=batch.get('explicit_owner_limits',[{}])[0])
            if not allowed: raise ValueError('No grammar-valid continuation')
            tokens=torch.tensor([[self.tokenizer.ids['BOS'],*prefix]],device=target.device)
            scores=self.logits(memory,padding,tokens)[0,-1]
            next_id=allowed[int(scores[allowed].argmax())];prefix.append(next_id)
            if next_id==self.tokenizer.ids['EOS']: return self.tokenizer.decode(prefix,len(states[0]))
        raise ValueError('Replacement exceeded grammar token limit')


def tree_batch(rows,device,max_nodes):
    """Current tree/geometry only; no retired numeric-delta legality mask."""
    from .css_owners import context_tree
    contexts=[context_tree(r['current'],r['current_boxes'],r['css_owners']) if 'css_owners' in r
              else (r['current'],r['current_boxes']) for r in rows]
    b=len(rows);n=max(len(t['nodes']) for t,_ in contexts)
    if n>max_nodes:raise ValueError(f'{n} CSS/DOM owners exceeds max_nodes={max_nodes}; increase --max-nodes')
    batch={'mask':torch.zeros(b,n,dtype=torch.bool,device=device),
           'relation':torch.zeros(b,n,n,dtype=torch.long,device=device),
           'owner_type':torch.zeros(b,n,dtype=torch.long,device=device),
           'owner_text':torch.zeros(b,n,256,dtype=torch.long,device=device)}
    rois=torch.zeros(b,n,4,device=device)
    for i,(row,(tree,boxes)) in enumerate(zip(rows,contexts)):
        values=[];scale=max(row['viewport'])
        nodes=tree['nodes'];ids={a['id']:j for j,a in enumerate(nodes)}
        parents=[ids.get(a.get('parent'),-1) for a in nodes]
        m=len(nodes);batch['mask'][i,:m]=True
        relation=[[1 if j==k else 2 if parents[k]==j else 3 if parents[j]==k else 4 if parents[j]==parents[k] else 0
                   for k in range(m)] for j in range(m)]
        for j,owner in enumerate(row.get('css_owners',[])):
            batch['owner_type'][i,j]={'root':0,'inline':1,'rule':2}[owner['kind']]
            text=json.dumps({k:v for k,v in owner.items() if k!='matches'},sort_keys=True).encode()[:256]
            batch['owner_text'][i,j,:len(text)]=torch.tensor([v+1 for v in text],device=device)
            if owner['kind']=='rule':
                for member in owner['matches']:
                    if member in ids:relation[j][ids[member]]=2;relation[ids[member]][j]=3
        batch['relation'][i,:m,:m]=torch.tensor(relation,device=device)
        for node in nodes:
            x,y,w,h=boxes[node['id']];values.append([x/scale,y/scale,(x+w)/scale,(y+h)/scale])
        rois[i,:len(values)]=torch.tensor(values,device=device).clamp(0,1)
    batch['roi']=rois
    from .explicit_html import owner_size_limits
    batch['explicit_owner_limits']=[owner_size_limits(r) for r in rows]
    return batch


def batch_inputs(net,rows,device,predicted=False):
    cfg=net.cfg;targets=[];currents=[]
    for row in rows:
        if row.get('hierarchy'):raise ValueError('Flat policy data is retired; use original DOM data')
        if cfg.semantic:
            if predicted:
                if not row.get('predicted_target_elements'): raise ValueError('Missing frozen-detector cache')
                items=read_json(row['predicted_target_elements'])
            else: items=row['target_elements']
            targets.append(semantic_masks(items,row['viewport'],cfg.size))
            currents.append(semantic_masks(elements(row['current'],row['current_boxes'],row['viewport'],row.get('current_clip_boxes')),row['viewport'],cfg.size))
        else:
            targets.append(image_tensor(row['target_image'],cfg.size));currents.append(image_tensor(row['current_image'],cfg.size))
    batch=tree_batch(rows,device,cfg.max_nodes)
    return batch,torch.stack(targets).to(device),torch.stack(currents).to(device),[r['declaration_state'] for r in rows]


def validation_target_states(rows,out,data_signature,stylesheets):
    """Load clean declaration states for diagnostics only, never model inputs.

    New corpora carry the state directly. Older already-prepared v3 corpora are
    upgraded lazily from their clean target HTML and cached inside the run, so a
    long preparation job does not need to be restarted.
    """
    cache=Path(out)/'validation-target-states.json'
    ids=[r['id'] for r in rows]
    if cache.exists():
        saved=read_json(cache)
        if saved.get('data_signature')==data_signature and saved.get('ids')==ids:
            return saved['states']
    states=[];browser=None
    try:
        for row in rows:
            state=row.get('target_declaration_state')
            if state is None:
                target_html=row.get('target_html')
                if not target_html:raise ValueError(f"Missing target declaration state/HTML for {row['id']}")
                if browser is None:
                    from .html_bridge import HtmlBrowser
                    browser=HtmlBrowser().__enter__()
                html=Path(target_html).read_text()
                if stylesheets:
                    from . import css_owners
                    parsed=css_owners.read(browser,row['current'],html,fixed=True)
                    if parsed['owners']!=row['css_owners']:
                        raise ValueError(f"CSS owner topology changed for {row['id']}")
                else:parsed=extract(browser,html,row['current'])
                state=parsed['state']
            if len(state)!=len(row['declaration_state']):
                raise ValueError(f"Target declaration topology changed for {row['id']}")
            states.append(state)
    finally:
        if browser is not None:browser.__exit__(None,None,None)
    write_json(cache,{'data_signature':data_signature,'ids':ids,'states':states})
    return states


@torch.no_grad()
def symbolic_policy_metrics(net,rows,target_states,device,predicted,amp):
    """One-step tree-distance diagnostics with every valid reverse edit positive.

    A prediction improves iff it reduces the number of declaration children that
    differ from the clean target. It is not compared with one shuffled teacher.
    """
    totals={'n':0,'improving':0,'worsening':0,'no_change':0,'invalid':0,
            'gain':0.,'normalized_gain':0.,'distance':0.,'positive_actions':0.}
    per_property={field:{'n':0,'improving':0} for field in FIELDS}
    for row,target in zip(rows,target_states):
        current=row['declaration_state'];before=len(repair_path(current,target))
        if before<1:continue
        totals['n']+=1;totals['distance']+=before;totals['positive_actions']+=before
        try:
            inputs=batch_inputs(net,[row],device,predicted)
            with amp():action=net.predict(*inputs)
            from .explicit_html import check_record_edit
            check_record_edit(row,action)
            after_state=apply_state(current,action);after=len(repair_path(after_state,target))
            gain=before-after;totals['gain']+=gain;totals['normalized_gain']+=gain/before
            entry=per_property[action[1]];entry['n']+=1
            if gain>0:totals['improving']+=1;entry['improving']+=1
            elif gain<0:totals['worsening']+=1
            else:totals['no_change']+=1
        except Exception:
            totals['invalid']+=1
    n=totals['n']
    if not n:return {'symbolic_metric_n':0}
    return {'symbolic_metric_n':n,
            'symbolic_improving_action_rate':totals['improving']/n,
            'symbolic_worsening_action_rate':totals['worsening']/n,
            'symbolic_no_change_action_rate':totals['no_change']/n,
            'invalid_action_rate':totals['invalid']/n,
            'mean_symbolic_distance_reduction':totals['gain']/n,
            'mean_normalized_symbolic_distance_reduction':totals['normalized_gain']/n,
            'mean_symbolic_distance':totals['distance']/n,
            'mean_positive_action_count':totals['positive_actions']/n,
            'symbolic_by_predicted_property':{field:{'n':v['n'],
                'improving_rate':v['improving']/v['n'] if v['n'] else None}
                for field,v in per_property.items()}}


def train(args):
    from .visual_train import validation_sample, update_early_stopping
    resolve_policy_scale(args)
    device=select_device(args.device);seed_all(args.seed);torch.set_num_threads(args.cpu_threads)
    if min(args.steps,args.batch_size,args.accumulation,args.eval_every,args.log_every,args.val_samples)<1:
        raise ValueError('Training counts must be positive')
    if args.policy_metric_samples<0:raise ValueError('policy-metric-samples must be nonnegative')
    if args.lr<=0 or args.early_stop_patience<0 or args.early_stop_min_delta<0: raise ValueError('Invalid optimizer/early stop settings')
    if args.resume and args.init_checkpoint: raise ValueError('Choose resume OR init-checkpoint')
    if getattr(args,'prediction_probability',0): raise ValueError('Use --predicted-targets; stochastic legacy target mixing is retired')
    training=list(read_jsonl(args.train));validation=list(read_jsonl(args.val))
    if not training or not validation: raise ValueError('Nonempty train and validation data required')
    normalizations={r.get('normalization_contract') for r in training+validation}
    if len(normalizations)>1:raise ValueError('Mixed normalized/original policy data; prepare matching train and validation')
    if next(iter(normalizations)) is not None:
        print({'normalization_contract':next(iter(normalizations)),
               'normalized_training_pages':len({r['id'].rsplit('/',1)[0] for r in training})},flush=True)
    validate_splits(training+validation)
    if any(r['split']!='train' for r in training) or any(r['split']!='val' for r in validation): raise ValueError('Wrong split files')
    cfg=TreeConfig(**{k:getattr(args,k) for k in ('mode','size','hidden','layers','heads','max_nodes','token_grid')})
    if getattr(args,'stylesheets',False):
        from .css_owners import CONTRACT as OWNER_CONTRACT
        cfg.stylesheets=True;cfg.action_contract=OWNER_CONTRACT;cfg.__post_init__()
        cfg.existing_values_only=True
    tok=EditTokenizer(cfg.max_nodes)
    for row in training+validation:
        if row.get('hierarchy'):raise ValueError('Flat policy data is retired; use original DOM data')
        if row.get('teacher_strategy')!=cfg.action_contract: raise ValueError('Run visual-tree-prepare with matching --stylesheets setting')
        node_count=len(row['css_owners']) if cfg.stylesheets else len(row['current']['nodes'])
        if len(row['declaration_state'])!=node_count or node_count>cfg.max_nodes:
            raise ValueError('Invalid declaration state node count')
        if any(len(p)!=len(FIELDS) for p in row['declaration_state']):
            raise ValueError('CSS field schema changed; reprepare six-field size/margin labels')
        tok.encode(row['replacement_edit'],node_count)
        from .explicit_html import check_record_edit
        check_record_edit(row,row['replacement_edit'])
    signatures={'train':digest(args.train),'val':digest(args.val)}
    online=getattr(args,'online_corruption',False);targets=None
    if online:
        from .tree_online import ONLINE_CONTRACT,SAMPLING_CONTRACT,target_pool,asset_signatures
        if min(args.online_workers,args.online_prefetch,args.online_max_noise,args.online_attempts,args.online_timeout)<1:
            raise ValueError('Online worker, queue, retry and noise settings must be positive')
        cache_root=Path(args.online_targets or args.train).resolve().parent/'.online-target-cache'
        targets=target_pool(training,args.online_targets,cache_dir=cache_root,observation_size=cfg.size)
        signatures['online']={'contract':ONLINE_CONTRACT,'sampling':SAMPLING_CONTRACT,'assets':asset_signatures(targets)}
        if cfg.stylesheets and any(r.get('corruption_contract')!=ONLINE_CONTRACT for r in validation):
            raise ValueError('Validation uses old corruption; rerun prepare_stylesheet_policy.sh in a new output directory (reuse target assets/cache)')
    out=Path(args.out).resolve();settings={k:v for k,v in vars(args).items() if k not in ('out','steps','resume','init_checkpoint','policy_scale')}
    config=asdict(cfg)
    if not cfg.existing_values_only:config.pop('existing_values_only')
    if not cfg.stylesheets:
        config.pop('stylesheets');settings.pop('stylesheets',None)
    # Resource tuning does not change the index-seeded sample stream.
    settings={k:v for k,v in settings.items() if k not in ('online_workers','online_prefetch','online_timeout')}
    if not online: settings={k:v for k,v in settings.items() if not k.startswith('online_')}
    incomplete_restart=(out/'config.json').exists() and not (out/'last.pt').exists() and not args.resume
    guard_run(out,{'kind':KIND,'config':config,'data':signatures,'settings':settings},
              bool(args.resume) or incomplete_restart)
    if incomplete_restart:
        print(f'Restarting incomplete run without a checkpoint: {out}',flush=True)
    ck=torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=True) if args.resume or args.init_checkpoint else None
    if ck and cfg.semantic and ck.get('config',{}).get('abstraction_contract')!=BOUNDARY_CONTRACT:
        raise ValueError('Old occupancy-only policy checkpoint; train a new occupancy+boundary policy. Detector weights can be reused.')
    if ck and (ck.get('kind')!=KIND or TreeConfig(**ck['config'])!=cfg): raise ValueError('Incompatible policy checkpoint; detector reuse is separate')
    groups={r['group'] for r in training};hashes={r['source_sha'] for r in training}
    parser_groups=set();parser_hashes=set();provenance={}
    if args.predicted_targets:
        if not cfg.semantic: raise ValueError('Predicted abstraction applies only to abstract policy')
        for row in training+validation:
            if not row.get('predicted_target_elements'): raise ValueError('Missing detector target cache')
            source=row.get('parser_provenance')
            if isinstance(source,str):
                if source not in provenance: provenance[source]=read_json(source)
                source=provenance[source]
            if not source: raise ValueError('Missing detector training provenance')
            parser_groups.update(source.get('parser_training_groups',[]));parser_hashes.update(source.get('parser_training_hashes',[]))
    if ck:
        groups.update(ck.get('training_groups',[]));hashes.update(ck.get('training_hashes',[]))
        parser_groups.update(ck.get('parser_training_groups',[]));parser_hashes.update(ck.get('parser_training_hashes',[]))
    if (groups|parser_groups)&{r['group'] for r in validation} or (hashes|parser_hashes)&{r['source_sha'] for r in validation}:
        raise ValueError('Validation overlaps policy/detector training')
    net=TreePolicy(cfg,not args.no_pretrained and ck is None).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01)
    rng=random.Random(args.seed);start=0;best=float('inf');early={'reference_loss':float('inf'),'bad_validations':0,'stopped':False}
    if ck: net.load_state_dict(ck['model'])
    if args.resume:
        if ck['data_signature']!=signatures: raise ValueError('Resume data changed')
        optimizer.load_state_dict(ck['optimizer']);rng.setstate(ck['rng']);torch.set_rng_state(ck['torch_rng'].cpu())
        if device.startswith('cuda') and ck.get('cuda_rng'): torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start=ck['step'];best=ck['best'];early=ck['early_stopping']
        if early['stopped']: print('Early stopping already completed; keeping best.pt',flush=True);return
        if start>=args.steps: print(f'Already completed {start} steps; no online workers started',flush=True);return
    selected=validation_sample(validation,args.val_samples)
    metric_rows=validation_sample(selected,min(args.policy_metric_samples,len(selected)),seed=90211) if args.policy_metric_samples else []
    metric_targets=validation_target_states(metric_rows,out,signatures['val'],cfg.stylesheets) if metric_rows else []
    write_json(out/'manifest.json',{'kind':KIND,'config':asdict(cfg),'settings':vars(args),'data':signatures,
                                 'parameters':sum(p.numel() for p in net.parameters()),
                                 'online_target_pages':len(targets) if targets else 0})
    def amp(): return torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') and args.bf16 else contextlib.nullcontext()
    def loss(rows): return net.loss(*batch_inputs(net,rows,device,args.predicted_targets),[r['replacement_edit'] for r in rows])
    def log(name,row):
        with (out/name).open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(row,flush=True)
    with contextlib.ExitStack() as stack:
        stream=None
        if online:
            from .tree_online import OnlineStream
            cursor=ck.get('online_cursor') if args.resume else 0
            if cursor is None: raise ValueError('Online resume requires a saved sample cursor')
            stream=stack.enter_context(OnlineStream(targets,cfg.mode,args.seed,args.online_max_noise,
                args.online_workers,args.online_prefetch,args.online_attempts,args.online_timeout,cursor))
        wait_seconds=0.;producer_seconds=0.;sample_count=0;retry_count=0;remove_count=0;remaining_total=0;recent_errors=[]
        rule_count=0;probe_rejections=0
        property_counts={field:0 for field in FIELDS};path_length_total=0;mutation_total=0
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        for step in range(start+1,args.steps+1):
            net.train();optimizer.zero_grad(set_to_none=True);total=0.
            for _ in range(args.accumulation):
                wait_start=time.perf_counter()
                rows=stream.take(args.batch_size) if stream else [rng.choice(training) for _ in range(args.batch_size)]
                wait_seconds+=time.perf_counter()-wait_start
                if stream:
                    for row in rows:
                        info=row['online'];sample_count+=1;retry_count+=info['retries']
                        remove_count+=not row['replacement_edit'][2]
                        remaining_total+=row['symbolic_distance']
                        property_counts[row['replacement_edit'][1]]+=1
                        path_length_total+=info['path_length'];mutation_total+=len(info['corruptions'])
                        producer_seconds+=info['seconds']
                        recent_errors=(recent_errors+info['errors'])[-5:]
                        rule_count+=info.get('teacher_owner')=='rule'
                        probe_rejections+=info.get('probe_rejections',0)
                with amp(): value,_=loss(rows)
                if not torch.isfinite(value): raise RuntimeError('Nonfinite token CE')
                (value/args.accumulation).backward();total+=float(value.detach())/args.accumulation
            nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            if step%args.log_every==0 or step==args.steps:
                elapsed=time.perf_counter()-started;session_examples=(step-start)*args.batch_size*args.accumulation
                metrics={'step':step,'loss':total,'token_ce':total,'elapsed_s':elapsed,
                         'learning_rate':optimizer.param_groups[0]['lr'],
                         'examples_seen':step*args.batch_size*args.accumulation,
                         'examples_per_second':session_examples/max(elapsed,1e-9)}
                if device.startswith('cuda'):metrics['peak_cuda_bytes']=torch.cuda.max_memory_allocated(device)
                if stream:
                    metrics.update(online_corruption_contract=ONLINE_CONTRACT,
                        online_sampling_contract=SAMPLING_CONTRACT,
                        online_samples=stream.cursor,data_wait_s=wait_seconds,
                        producer_seconds=producer_seconds,online_retries=retry_count,
                        online_teacher_remove_rate=remove_count/max(1,sample_count),
                        online_mean_remaining_edits=remaining_total/max(1,sample_count),
                        online_mean_path_length=path_length_total/max(1,sample_count),
                        online_mean_mutations=mutation_total/max(1,sample_count),
                        online_teacher_property_distribution={field:property_counts[field]/max(1,sample_count) for field in FIELDS},
                        online_recent_errors=recent_errors)
                    if cfg.stylesheets:metrics.update(online_teacher_rule_rate=rule_count/max(1,sample_count),
                        online_probe_rejections=probe_rejections)
                log('train.jsonl',metrics)
            if step%args.eval_every==0 or step==args.steps:
                net.eval();weighted=0.;count=0
                cohorts={'all':selected}
                if cfg.stylesheets:
                    cohorts={}
                    for row in selected:
                        edit=row['replacement_edit']
                        key=row['css_owners'][edit[0]]['kind']+'/'+('SET' if edit[2] else 'REMOVE')
                        cohorts.setdefault(key,[]).append(row)
                breakdown={}
                with torch.no_grad():
                    for key,cohort in cohorts.items():
                        subtotal=0.;tokens=0
                        for offset in range(0,len(cohort),args.batch_size):
                            with amp(): value,n=loss(cohort[offset:offset+args.batch_size])
                            subtotal+=float(value)*n;tokens+=n
                        weighted+=subtotal;count+=tokens
                        breakdown[key]={'n':len(cohort),'tokens':tokens,'token_ce':subtotal/tokens}
                val=weighted/count;improved=val<best;best=min(best,val)
                early=update_early_stopping(early,val,args.early_stop_patience,args.early_stop_min_delta)
                metrics={'step':step,'validation_loss':val,'validation_tokens':count,
                         'validation_n':len(selected),'best':best,'early_stopping':early}
                if cfg.stylesheets:metrics['validation_by_teacher']=breakdown
                if metric_rows:
                    metric_started=time.perf_counter()
                    metrics.update(symbolic_policy_metrics(net,metric_rows,metric_targets,device,
                                                           args.predicted_targets,amp))
                    metrics['symbolic_metric_seconds']=time.perf_counter()-metric_started
                log('train.jsonl',metrics)
                with (out/'validation.jsonl').open('a') as validation_file: validation_file.write(json.dumps(metrics)+'\n')
                state={'kind':KIND,'config':config,'model':net.state_dict(),'optimizer':optimizer.state_dict(),
                       'step':step,'best':best,'early_stopping':early,'data_signature':signatures,
                       'online_cursor':stream.cursor if stream else None,
                       'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                       'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,
                       'training_groups':sorted(groups),'training_hashes':sorted(hashes),
                       'parser_training_groups':sorted(parser_groups),'parser_training_hashes':sorted(parser_hashes)}
                for name in (['last.pt','best.pt'] if improved else ['last.pt']):
                    temporary=out/(name+'.tmp');torch.save(state,temporary);temporary.replace(out/name)
                if early['stopped']: break

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
from .tree_edits import CONTRACT, FIELDS, EditTokenizer
from .visual import SpatialEncoder, elements, semantic_masks, image_tensor
from .visual_data import validate_splits
from .web_experiment import digest, guard_run

KIND = 'visual-policy-css-tree-v1'


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

    def __post_init__(self):
        if self.mode not in ('abstract','screenshot') or self.action_contract != CONTRACT or self.policy_head != 'replacement':
            raise ValueError('Replacement policy contract mismatch')
        if self.size < 32 or min(self.hidden,self.layers,self.heads,self.max_nodes,self.token_grid)<1 or self.hidden%self.heads:
            raise ValueError('Invalid model dimensions')

    @property
    def semantic(self): return self.mode == 'abstract'

    @property
    def observation_contract(self): return 'semantic-mask-pair-diff-v1' if self.semantic else 'rgb-pair-diff-v1'


class TreePolicy(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__(); self.cfg=cfg; self.tokenizer=EditTokenizer(cfg.max_nodes)
        self.vision=SpatialEncoder(cfg.hidden,cfg.token_grid,pretrained,12 if cfg.semantic else 9)
        self.token=nn.Embedding(len(self.tokenizer.tokens),cfg.hidden)
        self.byte=nn.Embedding(257,cfg.hidden,padding_idx=0)
        self.css_encoder=nn.GRU(cfg.hidden,cfg.hidden,batch_first=True)
        self.css_projection=nn.Linear(6*cfg.hidden,cfg.hidden)
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
        declarations=torch.zeros(b,n,6,64,dtype=torch.long,device=device)
        for i,state in enumerate(states):
            if len(state)>self.cfg.max_nodes: raise ValueError('Too many DOM nodes')
            for j,props in enumerate(state):
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
            allowed=self.tokenizer.allowed(prefix,len(states[0]))
            if not allowed: raise ValueError('No grammar-valid continuation')
            tokens=torch.tensor([[self.tokenizer.ids['BOS'],*prefix]],device=target.device)
            scores=self.logits(memory,padding,tokens)[0,-1]
            next_id=allowed[int(scores[allowed].argmax())];prefix.append(next_id)
            if next_id==self.tokenizer.ids['EOS']: return self.tokenizer.decode(prefix,len(states[0]))
        raise ValueError('Replacement exceeded grammar token limit')


def tree_batch(rows,device,max_nodes):
    """Current tree/geometry only; no retired numeric-delta legality mask."""
    features=[encode(r['current'],[{'viewport':r['viewport'],'target':{}}],max_nodes,[r['current_boxes']]) for r in rows]
    batch=collate(features,device);b,n=batch['mask'].shape
    rois=torch.zeros(b,n,4,device=device)
    for i,row in enumerate(rows):
        values=[];scale=max(row['viewport'])
        for node in row['current']['nodes']:
            x,y,w,h=row['current_boxes'][node['id']];values.append([x/scale,y/scale,(x+w)/scale,(y+h)/scale])
        rois[i,:len(values)]=torch.tensor(values,device=device).clamp(0,1)
    batch['roi']=rois
    return batch


def batch_inputs(net,rows,device,predicted=False):
    cfg=net.cfg;targets=[];currents=[]
    for row in rows:
        if cfg.semantic:
            if predicted:
                if not row.get('predicted_target_elements'): raise ValueError('Missing frozen-detector cache')
                items=read_json(row['predicted_target_elements'])
            else: items=row['target_elements']
            targets.append(semantic_masks(items,row['viewport'],cfg.size))
            currents.append(semantic_masks(elements(row['current'],row['current_boxes'],row['viewport']),row['viewport'],cfg.size))
        else:
            targets.append(image_tensor(row['target_image'],cfg.size));currents.append(image_tensor(row['current_image'],cfg.size))
    batch=tree_batch(rows,device,cfg.max_nodes)
    return batch,torch.stack(targets).to(device),torch.stack(currents).to(device),[r['declaration_state'] for r in rows]


def train(args):
    from .visual_train import validation_sample, update_early_stopping
    device=select_device(args.device);seed_all(args.seed);torch.set_num_threads(args.cpu_threads)
    if min(args.steps,args.batch_size,args.accumulation,args.eval_every,args.log_every,args.val_samples)<1:
        raise ValueError('Training counts must be positive')
    if args.lr<=0 or args.early_stop_patience<0 or args.early_stop_min_delta<0: raise ValueError('Invalid optimizer/early stop settings')
    if args.resume and args.init_checkpoint: raise ValueError('Choose resume OR init-checkpoint')
    if getattr(args,'prediction_probability',0): raise ValueError('Use --predicted-targets; stochastic legacy target mixing is retired')
    training=list(read_jsonl(args.train));validation=list(read_jsonl(args.val))
    if not training or not validation: raise ValueError('Nonempty train and validation data required')
    validate_splits(training+validation)
    if any(r['split']!='train' for r in training) or any(r['split']!='val' for r in validation): raise ValueError('Wrong split files')
    cfg=TreeConfig(**{k:getattr(args,k) for k in ('mode','size','hidden','layers','heads','max_nodes','token_grid')})
    tok=EditTokenizer(cfg.max_nodes)
    for row in training+validation:
        if row.get('teacher_strategy')!=CONTRACT: raise ValueError('Run visual-tree-prepare; legacy labels are not declaration replacements')
        if len(row['declaration_state'])!=len(row['current']['nodes']) or len(row['current']['nodes'])>cfg.max_nodes:
            raise ValueError('Invalid declaration state node count')
        tok.encode(row['replacement_edit'],len(row['current']['nodes']))
    signatures={'train':digest(args.train),'val':digest(args.val)}
    out=Path(args.out).resolve();settings={k:v for k,v in vars(args).items() if k not in ('out','steps','resume','init_checkpoint')}
    guard_run(out,{'kind':KIND,'config':asdict(cfg),'data':signatures,'settings':settings},bool(args.resume))
    ck=torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=True) if args.resume or args.init_checkpoint else None
    if ck and (ck.get('kind')!=KIND or ck['config']!=asdict(cfg)): raise ValueError('Incompatible policy checkpoint; detector reuse is separate')
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
    selected=validation_sample(validation,args.val_samples)
    write_json(out/'manifest.json',{'kind':KIND,'config':asdict(cfg),'settings':vars(args),'data':signatures,
                                 'parameters':sum(p.numel() for p in net.parameters())})
    def amp(): return torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') and args.bf16 else contextlib.nullcontext()
    def loss(rows): return net.loss(*batch_inputs(net,rows,device,args.predicted_targets),[r['replacement_edit'] for r in rows])
    def log(name,row):
        with (out/name).open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(row,flush=True)
    started=time.perf_counter()
    for step in range(start+1,args.steps+1):
        net.train();optimizer.zero_grad(set_to_none=True);total=0.
        for _ in range(args.accumulation):
            rows=[rng.choice(training) for _ in range(args.batch_size)]
            with amp(): value,_=loss(rows)
            if not torch.isfinite(value): raise RuntimeError('Nonfinite token CE')
            (value/args.accumulation).backward();total+=float(value.detach())/args.accumulation
        nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if step%args.log_every==0 or step==args.steps:
            log('train.jsonl',{'step':step,'loss':total,'token_ce':total,'elapsed_s':time.perf_counter()-started})
        if step%args.eval_every==0 or step==args.steps:
            net.eval();weighted=0.;count=0
            with torch.no_grad():
                for offset in range(0,len(selected),args.batch_size):
                    with amp(): value,n=loss(selected[offset:offset+args.batch_size])
                    weighted+=float(value)*n;count+=n
            val=weighted/count;improved=val<best;best=min(best,val)
            early=update_early_stopping(early,val,args.early_stop_patience,args.early_stop_min_delta)
            metrics={'step':step,'validation_loss':val,'validation_tokens':count,
                     'validation_n':len(selected),'best':best,'early_stopping':early}
            log('train.jsonl',metrics)
            with (out/'validation.jsonl').open('a') as stream: stream.write(json.dumps(metrics)+'\n')
            state={'kind':KIND,'config':asdict(cfg),'model':net.state_dict(),'optimizer':optimizer.state_dict(),
                   'step':step,'best':best,'early_stopping':early,'data_signature':signatures,
                   'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                   'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,
                   'training_groups':sorted(groups),'training_hashes':sorted(hashes),
                   'parser_training_groups':sorted(parser_groups),'parser_training_hashes':sorted(parser_hashes)}
            for name in (['last.pt','best.pt'] if improved else ['last.pt']):
                temporary=out/(name+'.tmp');torch.save(state,temporary);temporary.replace(out/name)
            if early['stopped']: break

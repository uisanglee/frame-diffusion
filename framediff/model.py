from __future__ import annotations
from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from .ir import FIELDS, LIMITS, ROLES, execute, editable

@dataclass
class ModelConfig:
    hidden: int=256
    layers: int=4
    heads: int=8
    dropout: float=.1
    tree_bias: bool=True
    max_nodes: int=128
    conditioning: str='boxes'

    def __post_init__(self):
        if self.hidden%self.heads:raise ValueError('hidden must be divisible by heads')
        if not 0<=self.dropout<1:raise ValueError('dropout must be in [0,1)')
        if self.conditioning not in ('boxes','plan'):raise ValueError('Unknown conditioning')

def encode(current,observations,max_nodes=128,current_frames=None):
    nodes=current['nodes']; n=len(nodes)
    if n>max_nodes:
        raise ValueError(f"{n} nodes exceeds max_nodes={max_nodes}; do not silently truncate")
    ids={a['id']:i for i,a in enumerate(nodes)}
    props=torch.tensor([[a['props'][f] for f in FIELDS] for a in nodes],dtype=torch.long)
    roles=torch.tensor([ROLES.index(a['role']) if a['role'] in ROLES else len(ROLES)-1 for a in nodes])
    # Hash names deterministically; no Python process-dependent hash().
    names=torch.tensor([int(hashlib.sha256(a.get('name',a['id']).encode()).hexdigest()[:8],16)%4096 for a in nodes])
    parents=[ids.get(a['parent'],-1) for a in nodes]
    depths=[]
    for i in range(n):
        d=0; p=parents[i]
        while p>=0:
            d+=1; p=parents[p]
        depths.append(d)
    relation=torch.zeros(n,n,dtype=torch.long)
    for i in range(n):
        for j in range(n):
            relation[i,j]=1 if i==j else 2 if parents[j]==i else 3 if parents[i]==j else 4 if parents[i]==parents[j] else 0
    geometry=[]
    for index,obs in enumerate(observations):
        w,h=obs['viewport']; scale=[w,h,w,h]
        boxes=execute(current,(w,h)) if current_frames is None else current_frames[index]
        rows=[]
        for a in nodes:
            cur=boxes[a['id']]; tgt=obs['target'].get(a['id'])
            target=tgt if tgt is not None else [0,0,0,0]
            rows.append([cur[k]/scale[k] for k in range(4)]+[target[k]/scale[k] for k in range(4)]+
                        [(target[k]-cur[k])/scale[k] if tgt else 0 for k in range(4)]+[float(tgt is not None),w/1024,h/1024])
        geometry.append(rows)
    legal=torch.zeros(n,len(FIELDS),257,dtype=torch.bool)
    for i,f in editable(current):
        lo,hi=LIMITS[f]; j=FIELDS.index(f)
        legal[i,j,lo:hi+1]=True
        legal[i,j,props[i,j]]=False
    return {'props':props,'roles':roles,'names':names,'depth':torch.tensor(depths).clamp_max(32),
            'relation':relation,'geometry':torch.tensor(geometry,dtype=torch.float32),
            'legal':legal,'mask':torch.ones(n,dtype=torch.bool)}

def collate(features,device='cpu'):
    batch=len(features); n=max(len(x['roles']) for x in features); views=max(x['geometry'].shape[0] for x in features)
    out={}
    for key in ('props','roles','names','depth','relation','geometry','legal','mask'):
        sample=features[0][key]
        if key=='relation': shape=(batch,n,n)
        elif key=='geometry': shape=(batch,views,n,16)
        else: shape=(batch,n,*sample.shape[1:])
        out[key]=torch.zeros(shape,dtype=sample.dtype)
        for b,f in enumerate(features):
            m=len(f['roles'])
            if key=='relation': out[key][b,:m,:m]=f[key]
            elif key=='geometry':
                v=f[key].shape[0]; out[key][b,:v,:m,:15]=f[key];out[key][b,:v,:m,15]=1
            else: out[key][b,:m]=f[key]
    if 'plan' in features[0]:
        if any('plan' not in f for f in features):raise ValueError('Mixed conditioning in batch')
        out['plan']=torch.zeros(batch,n,features[0]['plan'].shape[-1])
        for b,f in enumerate(features):out['plan'][b,:len(f['roles'])]=f['plan']
        out['plan_loss']=torch.stack([f['plan_loss'] for f in features])
    return {k:v.to(device) for k,v in out.items()}

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.heads=cfg.heads; self.use_tree=cfg.tree_bias
        self.norm1=nn.LayerNorm(cfg.hidden);self.norm2=nn.LayerNorm(cfg.hidden)
        self.qkv=nn.Linear(cfg.hidden,cfg.hidden*3); self.proj=nn.Linear(cfg.hidden,cfg.hidden)
        self.bias=nn.Embedding(5,cfg.heads)
        self.mlp=nn.Sequential(nn.Linear(cfg.hidden,cfg.hidden*4),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(cfg.hidden*4,cfg.hidden))
        self.dropout=cfg.dropout
    def forward(self,x,relation,mask):
        b,n,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        bias=self.bias(relation).permute(0,3,1,2) if self.use_tree else x.new_zeros(b,self.heads,n,n)
        bias=bias.masked_fill(~mask[:,None,None,:],float('-inf'))
        y=nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=self.dropout if self.training else 0.)
        x=x+self.proj(y.transpose(1,2).reshape(b,n,d))
        return x+self.mlp(self.norm2(x))

class EditDenoiser(nn.Module):
    """Tree mutation denoiser; not the original LayoutDM or a DDPM reproduction.

    Iterative stochastic tree corruption is learned with edit supervision. Like the
    reference tree-edit policy this model need not know the unknown noise timestep.
    """
    def __init__(self,cfg=ModelConfig()):
        super().__init__();self.cfg=cfg
        self.prop=nn.ModuleList(nn.Embedding(257,cfg.hidden) for _ in FIELDS)
        self.role=nn.Embedding(len(ROLES),cfg.hidden);self.name=nn.Embedding(4096,cfg.hidden);self.depth=nn.Embedding(33,cfg.hidden)
        self.geometry=nn.Sequential(nn.Linear(15,cfg.hidden),nn.GELU(),nn.Linear(cfg.hidden,cfg.hidden))
        if cfg.conditioning=='plan':
            from .plans import PLAN_DIM
            self.plan_encoder=nn.Sequential(nn.Linear(PLAN_DIM,cfg.hidden),nn.GELU(),nn.Linear(cfg.hidden,cfg.hidden))
        self.blocks=nn.ModuleList(Block(cfg) for _ in range(cfg.layers));self.norm=nn.LayerNorm(cfg.hidden)
        self.action=nn.Linear(cfg.hidden,len(FIELDS)*257)
        self.stop=nn.Linear(cfg.hidden,1)
        self.value=nn.Sequential(nn.Linear(cfg.hidden,1),nn.Softplus())
        self.box=nn.Linear(cfg.hidden,4)
    def forward(self,b):
        if ('plan' in b)!=(self.cfg.conditioning=='plan'):
            raise ValueError('Checkpoint conditioning mismatch: plan checkpoints require plan features')
        x=self.role(b['roles'])+self.name(b['names'])+self.depth(b['depth'])
        if self.cfg.conditioning=='plan':x=x+self.plan_encoder(b['plan'])
        x=x+sum(e(b['props'][:,:,j]) for j,e in enumerate(self.prop))/len(FIELDS)
        g=b['geometry']; gmask=g[:,:,:,15:16]
        x=x+(self.geometry(g[:,:,:,:15])*gmask).sum(1)/gmask.sum(1).clamp_min(1)
        for block in self.blocks: x=block(x,b['relation'],b['mask'])
        x=self.norm(x); pooled=(x*b['mask'][...,None]).sum(1)/b['mask'].sum(1,keepdim=True)
        logits=self.action(x).reshape(*x.shape[:2],len(FIELDS),257)
        logits=logits.masked_fill(~b['legal'],float('-inf'))
        return {'logits':logits,'stop':self.stop(pooled).squeeze(-1),'value':self.value(pooled).squeeze(-1),
                'boxes':self.box(x)}

def action_index(edit,n):
    if edit is None:return n*len(FIELDS)*257
    i,f,v=edit;return (i*len(FIELDS)+FIELDS.index(f))*257+v

def flat_logits(output):
    return torch.cat([output['logits'].flatten(1),output['stop'][:,None]],dim=1)

def load_model(path,device='cpu'):
    ckpt=torch.load(path,map_location=device,weights_only=True)
    model=EditDenoiser(ModelConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model,ckpt

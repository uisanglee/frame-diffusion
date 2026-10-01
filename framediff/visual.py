"""Shared, ID-free abstraction contract and image-conditioned edit policy.

Target box predictions ONLY render an image. They never enter policy geometry,
an objective, or candidate selection. Current DOM geometry is observed input.
"""
from dataclasses import dataclass
import io
import math

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F

from .ir import FIELDS, ROLES
from .model import encode, collate, Block, ModelConfig

CONTRACT = 'visible-ui-v1'
CLASSES = ('background', 'text', 'image', 'control', 'painted-region')
COLORS = ('white', '#2864b4', '#26946c', '#df7832', '#b49ac7')
EDIT_FIELDS = ('width', 'height', 'dx', 'dy')
DELTAS = (-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64)
ACTIONS = len(EDIT_FIELDS)*len(DELTAS)


def annotate(browser, tree):
    """Visible primitives and painted regions, never invisible DOM wrappers.

    These labels are observable appearance categories, NOT semantic DOM roles.
    A control's descendants are suppressed to avoid double-counting its text.
    """
    labels = browser.page.evaluate('''() => Object.fromEntries(
      [...document.querySelectorAll('[data-fd-id]')].map(e=>{
        const id=e.getAttribute('data-fd-id'), s=getComputedStyle(e);
        const primitive='button,input,textarea,select,img,svg,canvas,video';
        if(e.parentElement?.closest(primitive))return [id,0];
        if(e.matches('button,input,textarea,select'))return [id,3];
        if(e.matches('img,svg,canvas,video'))return [id,2];
        const ownText=[...e.childNodes].some(n=>n.nodeType===3 && n.textContent.trim());
        if(ownText && !e.matches('body,html'))return [id,1];
        const color=s.backgroundColor;
        const painted=color!=='transparent' && color!=='rgba(0, 0, 0, 0)';
        const border=['Top','Bottom','Left','Right'].some(k=>parseFloat(s['border'+k+'Width'])>0 && s['border'+k+'Style']!=='none');
        return [id,!e.matches('body,html')&&(painted||border)?4:0];
      }))''')
    for node in tree['nodes']: node['visual_class'] = labels.get(node['id'], 0)
    return tree


def elements(tree, boxes, viewport):
    w, h = viewport
    result = []
    for node in tree['nodes']:
        label = node.get('visual_class', 0)
        if not label: continue
        x, y, bw, bh = boxes[node['id']]
        x1, y1, x2, y2 = max(0., x), max(0., y), min(w, x+bw), min(h, y+bh)
        if x2>x1 and y2>y1:
            result.append({'box':[x1, y1, x2, y2], 'label':label})
    return result


def abstract_image(items, viewport, size=384):
    """Canonical color, filled rectangles, no IDs/text; deterministic overlap order."""
    w,h=viewport; scale=min(size/w,size/h)
    canvas=Image.new('RGB',(size,size),'white'); draw=ImageDraw.Draw(canvas)
    def key(item):
        x1,y1,x2,y2=item['box']
        return (-(x2-x1)*(y2-y1),item['label'],x1,y1,x2,y2)
    for item in sorted(items,key=key):
        box=[float(v)*scale for v in item['box']]
        if all(math.isfinite(v) for v in box) and box[2]>box[0] and box[3]>box[1]:
            draw.rectangle(box,fill=COLORS[item['label']])
    return canvas


def image_tensor(image, size):
    if isinstance(image,(str,bytes)):
        image=Image.open(io.BytesIO(image) if isinstance(image,bytes) else image)
    image=image.convert('RGB'); scale=min(size/image.width,size/image.height)
    canvas=Image.new('RGB',(size,size),'white')
    canvas.paste(image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale)))),(0,0))
    tensor=torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1).float()/255
    return (tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def current_features(tree, boxes, viewport, max_nodes):
    feature=encode(tree,[{'viewport':viewport,'target':{}}],max_nodes,[boxes])
    # ROI coordinates match the aspect-preserving square letterbox used by images.
    divisor=max(viewport)
    feature['roi']=torch.tensor([[b[0]/divisor,b[1]/divisor,(b[0]+b[2])/divisor,(b[1]+b[3])/divisor]
                                 for n in tree['nodes'] for b in [boxes[n['id']]]]).clamp(0,1)
    feature['action_legal']=torch.ones(len(tree['nodes']),len(EDIT_FIELDS),len(DELTAS),dtype=torch.bool)
    feature['action_legal'][0]=False
    for i,n in enumerate(tree['nodes'][1:],1):
        for j,axis in ((0,2),(1,3)):
            for k,delta in enumerate(DELTAS):
                if boxes[n['id']][axis]+delta<=0:feature['action_legal'][i,j,k]=False
    return feature


def visual_batch(features, device):
    batch=collate(features,device);n=batch['mask'].shape[1]
    for key in ('roi','action_legal'):
        sample=features[0][key]
        tensor=torch.zeros(len(features),n,*sample.shape[1:],dtype=sample.dtype)
        for i,f in enumerate(features):tensor[i,:len(f['roles'])]=f[key]
        batch[key]=tensor.to(device)
    return batch


@dataclass
class VisualConfig:
    mode: str = 'abstract'
    size: int = 384
    hidden: int = 128
    layers: int = 3
    heads: int = 4
    max_nodes: int = 128
    token_grid: int = 12
    contract: str = CONTRACT

    def __post_init__(self):
        if self.mode not in ('abstract','screenshot') or self.contract!=CONTRACT:raise ValueError('Visual policy contract mismatch')
        if self.size<32 or min(self.hidden,self.layers,self.heads,self.max_nodes,self.token_grid)<1 or self.hidden%self.heads:
            raise ValueError('Invalid visual model dimensions')


class SpatialEncoder(nn.Module):
    """Pretrained ResNet18 with top-down stride-4/8/16 features and spatial tokens."""
    def __init__(self, hidden, grid, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision.ops.misc import FrozenBatchNorm2d
        backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None,
                          norm_layer=FrozenBatchNorm2d)
        self.stem=nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool)
        self.stages=nn.ModuleList([backbone.layer1,backbone.layer2,backbone.layer3])
        self.projections=nn.ModuleList(nn.Conv2d(c,hidden,1) for c in (64,128,256))
        self.position=nn.Linear(3,hidden);self.grid=grid

    def forward(self, images):
        x=self.stem(images);maps=[]
        for stage,proj in zip(self.stages,self.projections):
            x=stage(x);maps.append(proj(x))
        for i in (1,0):maps[i]=maps[i]+F.interpolate(maps[i+1],size=maps[i].shape[-2:],mode='bilinear',align_corners=False)
        tokens=[]
        for level,feature in enumerate(maps):
            grid=min(self.grid,*feature.shape[-2:]);pooled=F.adaptive_avg_pool2d(feature,(grid,grid))
            axis=(torch.arange(grid,device=feature.device,dtype=feature.dtype)+.5)/grid
            y,x=torch.meshgrid(axis,axis,indexing='ij')
            position=torch.stack([x,y,torch.full_like(x,level/2)],-1).reshape(-1,3)
            tokens.append(pooled.flatten(2).transpose(1,2)+self.position(position)[None])
        return torch.cat(tokens,1),maps[0]


class VisualPolicy(nn.Module):
    def __init__(self,cfg,pretrained=True):
        super().__init__();self.cfg=cfg
        self.vision=SpatialEncoder(cfg.hidden,cfg.token_grid,pretrained)
        self.role=nn.Embedding(len(ROLES),cfg.hidden);self.name=nn.Embedding(4096,cfg.hidden)
        self.depth=nn.Embedding(33,cfg.hidden)
        self.props=nn.ModuleList(nn.Embedding(257,cfg.hidden) for _ in FIELDS)
        self.geometry=nn.Linear(6,cfg.hidden)
        self.modality=nn.Parameter(torch.randn(2,cfg.hidden)*.02)
        self.cross=nn.MultiheadAttention(cfg.hidden,cfg.heads,batch_first=True)
        self.norm=nn.LayerNorm(cfg.hidden)
        bc=ModelConfig(hidden=cfg.hidden,layers=cfg.layers,heads=cfg.heads,dropout=0.)
        self.blocks=nn.ModuleList(Block(bc) for _ in range(cfg.layers))
        self.action=nn.Linear(cfg.hidden,ACTIONS);self.stop=nn.Linear(cfg.hidden,1)

    def encode_image(self,image):return self.vision(image)

    def decode(self,batch,target_tokens,current_tokens,current_map):
        from torchvision.ops import roi_align
        x=self.role(batch['roles'])+self.name(batch['names'])+self.depth(batch['depth'])
        x=x+sum(e(batch['props'][:,:,j]) for j,e in enumerate(self.props))/len(FIELDS)
        g=batch['geometry'][:,0]
        x=x+self.geometry(torch.cat([g[:,:,:4],g[:,:,13:15]],-1))
        # This samples CURRENT image pixels for each CURRENT DOM element, not target boxes.
        rois=[r*current_map.shape[-1] for r in batch['roi']]
        local=roi_align(current_map,rois,output_size=3,spatial_scale=1.,aligned=True).mean((-1,-2))
        x=x+local.reshape_as(x)
        memory=torch.cat([target_tokens+self.modality[0],current_tokens+self.modality[1]],1)
        x=x+self.cross(self.norm(x),memory,memory,need_weights=False)[0]
        for block in self.blocks:x=block(x,batch['relation'],batch['mask'])
        x=self.norm(x);pooled=(x*batch['mask'][...,None]).sum(1)/batch['mask'].sum(1,keepdim=True)
        logits=self.action(x).reshape(x.shape[0],x.shape[1],len(EDIT_FIELDS),len(DELTAS))
        legal=batch['action_legal'] & batch['mask'][:,:,None,None]
        return torch.cat([logits.masked_fill(~legal,float('-inf')).flatten(1),self.stop(pooled)],1)

    def forward(self,batch,target,current):
        target_tokens,_=self.encode_image(target);current_tokens,current_map=self.encode_image(current)
        return self.decode(batch,target_tokens,current_tokens,current_map)


def action_index(edit,n):
    if edit is None:return n*ACTIONS
    i,field,delta=edit
    return i*ACTIONS+EDIT_FIELDS.index(field)*len(DELTAS)+DELTAS.index(delta)


def decode_action(index,n):
    if index==n*ACTIONS:return None
    node,rest=divmod(index,ACTIONS);field,value=divmod(rest,len(DELTAS))
    return node,EDIT_FIELDS[field],DELTAS[value]


def load_policy(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('kind')!='visual-policy-v1':raise ValueError('Requires a visual policy checkpoint; legacy checkpoints are incompatible')
    model=VisualPolicy(VisualConfig(**checkpoint['config']),pretrained=False)
    model.load_state_dict(checkpoint['model']);return model.to(device).eval(),checkpoint

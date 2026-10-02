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
ACTION_CONTRACT = 'single-css-declaration-normalized-v3-no-padding-gap'
CLASSES = ('background', 'text', 'image', 'control', 'painted-region')
COLORS = ('white', '#2864b4', '#26946c', '#df7832', '#b49ac7')
MAX_ACTION_FRACTION = .05
# Numeric actions are fractions of the corresponding viewport axis.  Thus one
# action changes exactly one CSS declaration by at most 5% of the viewport.
NORMALIZED_DELTAS = (-.05, -.025, -.0125, -.00625, -.003125,
                       .003125, .00625, .0125, .025, .05)
NUMERIC_FIELDS = ('width','height',
                  'margin-left','margin-right','margin-top','margin-bottom')
CATEGORICAL_VALUES = {
    'flex-direction': ('row','row-reverse','column','column-reverse'),
    'justify-content': ('normal','flex-start','center','flex-end','space-between','space-around','space-evenly','start','end'),
    'align-items': ('normal','stretch','flex-start','center','flex-end','baseline','start','end'),
}
EDIT_FIELDS = NUMERIC_FIELDS + tuple(CATEGORICAL_VALUES)
ACTION_VALUES = {field:NORMALIZED_DELTAS for field in NUMERIC_FIELDS} | CATEGORICAL_VALUES
ACTION_SPECS = tuple((field,value) for field in EDIT_FIELDS for value in ACTION_VALUES[field])
ACTION_TO_INDEX = {action:i for i,action in enumerate(ACTION_SPECS)}
ACTIONS = len(ACTION_SPECS)


def action_axis(field):
    """Viewport axis used to turn a normalized numeric action into CSS pixels."""
    if field in ('width','margin-left','margin-right'):
        return 0
    if field in ('height','margin-top','margin-bottom'):
        return 1
    return None


def validate_action(edit, viewport=None):
    """Validate the v3 no-padding/gap action contract: one declaration per action.

    Numeric mutations are bounded by MAX_ACTION_FRACTION of one viewport axis;
    categorical mutations replace one value from a finite grammar.
    """
    if not isinstance(edit,(tuple,list)) or len(edit)!=3:raise ValueError('Action must be [node, field, value]')
    node,field,value=edit
    if type(node) is not int or node<0 or (field,value) not in ACTION_TO_INDEX:
        raise ValueError('Action is outside the structured CSS grammar')
    if field in NUMERIC_FIELDS:
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<abs(value)<=MAX_ACTION_FRACTION:
            raise ValueError('Numeric action exceeds normalized size limit')
        if viewport is not None and (len(viewport)!=2 or min(viewport)<=0):raise ValueError('Invalid viewport')
    return tuple(edit)


def action_size(edit):
    """Explicit complexity measure used by the structured-action contract."""
    _,field,value=validate_action(edit)
    return {'declarations':1,
            'normalized_magnitude':abs(float(value)) if field in NUMERIC_FIELDS else None}


def action_delta_px(field, value, viewport):
    validate_action((0,field,value),viewport)
    axis=action_axis(field)
    if axis is None:raise ValueError('Categorical action has no pixel delta')
    return float(value)*float(viewport[axis])


def candidate_fields(tree, index):
    """Syntactically legal declaration locations, independent of target geometry."""
    if index==0:return ()
    node_id=tree['nodes'][index]['id']
    has_children=any(n.get('parent')==node_id for n in tree['nodes'])
    base=('width','height','margin-left','margin-right','margin-top','margin-bottom')
    if not has_children:return base
    return base+('flex-direction','justify-content','align-items')


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
    legal=torch.zeros(len(tree['nodes']),ACTIONS,dtype=torch.bool)
    for i,n in enumerate(tree['nodes'][1:],1):
        for field in candidate_fields(tree,i):
            for value in ACTION_VALUES[field]:
                allowed=True
                if field in ('width','height'):
                    axis=0 if field=='width' else 1
                    allowed=boxes[n['id']][axis+2]+action_delta_px(field,value,viewport)>0
                if allowed:legal[i,ACTION_TO_INDEX[(field,value)]]=True
    feature['action_legal']=legal
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
    action_contract: str = ACTION_CONTRACT

    def __post_init__(self):
        if (self.mode not in ('abstract','screenshot') or self.contract!=CONTRACT or
                self.action_contract!=ACTION_CONTRACT):raise ValueError('Visual policy contract mismatch')
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
        logits=self.action(x)
        legal=batch['action_legal'] & batch['mask'][:,:,None]
        return torch.cat([logits.masked_fill(~legal,float('-inf')).flatten(1),self.stop(pooled)],1)

    def forward(self,batch,target,current):
        target_tokens,_=self.encode_image(target);current_tokens,current_map=self.encode_image(current)
        return self.decode(batch,target_tokens,current_tokens,current_map)


def action_index(edit,n):
    if edit is None:return n*ACTIONS
    i,field,value=validate_action(edit)
    return i*ACTIONS+ACTION_TO_INDEX[(field,value)]


def decode_action(index,n):
    if index==n*ACTIONS:return None
    node,action=divmod(index,ACTIONS);field,value=ACTION_SPECS[action]
    return node,field,value


def load_policy(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('kind')!='visual-policy-v3':raise ValueError('Requires a v3 no-padding/gap visual policy checkpoint; retrain legacy policies')
    model=VisualPolicy(VisualConfig(**checkpoint['config']),pretrained=False)
    model.load_state_dict(checkpoint['model']);return model.to(device).eval(),checkpoint

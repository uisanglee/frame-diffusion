from types import SimpleNamespace

import pytest
import torch

from framediff.tree_policy import TreeConfig, TreePolicy, resolve_policy_scale
from framediff.css_owners import CONTRACT


# The existing occupancy+boundary representation has 8 channels per image
# (24 pair/difference channels), not the former 4-channel representation.
@pytest.mark.parametrize('scale,expected', [('s',4804081),('m',15569957),('l',67125109)])
def test_scale_parameters_and_decoder_gradient(scale, expected):
    args=SimpleNamespace(policy_scale=scale,hidden=1,layers=1,heads=1)
    resolve_policy_scale(args)
    cfg=TreeConfig(hidden=args.hidden,layers=args.layers,heads=args.heads,max_nodes=512,
                   stylesheets=True,action_contract=CONTRACT,existing_values_only=True)
    torch.set_num_threads(2)
    model=TreePolicy(cfg,pretrained=False)
    assert sum(p.numel() for p in model.parameters())==expected
    memory=torch.randn(1,3,cfg.hidden)
    logits=model.logits(memory,torch.zeros(1,3,dtype=torch.bool),torch.zeros(1,2,dtype=torch.long))
    logits.square().mean().backward()
    assert torch.isfinite(model.output.weight.grad).all()


def test_no_preset_keeps_custom_dimensions():
    args=SimpleNamespace(hidden=64,layers=2,heads=4)
    resolve_policy_scale(args)
    assert (args.hidden,args.layers,args.heads)==(64,2,4)

from dataclasses import asdict
import pytest
import torch
from framediff.visual import VisualConfig, VisualPolicy, ACTIONS, current_features, visual_batch, action_index, load_policy
from framediff.ir import node


def test_hierarchical_normalization_gradients_padding_and_reload(tmp_path):
    torch.set_num_threads(2)
    cfg=VisualConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,
                     numeric_only=True,policy_head='hierarchical')
    net=VisualPolicy(cfg,False)
    tree={'version':1,'nodes':[node('page',None,'page'),node('card','page','card',width=80)]}
    boxes={'page':[0,0,320,240],'card':[16,20,80,30]}
    features=current_features(tree,boxes,[320,240],128)
    root=current_features({'version':1,'nodes':[tree['nodes'][0]]},{'page':boxes['page']},[320,240],128)
    batch=visual_batch([features,root],'cpu')
    images=torch.randn(2,3,32,32)
    logits=net(batch,images,images)
    assert torch.allclose(logits.exp().sum(-1),torch.ones(2),atol=1e-5)
    assert torch.isneginf(logits[0,60:81]).all()
    assert net.select(logits)[1]==2*ACTIONS
    loss=-logits[0,action_index((1,'width',.05),2)]-logits[1,-1]
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    for head in (net.operation,net.node_head,net.property_head,net.delta_head):
        assert head.weight.grad is not None
    path=tmp_path/'new.pt'
    torch.save({'kind':'visual-policy-v3','config':asdict(cfg),'model':net.state_dict()},path)
    restored,_=load_policy(path,'cpu')
    assert torch.allclose(logits,restored(batch,images,images))


def test_aggregate_changes_only_stop_decision():
    cfg=VisualConfig(numeric_only=True,decoding='joint')
    class Stub:
        pass
    net=Stub();net.cfg=cfg
    scores=torch.tensor([[0.,0.,.1]])
    assert VisualPolicy.select(net,scores).item()==2
    cfg.decoding='aggregate'
    assert VisualPolicy.select(net,scores).item()==0


def test_balanced_hierarchy_uses_visual_difference_for_operation():
    torch.set_num_threads(2);torch.manual_seed(4)
    cfg=VisualConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,numeric_only=True,
                     policy_head='hierarchical',training_scheme='balanced-v1')
    net=VisualPolicy(cfg,False)
    tree={'version':1,'nodes':[node('page',None,'page'),node('card','page','card',width=80)]}
    boxes={'page':[0,0,320,240],'card':[16,20,80,30]}
    batch=visual_batch([current_features(tree,boxes,[320,240],16)],'cpu')
    target=torch.randn(1,3,32,32,requires_grad=True);current=torch.randn(1,3,32,32,requires_grad=True)
    logits,parts=net(batch,target,current,return_components=True)
    assert set(parts)=={'operation','node','property','delta'}
    (-parts['operation'][0,1]).backward()
    assert target.grad.abs().sum()>0 and current.grad.abs().sum()>0
    assert net.operation[0].weight.shape[1]==cfg.hidden*4


def test_autoregressive_policy_uses_nine_channel_pair_and_token_conditioning(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(9)
    cfg=VisualConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,numeric_only=True,
                     policy_head='autoregressive',training_scheme='balanced-v1')
    net=VisualPolicy(cfg,False)
    assert net.vision.stem[0].in_channels==9
    tree={'version':1,'nodes':[node('page',None,'page'),node('card','page','card',width=80)]}
    boxes={'page':[0,0,320,240],'card':[16,20,80,30]}
    batch=visual_batch([current_features(tree,boxes,[320,240],16)],'cpu')
    target=torch.randn(1,3,32,32,requires_grad=True)
    current=torch.randn(1,3,32,32,requires_grad=True)
    logits,parts=net(batch,target,current,return_components=True)
    assert set(parts)=={'node','property','delta'}
    assert torch.allclose(logits.exp().sum(-1),torch.ones(1),atol=1e-5)
    assert torch.isneginf(logits[0,-1])
    loss=-(parts['node'][0,1]+parts['property'][0,1,0]+parts['delta'][0,1,0,9])/3
    loss.backward()
    assert target.grad.abs().sum()>0 and current.grad.abs().sum()>0
    assert net.node_query.weight.grad is not None
    assert net.property_token.weight.grad is not None
    path=tmp_path/'autoregressive.pt'
    torch.save({'kind':'visual-policy-v5-no-stop','config':asdict(cfg),'model':net.state_dict()},path)
    restored,_=load_policy(path,'cpu')
    with torch.no_grad():assert torch.allclose(logits,restored(batch,target,current))


@pytest.mark.parametrize('command',['visual-train-policy','visual-subset-numeric'])
def test_retired_delta_training_commands_are_not_exposed(command):
    from framediff.cli import main
    with pytest.raises(SystemExit):main([command])

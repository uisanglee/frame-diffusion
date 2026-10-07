import copy
from types import SimpleNamespace

import pytest
import torch

from framediff.css_diffusion import Kernel, numeric_slots, slot_edit
from framediff.tree_edits import FIELDS


def test_exact_chain_posterior_and_mixture():
    torch.set_num_threads(2)
    k = Kernel(bins=5, steps=4)
    assert torch.allclose(k.transitions.sum(-1), torch.ones(5,5,dtype=torch.float64))
    assert torch.allclose(k.cumulative[3],k.transitions[1]@k.transitions[2]@k.transitions[3])
    x0=torch.tensor([0,2,4]);xt=torch.tensor([1,2,3])
    posterior=k.posterior(x0,xt,3)
    for n in range(3):
        expected=torch.tensor([k.cumulative[2,x0[n],j]*k.transitions[3,j,xt[n]] for j in range(5)])
        expected/=expected.sum()
        assert torch.allclose(posterior[n],expected)
    logits=torch.randn(3,5,requires_grad=True)
    mixture=k.reverse(logits,xt,3)
    expected=sum(logits.double().softmax(-1)[:,i,None]*k.posterior(torch.full_like(x0,i),xt,3) for i in range(5))
    assert torch.allclose(mixture,expected)
    loss,_=k.loss(logits,x0,xt,3);loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    assert torch.allclose(k.posterior(x0,xt,1),torch.nn.functional.one_hot(x0,5).double())
    assert torch.allclose(k.reverse(logits,xt,1),logits.double().softmax(-1))


def test_quantization_preserves_topology_priority_and_units():
    state=[[['',''] for _ in FIELDS] for _ in range(2)]
    state[1]=[['301px','important'],['auto',''],['-21px',''],['10%',''],['2rem',''],['99999px','']]
    before=copy.deepcopy(state)
    slots,labels,errors=numeric_slots(state,FIELDS,129)
    assert state==before and len(slots)==3
    assert slot_edit(slots[0],labels[0],FIELDS,129)==[1,'width','304px','important']
    assert slots[-1]['unit']=='%' and all(e['absolute_error']>=0 for e in errors)


def test_abstraction_error_ignores_blank_background_and_empty_target():
    from framediff.css_diffusion_experiment import abstraction_error
    target=torch.zeros(4,32,32);target[0,0,0]=1
    assert abstraction_error(target,target)==0
    current=target.clone();current[0,0,0]=0;current[1,0,0]=1
    assert abstraction_error(target,current)==1  # wrong semantic class, same box
    current=target.clone();current[0,0,1]=1
    assert abstraction_error(target,current)==.5
    assert abstraction_error(torch.zeros_like(target),current) is None


def test_owner_normalization_accepts_text_only_and_rejects_topology():
    from framediff.css_diffusion_experiment import canonical_owners
    expected=[{'kind':'root','matches':[]},{'kind':'rule','block':0,'path':[1],
              'selector':'.a > .b','conditions':['@media (min-width:400px)'],'matches':['fd-1']}]
    actual=copy.deepcopy(expected);actual[1]['selector']='.a > .b';actual[1]['conditions']=['@media (min-width: 400px)']
    assert canonical_owners(expected,actual) is actual
    actual[1]['path']=[2]
    with pytest.raises(ValueError,match='topology'):canonical_owners(expected,actual)


@pytest.mark.browser
def test_prepare_train_resume_and_reverse_rollout(tmp_path, monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.css_diffusion_experiment import prepare,train,evaluate
    source=tmp_path/'source';source.mkdir()
    with HtmlBrowser() as browser:
        for split in ('train','val','test'):
            html='<html><head><style>.card{width:301px;height:64px;margin-left:10px}</style></head><body><div class="card">Hello</div></body></html>'
            dom=browser.snapshot(html,[500,300],max_nodes=8);tree,_,_=dom_tree(dom);annotate(browser,tree)
            target=source/(split+'.html');target.write_text(dom['html'])
            row=dict(id=split+'/t0-s0',group=split,source_sha=split,split=split,
                     current=tree,viewport=[500,300],target_html=str(target))
            write_jsonl(source/f'policy-{split}.jsonl',[row])
    data=tmp_path/'prepared';out=tmp_path/'run'
    prepare(SimpleNamespace(source=str(source),out=str(data),bins=129,limit=0,max_nodes=32,resume=False,
                            detector_checkpoint=None,device='cpu',threshold=.4))
    assert len(list(read_jsonl(data/'train.jsonl')))==1
    args=SimpleNamespace(data=str(data),out=str(out),mode='abstract',policy_scale='s',target_source='oracle',
        device='cpu',resume=False,no_pretrained=True,steps=1,batch_size=1,accumulation=1,eval_every=1,
        log_every=1,val_samples=1,diffusion_steps=3,seed=42,cpu_threads=2,size=64,max_nodes=32,token_grid=2,
        lr=1e-4,sigma_min=.35,sigma_max=2.,move_rate=.2,auxiliary_weight=.01)
    train(args)
    args.resume=True;args.steps=2;train(args)
    assert torch.load(out/'last.pt',weights_only=False)['step']==2
    evaluate(SimpleNamespace(data=str(data/'test.jsonl'),checkpoint=str(out/'best.pt'),out=str(tmp_path/'eval'),
                             device='cpu',limit=1,start_t=2,seed=1,cpu_threads=2,abstract_goal_threshold=-1))
    assert (tmp_path/'eval/0/repaired.html').exists()
    result=list(read_jsonl(tmp_path/'eval/results.jsonl'))[0]
    assert result['reverse_steps']==2 and result['stop_reason']=='step_budget'
    assert result['trace'][0]['next_t']==1 and 'changes' in result['trace'][0]
    import framediff.css_diffusion_experiment as experiment
    with monkeypatch.context() as patch:
        scores=iter([.5,0.])
        patch.setattr(experiment,'abstraction_error',lambda *a:next(scores))
        evaluate(SimpleNamespace(data=str(data/'test.jsonl'),checkpoint=str(out/'best.pt'),out=str(tmp_path/'early'),
                                 device='cpu',limit=1,start_t=2,seed=1,cpu_threads=2,abstract_goal_threshold=.01))
        result=list(read_jsonl(tmp_path/'early/results.jsonl'))[0]
        assert result['reverse_steps']==1 and result['stop_reason']=='abstract_goal'
    with monkeypatch.context() as patch:
        patch.setattr(experiment,'abstraction_error',lambda *a:0.)
        evaluate(SimpleNamespace(data=str(data/'test.jsonl'),checkpoint=str(out/'best.pt'),out=str(tmp_path/'already'),
                                 device='cpu',limit=1,start_t=2,seed=1,cpu_threads=2,abstract_goal_threshold=.01))
        result=list(read_jsonl(tmp_path/'already/results.jsonl'))[0]
        assert result['reverse_steps']==0 and result['abstract_goal_reached']
    args.resume=False;args.steps=1;args.mode='screenshot';args.out=str(tmp_path/'rgb')
    train(args)
    # Frozen-cache path must pass a tensor to detect(), never an image filename.
    import framediff.visual_train as visual_train
    monkeypatch.setattr(visual_train,'load_detector',lambda *a:(object(),{}))
    def fake_detect(net,image,threshold):
        assert image.ndim==3 and image.shape[0]==3
        return [{'box':[0,0,100,100],'label':1,'score':.9}]
    monkeypatch.setattr(visual_train,'detect',fake_detect)
    detector=tmp_path/'detector.pt';detector.write_bytes(b'test-checkpoint')
    predicted_data=tmp_path/'predicted'
    prepare(SimpleNamespace(source=str(source),out=str(predicted_data),bins=129,limit=0,max_nodes=32,resume=False,
                            detector_checkpoint=str(detector),device='cpu',threshold=.4))
    args.mode='abstract';args.data=str(predicted_data);args.out=str(tmp_path/'predicted-run');args.target_source='detector'
    train(args)
    # The previous replacement-policy corpus is untouched.
    assert len(list(read_jsonl(source/'policy-train.jsonl')))==1

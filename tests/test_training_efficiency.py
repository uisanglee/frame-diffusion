import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from framediff import tree_policy as policy
from framediff.tree_edits import FIELDS


def sample():
    from framediff.ir import node
    state=[[['',''] for _ in FIELDS] for _ in range(2)]
    state[1][0]=['123.5px','important'];state[1][1]=['auto','']
    return dict(current={'version':1,'nodes':[node('root',None,'page'),node('a','root','card')]},
                current_boxes={'root':[0,0,100,100],'a':[10,20,30,40]},viewport=[100,100],
                declaration_state=state,target_elements=[{'box':[10,20,30,40],'label':1}],
                replacement_edit=[1,'width','100px',''])


def test_declaration_bytes_and_loss_gradient_preserved():
    cfg=policy.TreeConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=8)
    row=sample();states=[row['declaration_state']]
    expected=torch.zeros(1,2,6,64,dtype=torch.long)
    for j,props in enumerate(states[0]):
        for k,(value,priority) in enumerate(props):
            values=[v+1 for v in (value+'!'+priority).encode()[:64]]
            expected[0,j,k,:len(values)]=torch.tensor(values)
    assert torch.equal(policy.declaration_tensor(states,2,FIELDS,'cpu'),expected)
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        net=policy.TreePolicy(cfg,False)
        batch=policy.tree_batch([row],'cpu',8)
        assert batch['relation'].tolist()==[[[1,2],[3,1]]]
        torch.testing.assert_close(batch['roi'],torch.tensor([[[0,0,1,1],[.1,.2,.4,.6]]]))
        image=torch.rand(1,8,32,32)
        first,_=net.loss(batch,image,image,states,[row['replacement_edit']]);first.backward()
        gradients={k:p.grad.clone() for k,p in net.named_parameters() if p.grad is not None}
        net.zero_grad();batch['declarations']=expected
        second,_=net.loss(batch,image,image,states,[row['replacement_edit']]);second.backward()
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        for k,p in net.named_parameters():
            if k in gradients:torch.testing.assert_close(p.grad,gradients[k],rtol=0,atol=0)
    finally:torch.set_num_threads(old_threads)


@pytest.mark.parametrize('error',[torch.OutOfMemoryError('OOM'),RuntimeError('CUDA launch failure'),
                                  ValueError('Unexpected input shape')])
def test_symbolic_device_and_model_errors_propagate(monkeypatch,error):
    row=sample();target=copy.deepcopy(row['declaration_state']);target[1][0]=['100px','important']
    monkeypatch.setattr(policy,'batch_inputs',lambda *a:(None,)*4)
    class Net:
        def predict(self,*a):raise error
    with pytest.raises(type(error),match=str(error)):
        policy.symbolic_policy_metrics(Net(),[row],[target],'cpu',False,contextlib.nullcontext)


@pytest.mark.parametrize('message',['No grammar-valid continuation','Replacement exceeded grammar token limit'])
def test_symbolic_expected_decoding_failure_is_invalid(monkeypatch,message):
    row=sample();target=copy.deepcopy(row['declaration_state']);target[1][0]=['100px','important']
    monkeypatch.setattr(policy,'batch_inputs',lambda *a:(None,)*4)
    class Net:
        def predict(self,*a):raise policy.InvalidPrediction(message)
    result=policy.symbolic_policy_metrics(Net(),[row],[target],'cpu',False,contextlib.nullcontext)
    assert result['invalid_action_rate']==1.0
    assert result['action_target_accuracy']==result['exact_repair_action_rate']==0.0


def test_validation_cache_is_bounded_cpu_inputs_only(monkeypatch):
    rows=[sample(),sample()];net=SimpleNamespace(cfg=policy.TreeConfig(size=32,max_nodes=8))
    calls=[];original=policy.batch_inputs
    def counted(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(policy,'batch_inputs',counted)
    cache=policy.ValidationInputs(max_bytes=100000)
    a=cache.get(net,rows[:1],'cpu');b=cache.get(net,rows[:1],'cpu')
    assert len(calls)==1 and cache.bytes<=cache.max_bytes
    for left,right in zip(a[1:3],b[1:3]):assert torch.equal(left,right)
    assert all(not t.requires_grad and t.device.type=='cpu' for t in (a[1],a[2],a[0]['declarations']))
    cache.get(net,rows[1:],'cpu')
    assert len(cache.entries)==1  # Second batch exceeds bounded admission limit.
    assert cache.bytes<=cache.max_bytes


def test_detector_checkpoint_failed_save_keeps_last(tmp_path,monkeypatch):
    from framediff import visual_train
    path=tmp_path/'last.pt';torch.save({'step':3},path);before=path.read_bytes()
    def failed_save(value,temporary):
        temporary.write_bytes(b'partial');raise OSError('disk full')
    monkeypatch.setattr(visual_train.torch,'save',failed_save)
    with pytest.raises(OSError,match='disk full'):visual_train.save_checkpoint({'step':4},path)
    assert path.read_bytes()==before and not path.with_name('last.pt.tmp').exists()


def test_detector_complete_prediction_cache_never_constructs_gpu_model(tmp_path,monkeypatch):
    from framediff import visual_train
    from framediff.ir import write_json,write_jsonl
    from framediff.visual import CONTRACT
    from framediff.web_experiment import digest
    checkpoint=tmp_path/'detector.pt'
    torch.save(dict(kind='visual-detector-v1',config={'contract':CONTRACT},
                    training_groups=['train'],training_hashes=['train']),checkpoint)
    png=tmp_path/'target.png';Image.new('RGB',(10,10)).save(png)
    manifest=tmp_path/'data.jsonl'
    write_jsonl(manifest,[dict(id='val/seed',split='val',group='val',source_sha='val',
                              target_image=str(png),viewport=[10,10])])
    out=tmp_path/'predicted';out.mkdir();key=digest(png)
    Image.new('RGB',(32,32)).save(out/f'{key}.png');write_json(out/f'{key}.json',[])
    monkeypatch.setattr(visual_train,'detector',lambda *a,**kw:pytest.fail('Unneeded model allocation'))
    args=SimpleNamespace(device='cpu',cpu_threads=1,data=str(manifest),checkpoint=str(checkpoint),
                         out=str(out),resume=True,threshold=.4,size=32,reuse_cache=None)
    assert len(visual_train.cache_targets(args))==1


def test_common_evaluation_alias_keeps_existing_results(tmp_path):
    from framediff.evaluation_alias import alias_result
    own=tmp_path/'own';own.mkdir();(own/'summary.json').write_text('new')
    common=tmp_path/'common';common.mkdir();(common/'summary.json').write_text('old')
    alias_result(own,common)
    assert common.is_symlink() and (common/'summary.json').read_text()=='new'
    assert next(tmp_path.glob('.evaluation-history/common-*/summary.json')).read_text()=='old'
    alias_result(own,common)
    assert len(list(tmp_path.glob('.evaluation-history/common-*')))==1

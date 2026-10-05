from dataclasses import asdict
from PIL import Image
import pytest
import torch
from framediff.visual import VisualConfig, VisualPolicy, ACTIONS, current_features, visual_batch, action_index, load_policy
from framediff.visual import NUMERIC_FIELDS
from framediff.visual_numeric import numeric_rows
from framediff.ir import node


def test_filter_removes_categorical_descendants_not_other_trajectories():
    rows=[{'id':'page/'+name,'teacher_edits':edit} for name,edit in [
        ('clean',[]),('t0-s0',[[1,'width',.05]]),('t0-s1',[[1,'align-items','center']]),
        ('t0-s2',[[1,'height',.05]]),('t1-s0',[[1,'margin-left',.05]]),
        ('t2-s1',[[1,'width',.05]])]]
    result=numeric_rows(rows)
    assert [r['id'] for r in result]==['page/clean','page/t0-s0','page/t1-s0']
    assert all('policy_subset' not in r for r in rows)


def test_filter_uses_actual_corruption_not_best_reverse_label():
    rows=[
        {'id':'page/clean','teacher_edits':[],'corruption_edit':None},
        {'id':'page/t0-s0','teacher_edits':[[1,'width',-.05]],
         'corruption_edit':[1,'align-items','center']},
        {'id':'page/t0-s1','teacher_edits':[[1,'height',-.05]],
         'corruption_edit':[1,'height',.05]},
    ]
    assert [r['id'] for r in numeric_rows(rows)]==['page/clean']


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


def test_balanced_sampler_equal_operations_and_uniform_properties():
    import random
    from collections import Counter
    from framediff.visual_train import policy_training_pools,sample_policy_rows
    rows=[{'id':'stop','teacher_edits':[]}]+[
        {'id':field,'teacher_edits':[[1,field,.05]]} for field in NUMERIC_FIELDS]
    pools=policy_training_pools(rows);rng=random.Random(7);counts=Counter()
    for _ in range(2000):
        batch=sample_policy_rows(pools,2,rng)
        assert sum(not r['teacher_edits'] for r in batch)==1
        counts.update(r['teacher_edits'][0][1] for r in batch if r['teacher_edits'])
    assert max(counts.values())-min(counts.values())<100


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


def test_numeric_prepare_train_resume_without_browser(tmp_path):
    from PIL import Image
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.cli import main
    from framediff.visual import CONTRACT,ACTION_CONTRACT
    source=tmp_path/'rendered';source.mkdir()
    image=tmp_path/'existing.png';Image.new('RGB',(32,32),'white').save(image)
    html=tmp_path/'existing.html';html.write_text('<html><body>existing</body></html>')
    original=(image.read_bytes(),html.read_bytes())
    tree={'version':1,'nodes':[node('page',None,'page'),node('card','page','card',width=80)]}
    boxes={'page':[0,0,320,240],'card':[16,20,80,30]}
    for split in ('train','val','test'):
        base={'group':split,'source_sha':split,'split':split,'contract':CONTRACT,
              'action_contract':ACTION_CONTRACT,'viewport':[320,240],
              'teacher_strategy':'best-improving-reverse-v1',
              'current':tree,'current_boxes':boxes,'current_html':str(html),
              **{k:str(image) for k in ('target_image','target_abstract','current_image','current_abstract')}}
        rows=[{**base,'id':split+'/clean','teacher_edits':[]}]+[
              {**base,'id':f'{split}/t0-s{i}','teacher_edits':[[1,field,.05]]}
              for i,field in enumerate(NUMERIC_FIELDS)]
        write_jsonl(source/f'policy-{split}.jsonl',rows)
        write_jsonl(source/f'pages-{split}.jsonl',[{'id':split,'split':split,'html':str(html)}])
    subset=tmp_path/'subset'
    command=['visual-subset-numeric','--rendered',str(source),'--out',str(subset)]
    main(command);main(command)
    assert len(list(read_jsonl(subset/'policy-train.jsonl')))==7
    out=tmp_path/'policy'
    command=['visual-train-policy','--train',str(subset/'policy-train.jsonl'),
             '--val',str(subset/'policy-val.jsonl'),'--out',str(out),'--mode','abstract',
             '--numeric-only','--policy-head','hierarchical','--device','cpu','--no-pretrained',
             '--balanced-policy',
             '--size','32','--hidden','16','--layers','1','--heads','2','--token-grid','2',
             '--eval-every','1','--val-samples','2','--batch-size','2','--accumulation','1']
    main(command+['--steps','1'])
    main(command+['--steps','2','--resume',str(out/'last.pt')])
    ck=torch.load(out/'last.pt',weights_only=True)
    assert ck['step']==2
    metrics=list(read_jsonl(out/'train.jsonl'))[-1]
    assert 'false_stop_rate' in metrics
    assert set(metrics['validation_components'])=={'operation','node','property','delta'}
    assert 0<ck['stop_threshold']<1
    ar=tmp_path/'autoregressive-policy'
    ar_command=list(command);ar_command[ar_command.index('--out')+1]=str(ar)
    ar_command[ar_command.index('--policy-head')+1]='autoregressive'
    main(ar_command+['--steps','1'])
    ar_ck=torch.load(ar/'last.pt',weights_only=True)
    assert ar_ck['kind']=='visual-policy-v6-semantic' and ar_ck['config']['policy_head']=='autoregressive'
    assert ar_ck['config']['observation_contract']=='semantic-mask-pair-diff-v1'
    assert original==(image.read_bytes(),html.read_bytes())


@pytest.mark.browser
def test_legacy_policy_relabel_reuses_assets_and_browser_reflow(tmp_path):
    from types import SimpleNamespace
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.visual_numeric import relabel_best_reverse
    from framediff.visual import CONTRACT,ACTION_CONTRACT
    source=tmp_path/'legacy';source.mkdir();assets=tmp_path/'assets';assets.mkdir()
    target_html=assets/'target.html';target_html.write_text(
        '<html><body data-fd-id="fd-0" style="margin:0"><div data-fd-id="fd-1" '
        'style="width:100px;height:20px"></div></body></html>')
    current_html=assets/'current.html';current_html.write_text(
        '<html><body data-fd-id="fd-0" style="margin:0"><div data-fd-id="fd-1" '
        'style="width:120px;height:20px"></div></body></html>')
    image=assets/'unchanged.png';Image.new('RGB',(400,200),'white').save(image);original=image.read_bytes()
    tree={'version':1,'nodes':[node('fd-0',None,'body'),node('fd-1','fd-0','div',width=100)]}
    tree['nodes'][1]['visual_class']=4
    common={'group':'page','source_sha':'page','split':'train','contract':CONTRACT,
            'action_contract':ACTION_CONTRACT,'viewport':[400,200],'target_image':str(image),
            'target_abstract':str(image),'current_image':str(image),'current_abstract':str(image),'current':tree}
    clean={**common,'id':'page/clean','current_boxes':{'fd-0':[0,0,400,200],'fd-1':[0,0,100,20]},
           'current_html':str(target_html),'teacher_edits':[]}
    edit={**common,'id':'page/t0-s0','current_boxes':{'fd-0':[0,0,400,200],'fd-1':[0,0,120,20]},
          'current_html':str(current_html),'teacher_edits':[[1,'width',-.05]],
          'corruption_edit':[1,'width',.05]}
    for split in ('train','val','test'):
        split_rows=[{**clean,'split':split,'group':f'page-{split}','source_sha':f'page-{split}',
                     'id':f'page-{split}/clean'},
                    {**edit,'split':split,'group':f'page-{split}','source_sha':f'page-{split}',
                     'id':f'page-{split}/t0-s0'}]
        write_jsonl(source/f'policy-{split}.jsonl',split_rows)
        write_jsonl(source/f'pages-{split}.jsonl',[{'id':f'page-{split}','split':split}])
        write_jsonl(source/f'detector-{split}.jsonl',[])
    out=tmp_path/'relabeled';args=SimpleNamespace(rendered=str(source),out=str(out),resume=False)
    result=relabel_best_reverse(args)
    assert result['kept_rows']==6 and result['rejected_rows']==0
    rows=list(read_jsonl(out/'policy-train.jsonl'))
    assert rows[1]['teacher_strategy']=='best-improving-reverse-v1'
    assert rows[1]['teacher_edits']==[[1,'width',-.05]]
    assert rows[1]['teacher_distance_after']<rows[1]['teacher_distance_before']
    assert image.read_bytes()==original

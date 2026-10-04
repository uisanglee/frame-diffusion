from dataclasses import asdict
import torch
from framediff.visual import VisualConfig, VisualPolicy, ACTIONS, current_features, visual_batch, action_index, load_policy
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
              'current':tree,'current_boxes':boxes,'current_html':str(html),
              **{k:str(image) for k in ('target_image','target_abstract','current_image','current_abstract')}}
        rows=[{**base,'id':split+'/clean','teacher_edits':[]},
              {**base,'id':split+'/t0-s0','teacher_edits':[[1,'width',.05]]}]
        write_jsonl(source/f'policy-{split}.jsonl',rows)
        write_jsonl(source/f'pages-{split}.jsonl',[{'id':split,'split':split,'html':str(html)}])
    subset=tmp_path/'subset'
    command=['visual-subset-numeric','--rendered',str(source),'--out',str(subset)]
    main(command);main(command)
    assert len(list(read_jsonl(subset/'policy-train.jsonl')))==2
    out=tmp_path/'policy'
    command=['visual-train-policy','--train',str(subset/'policy-train.jsonl'),
             '--val',str(subset/'policy-val.jsonl'),'--out',str(out),'--mode','abstract',
             '--numeric-only','--policy-head','hierarchical','--device','cpu','--no-pretrained',
             '--size','32','--hidden','16','--layers','1','--heads','2','--token-grid','2',
             '--eval-every','1','--val-samples','2','--batch-size','2','--accumulation','1']
    main(command+['--steps','1'])
    main(command+['--steps','2','--resume',str(out/'last.pt')])
    ck=torch.load(out/'last.pt',weights_only=True)
    assert ck['step']==2
    metrics=list(read_jsonl(out/'train.jsonl'))[-1]
    assert 'false_stop_rate' in metrics
    assert original==(image.read_bytes(),html.read_bytes())

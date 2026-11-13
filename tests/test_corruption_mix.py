import pytest
from framediff.corruption_mix import geometry_errors,summary
from framediff.tree_edits import EditTokenizer,FIELDS


def test_geometry_ignores_appearance_and_measures_size_and_position():
    a={'root':[0,0,1000,500],'one':[10,10,100,50]}
    assert geometry_errors(a,a,[1000,500],'root')['measured_level']=='clean'
    b={**a,'one':[30,10,200,50]}
    m=geometry_errors(a,b,[1000,500],'root')
    assert m['position_mae_px']==10 and m['size_mae_px']==50
    assert m['measured_level']=='medium' and m['changed_elements']==1
    assert geometry_errors(a,{'root':a['root']},[1000,500],'root')['measured_level']=='unmeasurable'


def test_stop_grammar_is_opt_in_and_preserves_vocabulary():
    old=EditTokenizer(8);new=EditTokenizer(8,allow_stop=True)
    assert old.tokens==new.tokens
    eos=new.ids['EOS'];state=[[['',''] for _ in FIELDS] for _ in range(2)]
    assert eos not in old.allowed([],2,state)
    assert eos in new.allowed([],2,state)
    assert new.encode(None,2)==[eos] and new.decode([eos],2) is None
    assert new.allowed([eos],2)==[]
    with pytest.raises(ValueError):old.decode([eos],2)


@pytest.mark.browser
def test_mixed_sampler_clean_and_corrupted_states(corpus):
    from framediff.tree_online import OnlineSampler,target_pool
    pool=target_pool([corpus['train']])
    sampler=OnlineSampler(pool,'screenshot',corruption_policy='mixed',attempts=12)
    try:
        rows=[sampler.generate(i) for i in range(5)]
        assert rows[0]['replacement_edit'] is None
        assert rows[0]['symbolic_distance']==0
        assert rows[0]['corruption_geometry']['measured_level']=='clean'
        assert 'clean_target_only' not in rows[0]
        for row in rows[1:]:
            assert row['replacement_edit'] is not None
            assert row['corruption_geometry']['changed_elements']>0
        assert rows[4]['corruption_geometry']['changed_owners']>=2
        assert rows[4]['online']['reverse_prefix']==0
        assert summary(rows)['requested']==dict(clean=1,mild=1,medium=1,strong=1,multi=1)
        assert sampler.generate(2)['declaration_state']==rows[2]['declaration_state']
    finally:sampler.close()

from tests.test_tree_online import corpus

@pytest.mark.browser
def test_mixed_training_learns_stop_and_logs_geometry(corpus,tmp_path):
    from framediff.tree_online import OnlineSampler,target_pool
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.cli import main
    import torch
    sampler=OnlineSampler(target_pool([{**corpus['val'],'split':'train'}]),'screenshot',corruption_policy='mixed')
    try:
        validation=[]
        for index in range(5):
            row=sampler.generate(index)
            png=tmp_path/f'val-{index}.png';png.write_bytes(row['current_image'])
            row.update(split='val',current_image=str(png))
            validation.append(row)
    finally:sampler.close()
    write_jsonl(tmp_path/'policy-val.jsonl',validation)
    main(['visual-tree-train','--train',str(tmp_path/'policy-train.jsonl'),
          '--val',str(tmp_path/'policy-val.jsonl'),'--out',str(tmp_path/'mixed-run'),
          '--mode','abstract','--online-corruption','--online-workers','1','--online-prefetch','1',
          '--device','cpu','--no-pretrained','--size','32','--hidden','16','--layers','1',
          '--heads','2','--token-grid','2','--max-nodes','16','--batch-size','1',
          '--accumulation','1','--eval-every','2','--val-samples','0','--cpu-threads','2',
          '--steps','5','--policy-metric-samples','0'])
    ck=torch.load(tmp_path/'mixed-run/last.pt',weights_only=True)
    assert ck['config']['learn_stop'] is True
    logs=list(read_jsonl(tmp_path/'mixed-run/corruption-train.jsonl'))
    assert [r['requested_profile'] for r in logs]==['clean','mild','medium','strong','multi']
    assert logs[0]['changed_elements']==0
    evaluations=list(read_jsonl(tmp_path/'mixed-run/validation.jsonl'))
    assert [r['step'] for r in evaluations]==[2,4,5]
    assert all(r['corruption_geometry']==summary(validation) for r in evaluations)
    # Validation at steps 2 and 4 must not consume the unlogged train interval.
    training_logs=[r for r in read_jsonl(tmp_path/'mixed-run/train.jsonl') if 'loss' in r]
    assert len(training_logs)==1
    assert training_logs[0]['corruption_geometry']['n']==5


def test_clean_stop_validation_metric(monkeypatch):
    import contextlib
    from types import SimpleNamespace
    from framediff import tree_policy
    state=[[['',''] for _ in FIELDS] for _ in range(2)]
    monkeypatch.setattr(tree_policy,'batch_inputs',lambda *args:())
    class Net:
        cfg=SimpleNamespace(learn_stop=True)
        def predict(self):return None
    metrics=tree_policy.symbolic_policy_metrics(Net(),[{'declaration_state':state}],
        [state],'cpu',False,contextlib.nullcontext)
    assert metrics['clean_metric_n']==1 and metrics['clean_stop_accuracy']==1

import copy
from types import SimpleNamespace

import pytest

from framediff.composition import COMBINATIONS, CATEGORIES, category_fields, diagnostic, build, evaluate, distance
from framediff.ir import read_json, read_jsonl
from framediff.tree_edits import FIELDS, apply_state


def test_categories_cover_all_nonempty_subsets():
    assert len(COMBINATIONS)==15
    assert len(set(COMBINATIONS))==15
    assert [sum(len(c.split('+'))==n for c in COMBINATIONS) for n in range(1,5)]==[4,6,4,1]
    assert {category_fields(i)['X'] for i in range(24)}=={'margin-left','margin-right'}
    assert {category_fields(i)['Y'] for i in range(24)}=={'margin-top','margin-bottom'}


def test_multiple_correct_actions_order_and_collateral():
    target = [[['',''] for _ in FIELDS] for _ in range(3)]
    initial = copy.deepcopy(target)
    edits=[]
    for category,node in [('W',1),('X',2)]:
        field=CATEGORIES[category]; j=FIELDS.index(field)
        target[node][j]=['100px',''];initial[node][j]=['50px','']
        edits.append(dict(category=category,edit=[node,field,'50px','']))
    box={'a':[0,0,10,10]}
    row=dict(initial_state=initial,target_state=target,target_boxes=box,corruption=edits,combination='W+X')
    for order in ([0,1],[1,0]):
        states=[initial]
        for n in order:
            i,f,_,_=edits[n]['edit']
            states.append(apply_state(states[-1],[i,f,'100px','']))
        result=diagnostic(row,states,[box]*3,1)
        assert result['improving_action_rate']==1
        assert result['symbolic_full_recovery']==1
        assert result['collateral_declaration_rate']==0
    broken=apply_state(target,[1,'height','20px',''])
    result=diagnostic(row,[initial,target,broken],[box]*3,1)
    assert result['category_recovery']==1
    assert result['symbolic_full_recovery']==0
    assert result['collateral_declaration_rate']>0


@pytest.mark.browser
def test_build_and_known_reverse_restore_all_combinations(tmp_path):
    data=tmp_path/'data'
    args=SimpleNamespace(out=str(data),samples_per_cell=6,seed=73129,resume=False)
    build(args)
    rows=list(read_jsonl(data/'manifest.jsonl'))
    assert len(rows)==360
    assert {r['owner_kind'] for r in rows}=={'inline','rule'}
    for row in rows:
        assert distance(row['initial_state'],row['target_state'])==row['category_count']
        assert row['affected_nodes']==(1 if row['placement']=='same' else row['category_count'])
    args.resume=True
    build(args)
    assert list(read_jsonl(data/'manifest.jsonl'))==rows
    out=tmp_path/'eval'
    evaluate(SimpleNamespace(data=str(data/'manifest.jsonl'),out=str(out),checkpoint=None,detector_checkpoint=None,
        sanity=True,oracle=False,device='cpu',cpu_threads=1,steps=4,threshold=.4,
        geometry_tolerance=.01,limit=0,resume=False))
    summary=read_json(out/'summary.json')['overall']
    assert summary['n']==360
    assert summary['failed_rate']==0
    assert summary['symbolic_full_recovery']==1
    assert summary['geometric_full_recovery']==1
    assert summary['final_box_iou']==1
    # Exercise actual neural decoding and modality preprocessing, not just the
    # known-reverse sanity path. Random weights establish no performance claim.
    from dataclasses import asdict
    import torch
    from framediff.css_owners import CONTRACT
    from framediff.tree_policy import TreeConfig, TreePolicy, KIND
    torch.manual_seed(42)
    for mode in ('abstract','screenshot'):
        cfg=TreeConfig(mode=mode,size=64,hidden=32,layers=1,heads=4,token_grid=2,
                       stylesheets=True,existing_values_only=True,action_contract=CONTRACT)
        net=TreePolicy(cfg,pretrained=False)
        checkpoint=tmp_path/f'{mode}.pt'
        torch.save(dict(kind=KIND,config=asdict(cfg),model=net.state_dict()),checkpoint)
        run=tmp_path/mode
        evaluate(SimpleNamespace(data=str(data/'manifest.jsonl'),out=str(run),checkpoint=str(checkpoint),
            detector_checkpoint=None,sanity=False,oracle=mode=='abstract',device='cpu',cpu_threads=1,
            steps=1,threshold=.4,geometry_tolerance=1.,limit=1,resume=False))
        result=read_json(run/'summary.json')['overall']
        assert result['failed_rate']==0
        assert result['actions']==1

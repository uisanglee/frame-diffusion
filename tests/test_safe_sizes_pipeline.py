import pytest
from framediff.html_bridge import HtmlBrowser

def test_retired_protocols_cannot_enter_current_training():
    from framediff.tree_policy import TreeConfig
    from framediff.tree_edits import EditTokenizer,FIELDS
    from framediff.tree_online import target_pool
    assert set(FIELDS)=={'width','height','margin-left','margin-right','margin-top','margin-bottom'}
    assert TreeConfig().fields==FIELDS
    with pytest.raises(ValueError,match='contract mismatch'):
        TreeConfig(stylesheets=True,action_contract='parent-relative-xywh-v1')
    with pytest.raises(ValueError,match='Only width/height/margin'):
        EditTokenizer(8,('x','y','width','height'))
    for key in ('recover_normalization','box_layout'):
        with pytest.raises(ValueError,match='Retired'):
            target_pool([{'lazy_contract':'old','lazy_settings':{key:True}}])

def test_preparation_online_training_resume_and_rollout(tmp_path,monkeypatch):
    import torch
    from framediff.cli import main
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.visual import VISUAL_GEOMETRY_CONTRACT
    sources=[]
    for split in ('train','val','test'):
        html=tmp_path/f'{split}.html'
        html.write_text(f'<html><body style="margin:0"><div style="width:120px;height:60px;background:red">{split}</div></body></html>')
        sources.append(dict(id=split,group=split,split=split,html=str(html),viewport=[300,200]))
    manifest=tmp_path/'source.jsonl';write_jsonl(manifest,sources)
    rendered=tmp_path/'rendered';labels=tmp_path/'labels';frozen=tmp_path/'frozen'
    main(['visual-build-data','--manifest',str(manifest),'--out',str(rendered),'--clean-only'])
    detection=list(read_jsonl(rendered/'detector-train.jsonl'))[0]
    assert detection['visual_geometry_contract']==VISUAL_GEOMETRY_CONTRACT
    assert all('owner_id' in e for e in detection['elements'] if e['label']==1)
    prepare=['visual-tree-normalize','--rendered',str(rendered),'--out',str(labels),'--resume']
    main(prepare)
    saved=(labels/'policy-train.jsonl').read_bytes()
    main(prepare)
    assert saved==(labels/'policy-train.jsonl').read_bytes()
    main(['visual-tree-freeze','--rendered',str(labels),'--out',str(frozen),'--max-noise','1','--resume'])
    common=['visual-tree-train','--train',str(frozen/'policy-train.jsonl'),'--val',str(frozen/'policy-val.jsonl'),
        '--out',str(tmp_path/'run'),'--stylesheets','--mode','screenshot','--online-corruption',
        '--online-workers','1','--online-prefetch','1','--device','cpu','--no-pretrained',
        '--size','32','--hidden','16','--layers','1','--heads','2','--token-grid','2','--max-nodes','32',
        '--batch-size','1','--accumulation','1','--eval-every','1','--val-samples','1','--cpu-threads','2']
    main([*common,'--steps','1'])
    checkpoint=tmp_path/'run/last.pt'
    main([*common,'--steps','2','--resume',str(checkpoint)])
    ck=torch.load(checkpoint,weights_only=True)
    from framediff.css_owners import CONTRACT as OWNER_CONTRACT
    assert ck['config']['action_contract']==OWNER_CONTRACT
    assert ck['config']['visual_geometry_contract']==VISUAL_GEOMETRY_CONTRACT
    assert ck['config']['safe_sizes'] and ck['online_cursor']==2
    from framediff.tree_policy import TreePolicy,TreeConfig
    from framediff.visual_experiment import rollout
    original=list(read_jsonl(rendered/'policy-test.jsonl'))[0]
    net=TreePolicy(TreeConfig(**ck['config']),False);net.load_state_dict(ck['model']);net.eval()
    from pathlib import Path
    with HtmlBrowser() as browser:
        converted,stats,_=rollout(browser,Path(original['current_html']).read_text(),original['current'],
            original['viewport'],original['target_image'],net,None,.4,0,0)
    assert not stats['failed'],stats
    assert stats['safe_size_conversion']['accepted']
    assert 'data-tuide-safe-sizes' in converted
    from framediff.ir import write_json
    stage1=list(common)
    stage1[stage1.index('--mode')+1]='abstract'
    stage1[stage1.index('--out')+1]=str(tmp_path/'abstract-stage1')
    main([*stage1,'--steps','1'])
    stage2=list(stage1);stage2[stage2.index('--out')+1]=str(tmp_path/'abstract-stage2')
    for split,flag in [('train','--train'),('val','--val')]:
        rows=list(read_jsonl(frozen/f'policy-{split}.jsonl'))
        for i,row in enumerate(rows):
            path=tmp_path/f'fake-detector-{split}-{i}.json'
            # Conditioning plumbing fixture, not a detector quality test.
            write_json(path,row['target_elements'])
            row['predicted_target_elements']=str(path)
            row['parser_provenance']={'parser_training_groups':[],'fixture':True,
                                     'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT}
        path=tmp_path/f'cached-{split}.jsonl';write_jsonl(path,rows)
        stage2[stage2.index(flag)+1]=str(path)
    main([*stage2,'--steps','1','--predicted-targets','--init-checkpoint',str(tmp_path/'abstract-stage1/best.pt')])
    trained=torch.load(tmp_path/'abstract-stage2/last.pt',weights_only=True)
    assert trained['config']['safe_sizes'] and trained['config']['action_contract']==OWNER_CONTRACT
    test=list(read_jsonl(frozen/'policy-test.jsonl'))[0]
    # Executor/evaluator test: inject the known single edit, not a quality claim
    # about the two-step smoke-trained checkpoint.
    monkeypatch.setattr(TreePolicy,'predict',lambda self,*a,**k:test['replacement_edit'])
    main(['visual-test-rollout','--data',str(frozen/'policy-test.jsonl'),'--pages',str(frozen/'pages-test.jsonl'),
          '--raw-checkpoint',str(checkpoint),'--out',str(tmp_path/'eval'),'--device','cpu','--steps','1'])
    result=list(read_jsonl(tmp_path/'eval/results.jsonl'))[0]
    assert result['failed'] is False
    assert result['final_box_iou']==pytest.approx(1.)


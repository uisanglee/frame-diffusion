from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from framediff import css_owners
from framediff.tree_edits import repair_path, FIELDS

HTML='''<html><head><style>
body {margin:0} .card {width:240px;height:40px}
@media (min-width: 400px) {.container .card {width:300px}}
.card {width:200px}
@media (max-width: 100px) {.card {width:99px!important}}
</style></head><body><section class="container"><div class="card">A</div><div class="card">B</div></section></body></html>'''


@pytest.mark.browser
def test_rule_paths_cascade_shared_reflow_and_export(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    with HtmlBrowser() as browser:
        dom=browser.snapshot(HTML,[500,300],max_nodes=12);tree,boxes,_=dom_tree(dom)
        target=css_owners.read(browser,tree)
        inert=css_owners.read(browser,tree,dom['html'],fixed=True)
        assert target['owners']==inert['owners'] and target['state']==inert['state']
        owners=target['owners']
        winner=next(i for i,o in enumerate(owners) if o.get('selector')=='.container .card')
        losers=[i for i,o in enumerate(owners) if o.get('selector')=='.card']
        edit=[winner,'width','350px','']
        assert css_owners.computed(browser,owners,edit)==['300px','300px']
        css_owners.execute(browser,owners,[losers[0],'width','390px',''])
        assert css_owners.computed(browser,owners,edit)==['300px','300px']
        css_owners.execute(browser,owners,edit)
        assert css_owners.computed(browser,owners,edit)==['350px','350px']
        exported=browser.page.content();browser.load(exported,[500,300])
        assert css_owners.computed(browser,owners,edit)==['350px','350px']
        current=css_owners.read(browser,tree)
        assert current['owners']==owners
        assert css_owners.read(browser,tree,exported,fixed=True)['fixed']==inert['fixed']
        for action in repair_path(current['state'],target['state']):css_owners.execute(browser,owners,action)
        assert css_owners.read(browser,tree)['state']==target['state']
        assert css_owners.computed(browser,owners,edit)==['300px','300px']
        # Deletion/reinsertion preserves rule identity and priority.
        css_owners.execute(browser,owners,[winner,'width','',''])
        assert css_owners.computed(browser,owners,edit)==['200px','200px']
        css_owners.execute(browser,owners,[winner,'width','300px','important'])
        css_owners.execute(browser,owners,[2,'width','320px',''])
        assert css_owners.computed(browser,owners,edit)==['300px','300px']


@pytest.mark.browser
def test_owner_online_samples_and_policy_rollout(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate,elements,load_policy
    from framediff.tree_online import OnlineSampler
    from framediff.tree_policy import TreeConfig,TreePolicy,batch_inputs,KIND
    from framediff.visual_experiment import rollout
    torch.set_num_threads(2)
    image=tmp_path/'target.png';html=tmp_path/'target.html'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(HTML,[500,300],image,12);tree,boxes,_=dom_tree(dom);annotate(browser,tree)
        html.write_text(dom['html']);parsed=css_owners.read(browser,tree)
    target=dict(id='train',group='train',split='train',source_sha='train',current=tree,viewport=[500,300],
        target_html=str(html),target_image=str(image),target_elements=elements(tree,boxes,[500,300]),css_owners=parsed['owners'])
    sampler=OnlineSampler([target],'abstract',seed=7,max_noise=3)
    try:
        rows=[sampler.generate(i) for i in range(16)]
        assert any(r['online']['teacher_owner']=='rule' for r in rows)
        assert any(r['replacement_edit'][2] for r in rows)
        for row in rows:
            assert row['teacher_strategy']==css_owners.CONTRACT
            assert row['replacement_edit'] in repair_path(row['declaration_state'],parsed['state'])
            assert row['css_owners']==parsed['owners']
            # No declaration is added/deleted and no priority is escalated.
            for before,after in zip(parsed['state'],row['declaration_state']):
                assert [bool(v) for v,p in before]==[bool(v) for v,p in after]
                assert [p for v,p in before]==[p for v,p in after]
            assert row['replacement_edit'][2]  # reverse teacher is SET
            assert all(e[2] for e in row['online']['corruptions'])
        row=next(r for r in rows if r['online']['teacher_owner']=='rule')
    finally:sampler.close()
    cfg=TreeConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=32,
        stylesheets=True,action_contract=css_owners.CONTRACT,existing_values_only=True)
    policy=TreePolicy(cfg,False)
    inputs=batch_inputs(policy,[row],'cpu')
    loss,_=policy.loss(*inputs,[row['replacement_edit']]);loss.backward()
    assert torch.isfinite(loss) and policy.owner_type.weight.grad.abs().sum()>0
    saved=tmp_path/'model.pt';torch.save(dict(kind=KIND,config=asdict(cfg),model=policy.state_dict()),saved)
    policy,_=load_policy(saved,'cpu');policy.eval()
    # Force a rule action to check rollout execution and exported HTML, without
    # claiming the untrained network predicts a useful action.
    policy.predict=lambda *args,**kw:row['replacement_edit']
    with HtmlBrowser() as browser:
        result,stats,_=rollout(browser,row['current_html_text'],tree,[500,300],str(image),policy,
            oracle_elements=target['target_elements'],steps=1)
        assert stats['history'][0]['css_owner']['kind']=='rule'
        browser.load(result,[500,300]);state=css_owners.read(browser,tree)['state']
        index,field,value,priority=row['replacement_edit']
        assert state[index][FIELDS.index(field)]==[value,priority]


@pytest.mark.browser
def test_prepare_freeze_cache_reuse_and_online_resume(tmp_path,monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff.ir import read_json,read_jsonl,write_json,write_jsonl
    from framediff.cli import main
    from framediff.web_experiment import digest
    import framediff.visual_train as vt
    source=tmp_path/'source';source.mkdir()
    with HtmlBrowser() as browser:
        for split in ('train','val','test'):
            work=source/split;work.mkdir()
            dom=browser.snapshot(HTML.replace('>A<',f'>{split}<'),[500,300],work/'target.png',12)
            tree,boxes,_=dom_tree(dom);annotate(browser,tree)
            clean=work/'clean.html';clean.write_text(dom['html'])
            base=dict(group=split,split=split,source_sha=digest(clean),viewport=[500,300],
                      target_image=str(work/'target.png'),current_image=str(work/'target.png'),
                      current=tree,current_boxes=boxes)
            clean_row=dict(base,id=split+'/clean',current_html=str(clean))
            parsed=css_owners.read(browser,tree)
            index=next(i for i,o in enumerate(parsed['owners']) if o.get('selector')=='.container .card')
            css_owners.execute(browser,parsed['owners'],[index,'width','350px',''])
            broken=work/'broken.html';broken.write_text(browser.page.content())
            browser.page.screenshot(path=str(work/'current.png'))
            boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]]);boxes[tree['nodes'][0]['id']]=[0,0,500,300]
            row=dict(base,id=split+'/t0-s0',current_html=str(broken),current_image=str(work/'current.png'),current_boxes=boxes)
            write_jsonl(source/f'policy-{split}.jsonl',[clean_row,row])
            write_jsonl(source/f'pages-{split}.jsonl',[dict(id=split,html=str(clean),split=split,group=split,source_sha=digest(clean))])
    prepared=tmp_path/'prepared';frozen=tmp_path/'frozen'
    main(['visual-tree-prepare','--rendered',str(source),'--out',str(prepared),'--stylesheets'])
    rows=list(read_jsonl(prepared/'policy-train.jsonl'));assert rows[0]['replacement_edit'][2]=='300px'
    args=['visual-tree-freeze','--rendered',str(prepared),'--out',str(frozen),'--samples-per-page','2']
    main(args);before=(frozen/'policy-val.jsonl').read_bytes();main([*args,'--resume'])
    assert before==(frozen/'policy-val.jsonl').read_bytes()
    assert read_json(frozen/'freeze-report.json')['val']['rows']==2
    # Reuse old detector outputs even when the policy manifest has new rows.
    old=tmp_path/'old-cache';old.mkdir();checkpoint=tmp_path/'detector.pt';checkpoint.write_bytes(b'test')
    val=list(read_jsonl(frozen/'policy-val.jsonl'))
    key=digest(val[0]['target_image']);(old/f'{key}.png').write_bytes(Path(val[0]['target_image']).read_bytes())
    write_json(old/f'{key}.json',val[0]['target_elements'])
    write_json(old/'config.json',dict(kind='visual-cache-v1',checkpoint=digest(checkpoint),threshold=.4,size=384))
    monkeypatch.setattr(vt,'load_detector',lambda *a:(None,dict(training_groups=[],training_hashes=[])))
    def no_detect(*a):raise AssertionError('Cache reuse must not run detector inference')
    monkeypatch.setattr(vt,'detect',no_detect)
    main(['visual-cache-targets','--data',str(frozen/'policy-val.jsonl'),'--checkpoint',str(checkpoint),
          '--out',str(tmp_path/'new-cache'),'--reuse-cache',str(old),'--device','cpu'])
    reused=list(read_jsonl(tmp_path/'new-cache/data.jsonl'))
    assert all(r['predicted_target_elements']==str(old/f'{key}.json') for r in reused)
    # Train both observation paths, validate fixed data, and resume consumed cursor.
    for mode in ('abstract','screenshot'):
        run=tmp_path/mode
        common=['visual-tree-train','--train',str(frozen/'policy-train.jsonl'),'--val',str(frozen/'policy-val.jsonl'),
            '--out',str(run),'--mode',mode,'--stylesheets','--online-corruption','--online-workers','1',
            '--online-prefetch','1','--size','32','--hidden','16','--layers','1','--heads','2','--max-nodes','32',
            '--token-grid','2','--device','cpu','--cpu-threads','2','--batch-size','1','--accumulation','1',
            '--val-samples','2','--eval-every','1','--no-pretrained']
        main([*common,'--steps','1']);main([*common,'--steps','2','--resume',str(run/'last.pt')])
        ck=torch.load(run/'last.pt',weights_only=True)
        assert ck['online_cursor']==2 and ck['config']['action_contract']==css_owners.CONTRACT
        assert 'validation_by_teacher' in list(read_jsonl(run/'validation.jsonl'))[-1]

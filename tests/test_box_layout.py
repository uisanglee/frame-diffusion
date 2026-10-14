from types import SimpleNamespace
import pytest

from framediff.html_bridge import HtmlBrowser
from framediff.plans import dom_tree
from framediff.visual import annotate
from framediff import box_layout


@pytest.mark.parametrize('css,body',[
    ('.p{height:auto;width:300px}.c{height:100%}.s{height:100px}',
     '<div class="p"><div class="c">child</div><div class="s">sibling</div></div>'),
    ('.p{display:flex;gap:20px;width:400px;padding:12px;border:3px solid red}.c{width:80px;height:40px;background:blue}',
     '<div class="p"><div class="c"></div><div class="c"></div></div>'),
    ('.p{display:grid;grid-template-columns:1fr 1fr;gap:10px;width:300px}.c{height:60px;background:green}',
     '<div class="p"><div class="c"></div><div class="c"></div></div>'),
    ('.p{width:200px;height:80px;overflow:hidden;border:2px solid red;padding:8px}.c{width:250px;height:120px;background:blue}.s{margin-top:400px;height:30px;background:green}',
     '<div class="p"><div class="c"></div></div><div class="s">below viewport</div>'),
])
def test_preserves_original_and_exact_ce_path(tmp_path,css,body):
    import random
    import torch
    from framediff.tree_edits import repair_path
    from framediff.tree_policy import TreeConfig,TreePolicy,batch_inputs
    html=f'<html><head><style>body{{margin:0}}{css}</style></head><body>{body}</body></html>'
    with HtmlBrowser() as browser:
        original=tmp_path/'original.png'
        dom=browser.snapshot(html,[500,300],original,32)
        tree,boxes,_=dom_tree(dom);annotate(browser,tree)
        source=browser.page.content()
        qa=box_layout.convert(browser,source,[500,300],tmp_path/'box',tree,max_pixel_mae=.001,max_box_error=.1)
        assert qa['accepted'],qa
        assert qa['parent_relationships_preserved']
        parsed=box_layout.read(browser,tree)
        from framediff.visual import elements
        target=dict(id='page',group='page',split='train',source_sha='x',current=tree,
            viewport=[500,300],target_image=str(original),target_elements=elements(tree,boxes,[500,300]),
            css_owners=parsed['owners'],target_declaration_state=parsed['state'],box_layout_contract=box_layout.CONTRACT)
        row=box_layout.generate(browser,target,random.Random(4),0,4,'screenshot')
        path=repair_path(row['declaration_state'],parsed['state'],fields=box_layout.FIELDS)
        assert path and all(e[1] in box_layout.FIELDS and e[2].endswith('px') for e in path)
        for edit in path:box_layout.execute(browser,parsed['owners'],edit)
        assert box_layout.read(browser,tree)['state']==parsed['state']
        for key,box in browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]]).items():
            assert box==pytest.approx(boxes[key],abs=.1)
    torch.set_num_threads(2)
    for mode in ('screenshot','abstract'):
        cfg=TreeConfig(mode=mode,size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=32,
                       stylesheets=True,action_contract=box_layout.CONTRACT,existing_values_only=True)
        net=TreePolicy(cfg,False)
        inputs=batch_inputs(net,[row],'cpu')
        loss,count=net.loss(*inputs,[row['replacement_edit']])
        assert torch.isfinite(loss) and count>0
        loss.backward()
        assert net.output.weight.grad.abs().sum()>0
        if mode=='abstract':
            predicted_row={**row,'lazy_predicted_elements':row['target_elements']}
            detector_loss,_=net.loss(*batch_inputs(net,[predicted_row],'cpu',predicted=True),[row['replacement_edit']])
            assert torch.isfinite(detector_loss)


def test_box_tokenizer_never_remove_or_percent():
    from framediff.tree_edits import EditTokenizer
    tok=EditTokenizer(8,box_layout.FIELDS)
    seq=tok.encode([1,'x','-12.5px','important'],3)
    prefix=[]
    for token in seq:
        assert token in tok.allowed(prefix,3)
        prefix.append(token)
    assert tok.decode(seq,3)==[1,'x','-12.5px','important']
    assert tok.ids['REMOVE'] not in tok.allowed([tok.ids['N1'],tok.ids['x']],3)


def test_parent_move_keeps_child_relative_position(tmp_path):
    with HtmlBrowser() as browser:
        html='<html><body style="margin:0"><div style="width:200px;height:100px;background:red"><div style="width:60px;height:20px;background:blue">child</div></div></body></html>'
        dom=browser.snapshot(html,[400,300],None,32)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        qa=box_layout.convert(browser,browser.page.content(),[400,300],tmp_path,tree)
        assert qa['accepted'],qa
        state=box_layout.read(browser,tree)
        indices=[i for i,s in enumerate(state['state']) if s[0][0]]
        parent,child=indices[:2]
        ids=[state['owners'][i]['id'] for i in (parent,child)]
        all_ids=[n['id'] for n in tree['nodes'][1:]]
        before=browser.tagged_boxes(all_ids)
        x=float(state['state'][parent][0][0][:-2])
        box_layout.execute(browser,state['owners'],[parent,'x',f'{x+20:g}px','important'])
        after=browser.tagged_boxes(all_ids)
        for id in ids:assert after[id][0]==pytest.approx(before[id][0]+20)
        assert box_layout.read(browser,tree)['state'][child]==state['state'][child]


@pytest.mark.parametrize('safe_flow',[False,True])
def test_preparation_online_training_resume_and_rollout(tmp_path,monkeypatch,safe_flow):
    import torch
    from framediff.cli import main
    from framediff.ir import write_jsonl,read_jsonl,read_json
    sources=[]
    for split in ('train','val','test'):
        html=tmp_path/f'{split}.html'
        html.write_text(f'<html><body style="margin:0"><div style="width:120px;height:60px;background:red">{split}</div></body></html>')
        sources.append(dict(id=split,group=split,split=split,html=str(html),viewport=[300,200]))
    manifest=tmp_path/'source.jsonl';write_jsonl(manifest,sources)
    rendered=tmp_path/'rendered';labels=tmp_path/'labels';frozen=tmp_path/'frozen'
    main(['visual-build-data','--manifest',str(manifest),'--out',str(rendered),'--clean-only'])
    prepare=['visual-tree-normalize','--rendered',str(rendered),'--out',str(labels),'--resume']
    if not safe_flow:prepare.append('--box-layout')
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
    assert ck['config']['action_contract']==(OWNER_CONTRACT if safe_flow else box_layout.CONTRACT)
    assert ck['config']['safe_sizes']==safe_flow and ck['online_cursor']==2
    from framediff.tree_policy import TreePolicy,TreeConfig
    from framediff.visual_experiment import rollout
    original=list(read_jsonl(rendered/'policy-test.jsonl'))[0]
    net=TreePolicy(TreeConfig(**ck['config']),False);net.load_state_dict(ck['model']);net.eval()
    from pathlib import Path
    with HtmlBrowser() as browser:
        converted,stats,_=rollout(browser,Path(original['current_html']).read_text(),original['current'],
            original['viewport'],original['target_image'],net,None,.4,0,0)
    assert not stats['failed'],stats
    assert stats['safe_size_conversion' if safe_flow else 'box_conversion']['accepted']
    assert ('data-tuide-safe-sizes' if safe_flow else 'data-tuide-box-layout') in converted
    if safe_flow:
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
                row['parser_provenance']={'parser_training_groups':[],'fixture':True}
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
    if not safe_flow:assert result['box_layout_contract']==box_layout.CONTRACT


def test_unsupported_transform_is_reported_not_relabeled(tmp_path):
    with HtmlBrowser() as browser:
        dom=browser.snapshot('<html><body><div style="transform:rotate(10deg);width:80px;height:30px">a</div></body></html>',[300,200],None,16)
        tree,_,_=dom_tree(dom);annotate(browser,tree)
        qa=box_layout.convert(browser,browser.page.content(),[300,200],tmp_path,tree)
        assert qa['accepted'] is False
        assert qa['errors']
        assert not (tmp_path/'normalized.html').exists()

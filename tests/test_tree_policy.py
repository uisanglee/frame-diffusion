import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from framediff.tree_edits import (CONTRACT, FIELDS, EditTokenizer, apply_state, repair_path,
                                 extract, execute, prepare, value_valid)


def empty_state(n=3): return [[['',''] for _ in FIELDS] for _ in range(n)]


def test_path_replaces_actual_declarations_not_inverse_history():
    target=empty_state();target[1][0]=['50%',''];target[2][1]=['auto','important']
    current=copy.deepcopy(target);current[1][0]=['213px','important'];current[2][1]=['140px','important']
    current[1][2]=['15px','important']
    path=repair_path(current,target,12)
    assert len(path)==3 and [1,'margin-left','',''] in path
    result=current
    for edit in path: result=apply_state(result,edit)
    assert result==target and current!=target
    # Multiple corruptions of one declaration still need one replacement.
    current[1][0]=['300px','important']
    assert len(repair_path(current,target))==3


@pytest.mark.parametrize('edit',[
    [1,'width','50%',''],[1,'height','auto','important'],[2,'margin-left','-12.5px','important'],
    [1,'width','',''],[1,'height','0',''],[1,'width','fit-content',''],
])
def test_edit_token_grammar_roundtrip(edit):
    tok=EditTokenizer(8);tokens=tok.encode(edit,3);prefix=[]
    for token in tokens:
        assert token in tok.allowed(prefix,3)
        prefix.append(token)
    assert tok.decode(tokens,3)==edit
    assert 'STOP' not in tok.tokens


def test_grammar_rejects_invalid_width_and_unbounded_expression():
    assert not value_valid('-1px','width')
    assert not value_valid('calc(100% - 1px)','width')
    tok=EditTokenizer(4)
    with pytest.raises(ValueError): tok.encode([1,'padding','1px',''],3)
    assert tok.ids['N0'] not in tok.allowed([],3)
    assert tok.ids['N3'] not in tok.allowed([],3)


def test_token_ce_gradient_and_checkpoint(tmp_path):
    from framediff.tree_policy import KIND,TreeConfig,TreePolicy
    from framediff.ir import node
    from framediff.visual import current_features,visual_batch,load_policy
    torch.set_num_threads(2)
    cfg=TreeConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=8)
    net=TreePolicy(cfg,False)
    tree={'version':1,'nodes':[node('root',None,'page'),node('a','root','card')]}
    batch=visual_batch([current_features(tree,{'root':[0,0,100,100],'a':[0,0,50,50]},[100,100],8)],'cpu')
    image=torch.randn(1,4,32,32)
    loss,count=net.loss(batch,image,image,[empty_state(2)],[[1,'width','50%','']])
    assert torch.isfinite(loss) and count==8
    labels=torch.tensor([net.tokenizer.encode([1,'width','50%',''],2)])
    inputs=labels.roll(1,1);inputs[:,0]=net.tokenizer.ids['BOS']
    memory,padding=net.memory(batch,image,image,[empty_state(2)])
    expected=torch.nn.functional.cross_entropy(net.logits(memory,padding,inputs).flatten(0,1),labels.flatten())
    assert torch.allclose(loss,expected)
    loss.backward();assert net.output.weight.grad.abs().sum()>0
    checkpoint=tmp_path/'policy.pt'
    torch.save({'kind':KIND,'config':asdict(cfg),'model':net.state_dict()},checkpoint)
    restored,_=load_policy(checkpoint,'cpu')
    assert torch.allclose(loss,restored.loss(batch,image,image,[empty_state(2)],[[1,'width','50%','']])[0])
    action=restored.predict(batch,image,image,[empty_state(2)])
    assert action[0]==1 and action[1] in FIELDS


HTML='''<html><head><style>body{margin:0} section{width:300px;display:flex} .card{height:50px}</style></head>
<body><section><div class="card" style="width:50%;margin:1px 2px 3px 4px">A</div><div class="card">B</div></section></body></html>'''


@pytest.mark.browser
def test_exact_css_replacement_restores_percent_absence_shorthand_and_reflow(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    with HtmlBrowser() as browser:
        dom=browser.snapshot(HTML,[500,300],max_nodes=12);tree,boxes,_=dom_tree(dom)
        target=extract(browser,dom['html'],tree)
        index=next(i for i,n in enumerate(tree['nodes']) if n['id']=='fd-2')
        execute(browser,tree,[index,'width','200px','important'])
        execute(browser,tree,[index,'height','90px','important'])
        execute(browser,tree,[index,'margin-left','30px','important'])
        current=extract(browser,browser.page.content(),tree)
        assert current['fixed']==target['fixed']
        for action in repair_path(current['state'],target['state']): execute(browser,tree,action)
        assert extract(browser,browser.page.content(),tree)==target
        restored=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
        for key in restored: assert restored[key]==pytest.approx(boxes[key],abs=.01)
        # A different action/program can still meet the image goal. This must
        # not be rejected merely because it differs from the sampled teacher.
        target_png=browser.page.screenshot()
        execute(browser,tree,[index,'width','50%','important'])
        assert extract(browser,browser.page.content(),tree)['state']!=target['state']
        from PIL import Image,ImageChops
        import io
        assert ImageChops.difference(Image.open(io.BytesIO(target_png)),
               Image.open(io.BytesIO(browser.page.screenshot()))).getbbox() is None


@pytest.mark.browser
def test_cached_prepare_train_resume_and_rollout(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate,elements,load_policy
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.cli import main
    from framediff.visual_experiment import rollout
    source=tmp_path/'source';source.mkdir()
    with HtmlBrowser() as browser:
        for split in ('train','val','test'):
            work=source/split;work.mkdir()
            dom=browser.snapshot(HTML,[500,300],work/'target.png',12)
            tree,boxes,_=dom_tree(dom);annotate(browser,tree)
            (work/'clean.html').write_text(dom['html'])
            base={'group':split,'split':split,'source_sha':split,'viewport':[500,300],
                  'target_image':str(work/'target.png'),'current_image':str(work/'target.png'),
                  'current':tree,'current_boxes':boxes}
            clean={**base,'id':split+'/clean','current_html':str(work/'clean.html')}
            execute(browser,tree,[3,'width','200px','important'])
            (work/'broken.html').write_text(browser.page.content())
            browser.page.screenshot(path=str(work/'broken.png'))
            edited_boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
            edited_boxes[tree['nodes'][0]['id']]=[0,0,500,300]
            broken={**base,'id':split+'/t0-s0','current_html':str(work/'broken.html'),
                    'current_image':str(work/'broken.png'),'current_boxes':edited_boxes}
            write_jsonl(source/f'policy-{split}.jsonl',[clean,broken])
            write_jsonl(source/f'pages-{split}.jsonl',[{'id':split}])
    prepared=tmp_path/'prepared';prepare(SimpleNamespace(rendered=str(source),out=str(prepared),seed=42,resume=False))
    before=(prepared/'policy-train.jsonl').read_bytes()
    prepare(SimpleNamespace(rendered=str(source),out=str(prepared),seed=42,resume=True))
    assert (prepared/'policy-train.jsonl').read_bytes()==before
    rows=list(read_jsonl(prepared/'policy-train.jsonl'));assert len(rows)==1
    assert rows[0]['replacement_edit']==[3,'width','50%',''] and 'teacher_edits' not in rows[0]
    common=['--train',str(prepared/'policy-train.jsonl'),'--val',str(prepared/'policy-val.jsonl'),
            '--out',str(tmp_path/'policy'),'--mode','abstract','--size','32','--hidden','16','--layers','1',
            '--heads','2','--max-nodes','16','--token-grid','2','--device','cpu','--cpu-threads','2',
            '--batch-size','1','--accumulation','1','--val-samples','1','--eval-every','1','--no-pretrained']
    main(['visual-tree-train',*common,'--steps','1'])
    main(['visual-tree-train',*common,'--steps','2','--resume',str(tmp_path/'policy/last.pt')])
    net,ck=load_policy(tmp_path/'policy/last.pt','cpu');assert ck['step']==2
    metrics=list(read_jsonl(tmp_path/'policy/validation.jsonl'))[-1]
    assert 'action_accuracy' not in metrics
    # Execute a known valid predicted edit, not a target-box-ranked candidate.
    net.predict=lambda *a:[3,'width','50%','']
    row=rows[0]
    with HtmlBrowser() as browser:
        html,stats,_=rollout(browser,Path(row['current_html']).read_text(),row['current'],row['viewport'],
            row['target_image'],net,steps=1,oracle_elements=row['target_elements'])
        assert stats['actions']==1 and stats['browser_screenshots']==0
        assert stats['stop_reason']=='steps' and stats['pair_image_encodings']==1


from pathlib import Path

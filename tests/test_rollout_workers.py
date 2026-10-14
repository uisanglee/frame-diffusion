import copy
import os
from types import SimpleNamespace

import pytest
import torch

from framediff.tree_policy import TreeConfig,TreePolicy,tree_batch
from framediff.tree_edits import FIELDS
from framediff.rollout_workers import merge_requests


def test_batched_memory_and_decode_match_single_pages(monkeypatch):
    from framediff.ir import node
    torch.set_num_threads(1);torch.manual_seed(4)
    cfg=TreeConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=8,existing_values_only=True)
    net=TreePolicy(cfg,False).eval();payloads=[]
    for n in (2,4):
        tree={'version':1,'nodes':[node('root',None,'page')]+[node(str(i),'root','card') for i in range(1,n)]}
        boxes={v['id']:[0,0,30,30] for v in tree['nodes']}
        row={'current':tree,'current_boxes':boxes,'viewport':[100,100]}
        state=[[['',''] for _ in FIELDS] for _ in range(n)];state[1][0]=['20px','']
        batch=tree_batch([row],'cpu',8)
        payloads.append(dict(batch={k:v.numpy() if torch.is_tensor(v) else v for k,v in batch.items()},
            target=torch.rand(1,8,32,32).numpy(),current=torch.rand(1,8,32,32).numpy(),states=[state],allowed_owners=[1]))
    combined=merge_requests(payloads,'cpu')
    with torch.inference_mode():
        memory,padding=net.memory(*combined)
        for i,p in enumerate(payloads):
            single=merge_requests([p],'cpu');m,mask=net.memory(*single)
            assert torch.allclose(memory[i][~padding[i]],m[0][~mask[0]],atol=1e-5)
    # Deterministic legal logits exercise batched autoregressive EOS/grammar.
    sequence=net.tokenizer.encode([1,'width','12px',''],2)
    def logits(memory,padding,tokens):
        scores=torch.zeros(len(tokens),tokens.shape[1],len(net.tokenizer.tokens))
        scores[:,-1,sequence[min(tokens.shape[1]-1,len(sequence)-1)]]=10
        return scores
    monkeypatch.setattr(net,'logits',logits)
    batch_actions=net.predict_batch(*combined,allowed_owners=[[1],[1]])
    singles=[net.predict(*merge_requests([p],'cpu'),allowed_owners=[1]) for p in payloads]
    assert batch_actions==singles==[[1,'width','12px','']]*2


def make_corpus(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate,elements
    from framediff import css_owners
    from framediff.ir import write_jsonl
    rows=[];pages=[]
    with HtmlBrowser() as browser:
        for i in range(4):
            folder=tmp_path/str(i);folder.mkdir()
            html=f'<html><body style="margin:0"><div style="width:200px;height:80px;background:red">page {i}</div></body></html>'
            dom=browser.snapshot(html,[320,200],folder/'target.png',16)
            tree,boxes,_=dom_tree(dom);annotate(browser,tree)
            parsed=css_owners.read(browser,tree)
            (folder/'target.html').write_text(browser.page.content())
            owner=next(j for j,s in enumerate(parsed['state']) if s[0][0]=='200px')
            css_owners.execute(browser,parsed['owners'],[owner,'width','120px',''])
            (folder/'current.html').write_text(browser.page.content())
            browser.page.screenshot(path=str(folder/'current.png'))
            current=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
            current[tree['nodes'][0]['id']]=[0,0,320,200]
            rows.append(dict(id=f'p{i}/sample',group=f'p{i}',source_sha=f's{i}',split='test',
                viewport=[320,200],current=tree,current_boxes=current,
                current_html=str(folder/'current.html'),current_image=str(folder/'current.png'),
                target_html=str(folder/'target.html'),target_image=str(folder/'target.png'),
                target_elements=elements(tree,boxes,[320,200]),css_owners=parsed['owners'],
                declaration_state=css_owners.read(browser,tree)['state'],target_declaration_state=parsed['state']))
            pages.append(dict(id=f'p{i}',evaluation_target_boxes=boxes))
    write_jsonl(tmp_path/'policy-test.jsonl',rows);write_jsonl(tmp_path/'pages-test.jsonl',pages)
    return rows,pages


class FixedPolicy(torch.nn.Module):
    def __init__(self,mode):
        super().__init__()
        from framediff.css_owners import CONTRACT
        from framediff.tree_edits import EditTokenizer
        self.weight=torch.nn.Parameter(torch.zeros(1))
        self.cfg=TreeConfig(mode=mode,size=32,hidden=16,layers=1,heads=2,max_nodes=32,
                            stylesheets=True,action_contract=CONTRACT,existing_values_only=True)
        self.tokenizer=EditTokenizer(32);self.batch_sizes=[];self.call_pids=[]
    def encode_pair(self,*a):return None
    def predict(self,batch,target,current,states,*a,**kw):
        owner=next(i for i,state in enumerate(states[0]) if state[0][0])
        return [owner,'width','200px','']
    def predict_batch(self,batch,target,current,states,**kw):
        self.batch_sizes.append(len(states));self.call_pids.append(os.getpid())
        return [self.predict(batch,target,current,[state]) for state in states]


class FixedDetector(torch.nn.Module):
    def __init__(self,items):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.items=items;self.sizes=[]
    def forward(self,images):
        self.sizes.append(len(images))
        return [dict(boxes=torch.tensor([e['box'] for e in self.items]),
            labels=torch.tensor([e['label'] for e in self.items]),scores=torch.ones(len(self.items))) for _ in images]


@pytest.mark.parametrize('mode',['screenshot','abstract'])
def test_parallel_rollout_matches_serial_and_resumes(tmp_path,monkeypatch,mode):
    from framediff import fixed_test
    from framediff.cli import main
    from framediff.ir import read_json,read_jsonl
    rows,pages=make_corpus(tmp_path)
    policy=FixedPolicy(mode);parser=FixedDetector(rows[0]['target_elements'])
    monkeypatch.setattr(fixed_test,'load_policy',lambda *a:(policy,{}))
    monkeypatch.setattr(fixed_test,'load_detector',lambda *a:(parser,{}))
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixture')
    flag='--raw-checkpoint' if mode=='screenshot' else '--abstract-checkpoint'
    base=['visual-test-rollout','--data',str(tmp_path/'policy-test.jsonl'),
          '--pages',str(tmp_path/'pages-test.jsonl'),flag,str(checkpoint),
          '--detector-checkpoint',str(checkpoint),'--steps','1','--device','cpu','--cpu-threads','1']
    main([*base,'--out',str(tmp_path/'serial')])
    main([*base,'--out',str(tmp_path/'parallel'),'--workers','2','--inference-batch-size','2','--batch-wait-ms','1000'])
    serial=list(read_jsonl(tmp_path/'serial/results.jsonl'))
    parallel=list(read_jsonl(tmp_path/'parallel/results.jsonl'))
    for a,b in zip(serial,parallel):
        for key in ('id','failed','final_box_iou','pixel_mae','actions','final_symbolic_distance','rolled_back_actions'):
            assert a[key]==b[key],(key,a,b)
        assert not b['failed']
    assert max(policy.batch_sizes)==2 and set(policy.call_pids)=={os.getpid()}
    if mode=='abstract':assert max(parser.sizes)==2
    root=tmp_path/'parallel'
    files={p:p.stat().st_mtime_ns for p in root.glob('pages/*/result.json')}
    # Missing artifact is incomplete; all other pages must be retained.
    from pathlib import Path
    Path(parallel[0]['png']).unlink()
    main([*base,'--out',str(root),'--workers','3','--inference-batch-size','1','--resume'])
    report=read_json(root/'execution.json')
    assert report['cached_pages']==3 and report['pending_pages']==1
    assert sum(p.stat().st_mtime_ns==stamp for p,stamp in files.items())==3
    assert len(list(read_jsonl(root/'results.jsonl')))==4
    # Complete rerun with a different worker count performs no inference.
    count=len(policy.batch_sizes)
    main([*base,'--out',str(root),'--workers','1','--resume'])
    assert len(policy.batch_sizes)==count and read_json(root/'execution.json')['cached_pages']==4


def test_interrupt_closes_browsers_and_keeps_completed_pages(tmp_path,monkeypatch):
    from framediff.rollout_workers import parallel_pages
    from framediff.fixed_test import cached_page
    from test_browser_lifecycle import descendants,assert_processes_gone
    rows,pages=make_corpus(tmp_path);policy=FixedPolicy('screenshot')
    tracked={};original=policy.predict_batch
    def predict(*args,**kwargs):
        tracked.update(descendants(os.getpid()))
        return original(*args,**kwargs)
    monkeypatch.setattr(policy,'predict_batch',predict)
    args=SimpleNamespace(repeats=1,threshold=.4,steps=2,time_budget=0.,goal_threshold=-1.,abstract_goal_threshold=-1.)
    tasks=[(i,args,r,p,tmp_path/'out') for i,(r,p) in enumerate(zip(rows,pages))]
    stream=parallel_pages(tasks,{'screenshot-policy':policy},None,2,2,100,{})
    index,result=next(stream)
    stream.close()  # Equivalent cancellation path to interruption of the consumer.
    assert tracked
    assert_processes_gone(tracked)
    assert cached_page(tmp_path/'out',rows[index],{'screenshot-policy':policy},1)==result
    missing=[task for task in tasks if cached_page(tmp_path/'out',task[2],{'screenshot-policy':policy},1) is None]
    assert missing  # Later unsubmitted pages remain pending, not falsely completed.
    list(parallel_pages(missing,{'screenshot-policy':policy},None,2,2,10,{}))
    assert all(cached_page(tmp_path/'out',r,{'screenshot-policy':policy},1) for r in rows)


def test_batch_error_falls_back_without_poisoning_other_pages(monkeypatch):
    from framediff import rollout_workers as mod
    class Reply:
        def __init__(self):self.value=None;self.closed=False
        def send(self,value):self.value=value
        def close(self):self.closed=True
    def infer(kind,payloads,*args):
        if len(payloads)>1:raise RuntimeError('oversized batch')
        if payloads[0]=='bad':raise ValueError('bad page')
        return ['action']
    monkeypatch.setattr(mod,'infer',infer)
    good=Reply();bad=Reply();stats={}
    mod._serve([('policy','good',good),('policy','bad',bad)],{},None,stats)
    assert good.value==(True,'action') and bad.value==(False,'ValueError: bad page')
    assert good.closed and bad.closed and stats['batch_fallbacks']==1

import copy
import json
import random
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from framediff.ir import node,execute,read_json,write_jsonl
from framediff.plans import oracle_plan,encode_plan,plan_metrics,validate_plan,measure,dom_tree
from framediff.model import ModelConfig,EditDenoiser,collate,load_model
from framediff.train import sample,loss_for
from framediff.plan_search import repair_ir,repair_html


def pair():
    clean={'version':1,'nodes':[node('page',None,'page',padding=0,gap=0),
        node('a','page','card',width=128,height=20),node('b','page','card',width=128,height=20,order=1)]}
    current=copy.deepcopy(clean);current['nodes'][1]['props']['width']=64
    obs={'viewport':[400,300],'target':execute(clean,[400,300])}
    return clean,current,obs


def test_plan_truth_not_in_geometry_and_loss_recomputed():
    clean,current,obs=pair();plan=oracle_plan(clean,obs['target'],obs['viewport'])
    a=encode_plan(current,plan);b=encode_plan(clean,plan)
    assert torch.count_nonzero(a['geometry'][:,:,4:13])==0
    assert a['plan_loss']>b['plan_loss']
    assert plan_metrics(plan,obs['target'])['plan_loss']==pytest.approx(0,abs=1e-8)
    assert not torch.equal(a['plan'],b['plan'])
    invalid=copy.deepcopy(plan);invalid['constraints'][0]['a']='invented'
    with pytest.raises(ValueError):validate_plan(invalid,current,obs['viewport'])


def test_relations_and_fixed_interval_semantics():
    boxes={'a':[10,10,20,20],'b':[40,10,20,20],'p':[0,0,100,100]}
    assert measure({'kind':'left_of','a':'a','b':'b'},boxes,[100,100])==pytest.approx(.1)
    assert measure({'kind':'inside','a':'a','b':'p'},boxes,[100,100])==0
    boxes['a'][0]=-10
    assert measure({'kind':'inside','a':'a','b':'p'},boxes,[100,100])==pytest.approx(.1)


def test_plan_checkpoint_training_and_mismatch(tmp_path):
    clean,current,obs=pair();cfg=ModelConfig(hidden=16,layers=1,heads=2,conditioning='plan')
    record={'clean':clean,'current':current,'observations':[obs]}
    samples=[sample(record,random.Random(3),cfg,3,clean_probability=0,online=False)]
    model=EditDenoiser(cfg);loss,_=loss_for(model,samples,'cpu','edits');loss.backward()
    assert torch.isfinite(loss) and model.plan_encoder[0].weight.grad.abs().sum()>0
    path=tmp_path/'model.pt';torch.save({'config':asdict(cfg),'model':model.state_dict()},path)
    restored,_=load_model(path)
    assert restored.cfg.conditioning=='plan'
    legacy=EditDenoiser(ModelConfig(hidden=16,layers=1,heads=2))
    with pytest.raises(ValueError,match='conditioning'):legacy(collate([samples[0][0]]))


def test_coordinate_plan_search_improves_without_target_boxes():
    clean,current,obs=pair();plan=oracle_plan(clean,obs['target'],obs['viewport'])
    before=plan_metrics(plan,execute(current,obs['viewport']))['plan_loss']
    result,stats=repair_ir(current,plan,method='coordinate',steps=4,beam=1,topk=256,budget=1024)
    assert result['plan_loss']<before
    assert all(b['plan_loss']<=a['plan_loss'] for a,b in zip(stats['history'],stats['history'][1:]))
    assert stats['browser_screenshots']==0


HTML='''<html><body style="margin:0;min-height:300px"><main style="width:300px">
<div style="width:240px;font:16px/20px monospace">one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen</div>
<div style="height:30px">next</div></main></body></html>'''


@pytest.mark.browser
def test_plan_browser_reflow_without_intermediate_screenshots(tmp_path,monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    with HtmlBrowser() as browser:
        dom=browser.snapshot(HTML,[400,300],tmp_path/'initial.png')
        tree,boxes,_=dom_tree(dom);html=dom['html'];ids=[n['id'] for n in tree['nodes'][1:]]
        i=next(i for i,n in enumerate(tree['nodes']) if n['id']=='fd-2')
        from framediff.html_feedback import action_unit
        value=tree['nodes'][i]['props']['width']-64
        delta=-64*action_unit(tree,boxes,i,'width')
        browser.edit_property(html,[400,300],'fd-2','width',delta)
        target=browser.tagged_boxes(ids);target[tree['nodes'][0]['id']]=[0,0,400,300]
        plan=oracle_plan(tree,target,[400,300]);shots=browser.screenshots
        monkeypatch.setattr('framediff.plan_search.proposals',lambda *a,**k:[(i,'width',value)])
        result,stats=repair_html(browser,html,tree,plan,method='coordinate',steps=1,beam=1,trace_dir=tmp_path/'steps')
        assert result['boxes']['fd-3'][1]>boxes['fd-3'][1]
        assert 'fd-3' in stats['candidate_log'][0]['changed_nodes']
        assert stats['browser_screenshots']==0 and browser.screenshots==shots
        assert (tmp_path/'steps/step-001.png').exists()
        assert stats['history'][-1]['plan_loss']<stats['history'][0]['plan_loss']


@pytest.mark.browser
def test_one_shot_plan_web_pipeline(tmp_path,monkeypatch):
    from framediff.web_experiment import prepare,repair_pages,evaluate_pages
    root=tmp_path/'input';root.mkdir();Image.new('RGB',(400,300),'white').save(root/'a.png')
    (root/'a.html').write_text(HTML.replace('next','SECRET_REFERENCE'))
    seen=[]
    def generate(args,prompt,images,runtime):
        assert 'SECRET_REFERENCE' not in prompt
        seen.append(args.task)
        if args.task=='generate-html':return HTML,{}
        assert args.task=='repair-plan' and len(images)==3
        tree=read_json(args.current)
        return json.dumps({'version':1,'viewport':[400,300],'constraints':[
            {'kind':'width','a':tree['nodes'][-1]['id'],'low':.2,'high':.3,'weight':1.}]}),{}
    monkeypatch.setattr('framediff.vlm.generate',generate)
    args=SimpleNamespace(root=str(root),manifest=None,out=str(tmp_path/'prepare'),rounds=0,
        max_nodes=16,max_pixels=200000,max_new_tokens=1024,limit=0,seed=42,resume=False,
        backend='openai-compatible',model='mock',revision='main',endpoint='unused',api_key_env='TEST',
        four_bit=False,initial_mode='direct',repair_conditioning='plan',vlm_retries=0)
    rows=prepare(args)
    assert seen==['generate-html','repair-plan'] and rows[0]['errors']=={}
    assert 'observations' not in rows[0] and rows[0]['plan_vlm_calls']==1
    cfg=ModelConfig(hidden=16,layers=1,heads=2,max_nodes=16,conditioning='plan');model=EditDenoiser(cfg)
    checkpoint=tmp_path/'plan.pt'
    torch.save({'config':asdict(cfg),'model':model.state_dict(),'objective':'edits','training_groups':[]},checkpoint)
    repaired=repair_pages(SimpleNamespace(data=str(tmp_path/'prepare/prepared.jsonl'),out=str(tmp_path/'repair'),
        checkpoint=str(checkpoint),device='cpu',cpu_threads=1,methods='coordinate-plan,model-plan',
        steps=1,beam=1,topk=2,budget=2,seed=42,resume=False))
    for method in ('coordinate-plan','model-plan'):
        assert not repaired[0]['methods'][method]['failed']
        assert repaired[0]['methods'][method]['feedback_browser_screenshots']==0
        assert repaired[0]['methods'][method]['vlm_calls']==2 # initial generation + planning
    metrics=evaluate_pages(SimpleNamespace(data=str(tmp_path/'repair/results.jsonl'),out=str(tmp_path/'eval'),
        max_nodes=128,official_repo=None,resume=False))
    assert len(metrics)==3 and all(r['evaluation_error'] is None for r in metrics)


@pytest.mark.browser
def test_browser_training_data_verified_edits(tmp_path):
    from framediff.plan_experiment import build_html_data
    path=tmp_path/'train.html';path.write_text(HTML)
    manifest=tmp_path/'manifest.jsonl';write_jsonl(manifest,[{'html':str(path),'group':'train-only','split':'train'}])
    records=build_html_data(SimpleNamespace(manifest=str(manifest),out=str(tmp_path/'data'),
        per_page=2,seed=42,max_nodes=16,width=400,height=300))
    assert len(records)==3 and records[0]['teacher_edits']==[]
    cfg=ModelConfig(hidden=16,layers=1,heads=2,max_nodes=16,conditioning='plan')
    samples=[sample(r,random.Random(42),cfg,5) for r in records]
    loss,_=loss_for(EditDenoiser(cfg),samples,'cpu','edits')
    assert torch.isfinite(loss)

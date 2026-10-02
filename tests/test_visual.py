import copy
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from framediff.ir import node,write_jsonl,read_jsonl
from framediff.visual import (CONTRACT,ACTION_CONTRACT,VisualConfig,VisualPolicy,abstract_image,elements,current_features,
    visual_batch,action_index,ACTIONS,load_policy,validate_action,action_size)


def test_webui_visual_import_domain_split_and_native_labels(tmp_path):
    from framediff.webui_visual import import_webui
    root=tmp_path/'webui'
    for i in range(6):
        folder=root/f'page-{i}';folder.mkdir(parents=True)
        stem=folder/'default_100-80'
        Image.new('RGB',(100,80),'white').save(str(stem)+'-screenshot.png')
        Path(str(stem)+'-html.html').write_text(f'<html><body><button>Page {i}</button></body></html>')
        Path(str(stem)+'-url.txt').write_text(f'https://domain-{i}.example/page')
        nodes={'nodes':[{'nodeId':str(i),'backendDOMNodeId':i+1,'ignored':False,
                         'role':{'value':'button'}}]}
        boxes={str(i+1):{'x':10,'y':12,'width':30,'height':20}}
        with gzip.open(str(stem)+'-axtree.json.gz','wt') as stream:json.dump(nodes,stream)
        with gzip.open(str(stem)+'-bb.json.gz','wt') as stream:json.dump(boxes,stream)
    args=SimpleNamespace(root=str(root),out=str(tmp_path/'out'),view='default_100-80',
        train_count=2,val_count=2,test_count=2,seed=42,resume=False)
    rows=import_webui(args)
    assert {split:sum(r['split']==split for r in rows) for split in ('train','val','test')}=={
        'train':2,'val':2,'test':2}
    groups={split:{r['group'] for r in rows if r['split']==split} for split in ('train','val','test')}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    native=list(read_jsonl(tmp_path/'out/native-detector-train.jsonl'))
    assert len(native)==2 and native[0]['elements']==[{'box':[10.,12.,40.,32.],'label':3}]
    assert native[0]['annotation_scope'].endswith('no-painted-region')


def fixture_tree():
    tree={'version':1,'nodes':[node('page',None,'page'),node('card','page','card',width=128)]}
    tree['nodes'][1]['visual_class']=3
    return tree,{'page':[0,0,320,240],'card':[16,20,80,30]}


def test_validation_sampling_covers_pages_and_is_repeatable():
    from framediff.visual_train import validation_sample
    rows=[{'id':f'{page}/{state}','source_sha':str(page),'target_image':f'{page}.png'}
          for page in range(20) for state in range(13)]
    sample=validation_sample(rows,32)
    assert sample==validation_sample(rows,32)
    assert len(sample)==32 and len({r['source_sha'] for r in sample})==20
    assert len({r['id'] for r in sample})==32
    assert validation_sample(rows,1000)==rows


def test_abstraction_is_id_free_and_spatial():
    tree,boxes=fixture_tree();a=abstract_image(elements(tree,boxes,[320,240]),[320,240],64)
    renamed=copy.deepcopy(tree);renamed['nodes'][1]['id']='other';renamed_boxes={'page':boxes['page'],'other':boxes['card']}
    b=abstract_image(elements(renamed,renamed_boxes,[320,240]),[320,240],64)
    assert np.array_equal(a,b)
    boxes['card'][0]+=20
    c=abstract_image(elements(tree,boxes,[320,240]),[320,240],64)
    assert not np.array_equal(a,c)


def test_structured_action_contract_is_single_and_normalized():
    assert validate_action((1,'margin-left',.05))==(1,'margin-left',.05)
    assert validate_action((1,'justify-content','space-between'))==(1,'justify-content','space-between')
    assert action_size((1,'margin-left',.05))=={'declarations':1,'normalized_magnitude':.05}
    assert action_size((1,'align-items','center'))=={'declarations':1,'normalized_magnitude':None}
    assert ACTIONS==81
    with pytest.raises(ValueError,match='grammar'):validate_action((1,'padding-left',.05))
    with pytest.raises(ValueError,match='grammar'):validate_action((1,'column-gap',.05))
    with pytest.raises(ValueError,match='grammar'):validate_action((1,'unknown',.01))


@pytest.mark.browser
def test_structured_actions_apply_normalized_reflow():
    from framediff.html_bridge import HtmlBrowser
    html='''<html><body style="margin:0"><section id="row" style="display:flex;width:400px;height:120px;gap:0;padding:0">
    <div id="a" style="width:40px;height:20px"></div><div id="b" style="width:40px;height:20px"></div>
    </section></body></html>'''
    viewport=[400,200]
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,viewport,max_nodes=16);base=dom['html']
        ids=browser.page.evaluate("()=>Object.fromEntries(['row','a','b'].map(id=>[id,document.getElementById(id).getAttribute('data-fd-id')]))")
        browser.edit_visual_action(base,viewport,ids['b'],'margin-left',.05)
        gap=browser.page.locator('#b').bounding_box()['x']-browser.page.locator('#a').bounding_box()['x']-40
        assert gap==pytest.approx(20,abs=.1)
        browser.edit_visual_action(base,viewport,ids['a'],'margin-left',.05)
        assert browser.page.locator('#a').bounding_box()['x']==pytest.approx(20,abs=.1)
        inverse=browser.edit_visual_action(base,viewport,ids['row'],'flex-direction','column',return_inverse=True)
        assert inverse==('flex-direction','row')
        assert browser.page.locator('#b').bounding_box()['y']>browser.page.locator('#a').bounding_box()['y']


def test_pixels_drive_policy_and_receive_gradients(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(17)
    tree,boxes=fixture_tree();cfg=VisualConfig(size=64,hidden=16,layers=1,heads=2,token_grid=3,max_nodes=16)
    net=VisualPolicy(cfg,pretrained=False)
    features=current_features(tree,boxes,[320,240],16)
    assert torch.count_nonzero(features['geometry'][:,:,4:13])==0
    assert 'plan' not in features
    batch=visual_batch([features],'cpu');target=torch.rand(1,3,64,64,requires_grad=True);current=torch.rand(1,3,64,64,requires_grad=True)
    logits=net(batch,target,current);label=action_index((1,'width',-.003125),2)
    torch.nn.functional.cross_entropy(logits,torch.tensor([label])).backward()
    assert target.grad.abs().sum()>0 and current.grad.abs().sum()>0
    assert net.vision.projections[0].weight.grad.abs().sum()>0
    net.eval()
    with torch.no_grad():
        before=net(batch,target,current);after=net(batch,torch.zeros_like(target),current)
    valid=torch.isfinite(before)
    assert not torch.allclose(before[valid],after[valid])
    path=tmp_path/'visual.pt';torch.save({'kind':'visual-policy-v3','config':asdict(cfg),'model':net.state_dict()},path)
    restored,_=load_policy(path,'cpu')
    with torch.no_grad():assert torch.allclose(before,restored(batch,target,current))


def test_detector_loss_inference_and_reload(tmp_path):
    from framediff.visual_train import detector,detection_loss,detect,load_detector
    torch.set_num_threads(2)
    cfg={'contract':CONTRACT,'min_size':64,'max_size':96,'max_detections':8}
    net=detector(cfg,pretrained=False);image=tmp_path/'image.png';Image.new('RGB',(96,64),'white').save(image)
    rows=[{'image':str(image),'elements':[{'box':[10.,10.,40.,30.],'label':3}]}]
    net.train();loss=detection_loss(net,rows,'cpu');loss.backward()
    assert torch.isfinite(loss) and net.roi_heads.box_predictor.bbox_pred.weight.grad is not None
    path=tmp_path/'detector.pt'
    torch.save({'kind':'visual-detector-v1','config':cfg,'model':net.state_dict()},path)
    restored,_=load_detector(path,'cpu')
    assert isinstance(detect(restored,torch.ones(3,64,96),0.),list)


HTML='''<html><body style="margin:0;min-height:240px"><main style="width:300px;background:#eee">
<div style="width:220px;background:#acd;font:16px/20px monospace">one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen</div>
<button style="width:70px;height:24px">next</button></main></body></html>'''


@pytest.mark.browser
def test_rollout_uses_one_target_parse_and_real_feedback(tmp_path,monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff.visual_experiment import rollout
    target_path=tmp_path/'target.png';parse_calls=[]
    class ScriptedPolicy(torch.nn.Module):
        def __init__(self,mode,index):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.cfg=VisualConfig(mode=mode,size=64,hidden=16,layers=1,heads=2,max_nodes=16)
            self.index=index;self.calls=0;self.images=[]
        def encode_image(self,image):
            self.images.append(image.clone());return torch.zeros(1,1,16),torch.zeros(1,16,4,4)
        def decode(self,batch,*args):
            n=batch['mask'].shape[1];logits=torch.full((1,n*ACTIONS+1),-100.)
            edit=(self.index,'width',-.05) if self.calls==0 else None
            logits[0,action_index(edit,n)]=100.;self.calls+=1;return logits
    def parse(*args):parse_calls.append(1);return [{'box':[0,0,300,100],'label':4}]
    monkeypatch.setattr('framediff.visual_experiment.detect',parse)
    with HtmlBrowser() as browser:
        dom=browser.snapshot(HTML,[320,240],target_path,15);tree,boxes,_=dom_tree(dom);annotate(browser,tree)
        index=next(i for i,n in enumerate(tree['nodes']) if n['id']=='fd-2')
        outputs=[]
        for mode in ('screenshot','abstract'):
            policy=ScriptedPolicy(mode,index)
            result,stats,_=rollout(browser,dom['html'],tree,[320,240],str(target_path),policy,object(),steps=3,
                                   diagnostic_target_boxes=boxes)
            assert stats['actions']==1 and stats['stop_reason']=='policy_stop'
            assert stats['target_image_encodings']==1 and stats['current_image_encodings']==2
            assert stats['browser_screenshots']==(2 if mode=='screenshot' else 0)
            assert stats['history'][1]['boxes']['fd-2'][2]<boxes['fd-2'][2]
            assert 'box_iou' in stats['history'][0]['diagnostic_metrics']
            assert not torch.equal(policy.images[1],policy.images[2])
            outputs.append(result)
        assert len(parse_calls)==1 and outputs[0]==outputs[1]


@pytest.mark.browser
def test_visual_data_training_resume_and_preparation(tmp_path,monkeypatch):
    from framediff.visual_data import build
    from framediff.cli import main
    from framediff.web_experiment import prepare
    sources=[]
    for i,split in enumerate(('train','val','test')):
        path=tmp_path/f'{split}.html';path.write_text(HTML.replace('next',f'next {i}'))
        sources.append({'id':split,'group':split,'split':split,'html':str(path),'viewport':[320,240]})
    manifest=tmp_path/'manifest.jsonl';write_jsonl(manifest,sources)
    build(SimpleNamespace(manifest=str(manifest),out=str(tmp_path/'data'),resume=False,
        trajectories=1,max_noise=2,seed=17,max_nodes=16,abstract_size=64))
    train=list(read_jsonl(tmp_path/'data/policy-train.jsonl'))
    assert len(train)>=2 and train[0]['teacher_edits']==[]
    assert validate_action(train[1]['teacher_edits'][0])
    common=['--train',str(tmp_path/'data/policy-train.jsonl'),'--val',str(tmp_path/'data/policy-val.jsonl'),
            '--out',str(tmp_path/'policy'),'--mode','abstract','--size','64','--hidden','16','--layers','1',
            '--heads','2','--max-nodes','16','--token-grid','3','--device','cpu','--cpu-threads','2',
            '--batch-size','1','--accumulation','1','--val-samples','1','--eval-every','1','--no-pretrained']
    main(['visual-train-policy',*common,'--steps','1'])
    policy_log=list(read_jsonl(tmp_path/'policy/train.jsonl'))
    assert policy_log[-1]['validation_n']==1 and 'action_accuracy' in policy_log[-1]
    main(['visual-train-policy',*common,'--steps','2','--resume',str(tmp_path/'policy/last.pt')])
    _,ck=load_policy(tmp_path/'policy/last.pt','cpu');assert ck['step']==2
    # Existing initial HTML -> no VLM call, no plan, no target-box extraction.
    def unexpected(*a,**kw):raise AssertionError('VLM must not run for supplied initial HTML')
    monkeypatch.setattr('framediff.vlm.generate',unexpected)
    args=SimpleNamespace(root=None,manifest=str(tmp_path/'data/pages-test.jsonl'),out=str(tmp_path/'prepared'),
        rounds=0,max_nodes=16,max_pixels=65536,max_new_tokens=1024,limit=0,seed=42,resume=False,
        backend='openai-compatible',model='mock',revision='main',endpoint='unused',api_key_env='TEST',
        four_bit=False,initial_mode='direct',repair_conditioning='visual',vlm_retries=0)
    rows=prepare(args)
    assert not rows[0]['errors'] and 'plan' not in rows[0] and 'observations' not in rows[0]
    assert rows[0]['visual_contract']==CONTRACT and rows[0]['visual_action_contract']==ACTION_CONTRACT
    assert rows[0]['frame_vlm_calls']==0
    assert any(n.get('visual_class') for n in rows[0]['current']['nodes'])
    main(['visual-train-detector','--train',str(tmp_path/'data/detector-train.jsonl'),
          '--val',str(tmp_path/'data/detector-val.jsonl'),'--out',str(tmp_path/'detector'),
          '--device','cpu','--cpu-threads','2','--steps','1','--batch-size','1','--accumulation','1',
          '--val-samples','1','--min-size','64','--max-size','96','--max-detections','8','--no-pretrained'])
    detector_log=list(read_jsonl(tmp_path/'detector/train.jsonl'))
    assert 'detector_recall_iou50' in detector_log[-1] and 'validation_components' in detector_log[-1]
    for split in ('train','val'):
        main(['visual-cache-targets','--data',str(tmp_path/f'data/policy-{split}.jsonl'),
              '--checkpoint',str(tmp_path/'detector/best.pt'),'--out',str(tmp_path/f'cached-{split}'),
              '--device','cpu','--cpu-threads','2','--size','64','--threshold','0'])
    fine=list(common)
    for flag,value in [('--train','cached-train/data.jsonl'),('--val','cached-val/data.jsonl'),('--out','fine')]:
        fine[fine.index(flag)+1]=str(tmp_path/value)
    main(['visual-train-policy',*fine,'--steps','1','--init-checkpoint',str(tmp_path/'policy/best.pt'),
          '--prediction-probability','1'])
    raw=list(common);raw[raw.index('--mode')+1]='screenshot';raw[raw.index('--out')+1]=str(tmp_path/'raw')
    main(['visual-train-policy',*raw,'--steps','1'])
    main(['visual-evaluate','--data',str(tmp_path/'prepared/prepared.jsonl'),
          '--raw-checkpoint',str(tmp_path/'raw/best.pt'),'--abstract-checkpoint',str(tmp_path/'fine/best.pt'),
          '--detector-checkpoint',str(tmp_path/'detector/best.pt'),'--out',str(tmp_path/'comparison'),
          '--device','cpu','--cpu-threads','2','--threshold','0','--warmup','0','--steps','1','--repeats','2',
          '--oracle-ablation'])
    results=list(read_jsonl(tmp_path/'comparison/results.jsonl'))
    assert len(results)==2
    assert all(not m['failed'] for r in results for m in r['methods'].values())
    assert results[0]['methods']['abstract-policy']['visual_timing']['abstraction_calls']==1
    main(['web-evaluate','--data',str(tmp_path/'comparison/results.jsonl'),
          '--out',str(tmp_path/'evaluation')])
    assert (tmp_path/'evaluation/report.md').exists()


def test_publication_figures_from_persisted_metrics(tmp_path):
    pytest.importorskip('matplotlib')
    from framediff.cli import main
    run=tmp_path/'run'
    detector=[{'step':50,'loss':2.},
              {'step':50,'validation_loss':1.5,'best':1.5,'detector_precision_iou50':.6,
               'detector_recall_iou50':.5,'detector_small_recall_iou50':.3}]
    policy=[{'step':50,'loss':4.},
            {'step':50,'validation_loss':3.5,'best':3.5,'action_accuracy':.4,'node_accuracy':.7,
             'property_accuracy':.6,'value_accuracy':.5,
             'per_property':{'width':{'n':3,'accuracy':2/3}}}]
    write_jsonl(run/'detector/train.jsonl',detector)
    for stage in ('raw-stage1','policy-raw','abstract-stage1','policy-abstract'):
        write_jsonl(run/f'{stage}/train.jsonl',policy)
    evaluation=tmp_path/'evaluation';trace=evaluation/'pages/page-1/abstract-policy-r1-trace.json'
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({'history':[
        {'step':0,'diagnostic_metrics':{'box_iou':.4,'relation_accuracy':.7}},
        {'step':1,'diagnostic_metrics':{'box_iou':.6,'relation_accuracy':.8}}]}))
    out=tmp_path/'figures'
    main(['visual-figures','--run-root',str(run),'--evaluation',str(evaluation),'--out',str(out)])
    for name in ('training-curves','action-accuracy','per-property-accuracy',
                 'detector-validation','denoising-trajectory'):
        assert (out/f'{name}.png').stat().st_size>0
        assert (out/f'{name}.pdf').stat().st_size>0
    assert json.loads((out/'trajectory-summary.json').read_text())['abstract-policy']['1']['box_iou']==.6

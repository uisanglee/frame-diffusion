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
    visual_batch,action_index,ACTIONS,load_policy,validate_action,action_size,action_is_legal)


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


@pytest.mark.browser
def test_visual_owner_boxes_refresh_after_edit():
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff.explicit_html import feedback_metadata
    from framediff.tree_online import abstraction
    with HtmlBrowser() as browser:
        def measure(width):
            dom=browser.snapshot('<html><body style="margin:0;font:20px/30px Arial">'
                f'<div style="width:{width};height:30px">Hello</div></body></html>',[600,400])
            tree,boxes,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
            return tree,boxes,elements(tree,boxes,[600,400]),browser.page.screenshot()
        wide,wb,wi,wp=measure('600px');tight,tb,ti,tp=measure('max-content')
        assert wp==tp and wi!=ti
        assert wi[0]['box'][2]-wi[0]['box'][0]==600
        owner=wi[0]['owner_id']
        assert wb[owner][2]==600 and tb[owner][2]<60
        browser.page.eval_on_selector(f'[data-fd-id="{owner}"]','e=>e.style.marginLeft="80px"')
        _,boxes,items=abstraction(browser,tight,[600,400],64)
        assert items[0]['box'][0]==80 and boxes[owner][0]==80
        feedback_metadata(browser,tight)
        assert elements(tight,boxes,[600,400])==items


@pytest.mark.browser
def test_visible_selection_excludes_offscreen_and_transparent_ancestors():
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    html='<html><body style="margin:0"><p>Visible</p><div style="opacity:0"><button>Hidden</button></div>'
    html+='<div style="position:absolute;top:1000px">'+'<button>Outside</button>'*140+'</div></body></html>'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[600,400],max_nodes=16)
        assert all(n['name'] not in ('Hidden','Outside') for n in dom['nodes'])
        tree,boxes,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        assert len(elements(tree,boxes,[600,400]))==1


@pytest.mark.browser
def test_text_lines_nested_owners_clipping_and_transparency():
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate,refresh_visual_geometry
    html='<html><body style="margin:0;font:20px/30px Arial">'
    html+='<div style="width:100px">alpha beta gamma delta <span>child</span></div>'
    html+='<div style="height:10px;overflow:hidden"><p style="margin:0">clipped</p></div>'
    html+='<p style="color:transparent">invisible</p></body></html>'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[600,400]);tree,boxes,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        items=elements(tree,boxes,[600,400]);owners={n['name']:n['id'] for n in dom['nodes']}
        assert sum(e['owner_id']==owners['alpha beta gamma delta child'] for e in items)==1
        assert sum(e['owner_id']==owners['child'] for e in items)==1
        assert not any(e['owner_id']==owners['invisible'] for e in items)
        assert all(e['box'][3]-e['box'][1]<=10 for e in items if e['owner_id']==owners['clipped'])
        browser.page.evaluate('document.body.style.opacity="0"')
        refresh_visual_geometry(browser,tree)
        assert elements(tree,boxes,[600,400])==[]


def test_visual_label_contract_rejects_mixed_versions_and_legacy_config_loads():
    from framediff.visual import VISUAL_GEOMETRY_CONTRACT,label_contract
    from framediff.tree_policy import TreeConfig
    assert TreeConfig().visual_geometry_contract is None
    assert label_contract([{}]) is None
    assert label_contract([{'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT}])==VISUAL_GEOMETRY_CONTRACT
    with pytest.raises(ValueError,match='Mixed visual label'):
        label_contract([{}, {'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT}])


@pytest.mark.browser
def test_detector_labels_are_not_limited_by_policy_owner_budget():
    from framediff.html_bridge import HtmlBrowser
    from framediff.visual import detector_labels,annotate
    from framediff.plans import dom_tree
    from framediff.detector_support import compare
    html='<html><body style="margin:0;font:12px/14px Arial">'+''.join(
        f'<div style="height:14px">Visible {i}</div>' for i in range(30))+'</body></html>'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[600,500],max_nodes=4)
        result=detector_labels(browser)
        assert len(dom['nodes'])<=4
        assert sum(e['label']==1 for e in result['elements'])==30
        assert result['label_audit']['untagged_owners']>=26
        assert any(e['owner_id'] is None for e in result['elements'])
        tree,boxes,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        limited=elements(tree,boxes,[600,500])
        prediction=[next(e for e in result['elements'] if e['owner_id'] is None)]
        assert compare(prediction,limited)['matches']
        assert limited==result['elements']
        assert len(tree['nodes'])==len(dom['nodes'])+1  # Synthetic viewport only.
        assert compare(prediction,result['elements'])['matches']


@pytest.mark.browser
def test_unselected_parent_text_in_full_observation_and_refresh():
    import copy
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate,detector_labels,visual_geometry,semantic_masks
    from framediff.tree_online import abstraction
    from framediff.explicit_html import target_abstraction
    html='<body style="margin:0"><h1>Coverage for <b>file.py</b></h1>'
    html+='<h2>41 statements <span>33 run</span><span>8 missing</span></h2>'
    html+='<p>toggle line displays <span>r</span></p>'
    html+=''.join(f'<span>Leaf {i}</span>' for i in range(20))+'</body>'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[1000,600],max_nodes=4,visual_geometry=False)
        assert browser.page.locator('h2').get_attribute('data-fd-id') is None
        tree,boxes,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        identities=[(n['id'],n.get('parent')) for n in tree['nodes']]
        items=elements(tree,boxes,[1000,600])
        assert len(items)>len(tree['nodes'])
        assert items==detector_labels(browser)['elements']
        h2=browser.page.locator('h2').bounding_box()
        expected=[h2['x'],h2['y'],h2['x']+h2['width'],h2['y']+h2['height']]
        assert any(e['owner_id'] is None and e['box']==expected for e in items)
        before=copy.deepcopy(items)
        geometry=visual_geometry(browser)
        # Preservation checks use their own captured geometry, not stale tree data.
        assert (target_abstraction([],set(),[1000,600],64,tree,geometry)
                ==semantic_masks(items,[1000,600],64)).all()
        browser.page.locator('h2').evaluate('e=>e.style.width="400px"')
        _,_,after=abstraction(browser,tree,[1000,600],64)
        assert after!=before and after==detector_labels(browser)['elements']
        assert identities==[(n['id'],n.get('parent')) for n in tree['nodes']]


@pytest.mark.browser
def test_full_detector_labels_background_body_contents_and_unsupported_audit():
    from framediff.html_bridge import HtmlBrowser
    from framediff.visual import detector_labels
    html='''<html><style>.pseudo::before{content:"star"}</style>
    <body style="margin:0;font:14px/20px Arial">Body text
    <span style="display:contents">Contents text</span>
    <div style="height:0;overflow:visible">Overflow text</div>
    <div style="margin-top:30px;width:120px;height:40px;background-image:linear-gradient(red,blue)">Artwork text</div>
    <div class="pseudo" style="height:20px"></div>
    <div style="opacity:0">Hidden text</div>
    <div style="position:absolute;top:1000px">Outside text</div></body></html>'''
    with HtmlBrowser() as browser:
        browser.snapshot(html,[600,400],max_nodes=4)
        result=detector_labels(browser)
        # No owner rectangle for display:contents/zero-height, no body label.
        assert sum(e['label']==1 for e in result['elements'])==1
        assert sum(e['label']==2 for e in result['elements'])==0
        assert result['label_audit']['pseudo_content_owners']==1


@pytest.mark.browser
def test_detector_and_policy_share_owner_rectangle_definition():
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate,detector_labels
    from framediff.explicit_html import feedback_metadata
    html='''<body style="margin:0">
      <div style="width:500px;height:100px">Hello</div>
      <button>Go</button><div style="width:60px;height:20px;background:red"></div>
      <div style="width:100px;height:20px;overflow:hidden"><p style="width:200px;margin:0">Clipped</p></div>
      <div style="opacity:0">Hidden</div></body>'''
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[600,400],max_nodes=100)
        tree,boxes,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        feedback=feedback_metadata(browser,tree)
        actual=elements(tree,boxes,[600,400],feedback['current_clip_boxes'])
        expected=detector_labels(browser)['elements']
        canonical=lambda items:sorted((e['label'],tuple(e['box'])) for e in items)
        assert canonical(actual)==canonical(expected)
        assert (1,(0,0,500,100)) in canonical(actual)


def test_text_fragment_labels_cannot_mix_with_owner_boxes():
    from framediff.visual import label_contract
    with pytest.raises(ValueError,match='Unknown visual label'):
        label_contract([{'visual_geometry_contract':'visible-text-fragments-owner-v1'}])


def test_legacy_preflight_is_read_only_and_rejects_versioned_data(tmp_path):
    import json
    from framediff.visual import validate_legacy_dataset,VISUAL_GEOMETRY_CONTRACT
    for split in ('train','val','test'):
        for folder,kind in (('rendered','detector'),('css','policy')):
            path=tmp_path/'7000'/folder/f'{kind}-{split}.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'id':split+'/clean','current':{'nodes':[]}})+'\n')
    before={p:p.read_bytes() for p in tmp_path.rglob('*.jsonl')}
    assert len(validate_legacy_dataset(tmp_path,7000))==6
    assert all(p.read_bytes()==value for p,value in before.items())
    path=tmp_path/'7000/css/policy-test.jsonl'
    path.write_text(json.dumps({'id':'bad','visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT})+'\n')
    with pytest.raises(ValueError,match='Not legacy'):
        validate_legacy_dataset(tmp_path,7000)
    path.write_text(json.dumps({'id':'bad','lazy_settings':{'policy_relabel':{'source':{}}}})+'\n')
    with pytest.raises(ValueError,match='Not legacy'):
        validate_legacy_dataset(tmp_path,7000)


@pytest.mark.browser
def test_unversioned_data_retains_legacy_classification():
    from framediff.html_bridge import HtmlBrowser
    from framediff.visual import annotate
    with HtmlBrowser() as browser:
        browser.load('<div data-fd-id="a" style="color:transparent;width:500px;height:100px">Hello</div>',[600,400])
        tree={'nodes':[{'id':'a'}]}
        annotate(browser,tree)
        # Old labels classify direct text; newer visibility filtering must not
        # silently alter an already-published unversioned experiment.
        assert tree['nodes'][0]['visual_class']==1
        assert 'visual_geometry_contract' not in tree
        assert elements(tree,{'a':[0,0,500,100]},[600,400])==[{'label':1,'box':[0.,0.,500,100]}]


def test_validation_sampling_covers_pages_and_is_repeatable():
    from framediff.visual_train import validation_sample
    rows=[{'id':f'{page}/{state}','source_sha':str(page),'target_image':f'{page}.png'}
          for page in range(20) for state in range(13)]
    sample=validation_sample(rows,32)
    assert sample==validation_sample(rows,32)
    assert len(sample)==32 and len({r['source_sha'] for r in sample})==20
    assert len({r['id'] for r in sample})==32
    assert validation_sample(rows,1000)==rows


def test_page_view_sampler_does_not_overweight_pages_with_more_views():
    import random
    from framediff.visual_train import sample_page_views, page_view_pools
    rows=[{'id':'a/v1','page_id':'a'},{'id':'a/v2','page_id':'a'},{'id':'a/v3','page_id':'a'},
          {'id':'b/v1','page_id':'b'}]
    sampled=sample_page_views(page_view_pools(rows),4000,random.Random(42))
    a=sum(row['page_id']=='a' for row in sampled)
    assert 1800<a<2200
    assert {row['id'] for row in sampled if row['page_id']=='a'}=={'a/v1','a/v2','a/v3'}


def test_abstraction_is_id_free_and_spatial():
    tree,boxes=fixture_tree();a=abstract_image(elements(tree,boxes,[320,240]),[320,240],64)
    renamed=copy.deepcopy(tree);renamed['nodes'][1]['id']='other';renamed_boxes={'page':boxes['page'],'other':boxes['card']}
    b=abstract_image(elements(renamed,renamed_boxes,[320,240]),[320,240],64)
    assert np.array_equal(a,b)
    boxes['card'][0]+=20
    c=abstract_image(elements(tree,boxes,[320,240]),[320,240],64)
    assert not np.array_equal(a,c)


def test_semantic_masks_preserve_overlaps_small_objects_and_letterbox():
    from framediff.visual import semantic_masks,observation_mae
    items=[{'label':1,'box':[10,10,40,40]}, {'label':4,'box':[0,0,80,50]},
           {'label':3,'box':[90,20,90.5,20.5]}]
    masks=semantic_masks(items,[100,50],32)
    assert masks.shape==(8,32,32) and masks.min()>=0 and masks.max()<=1
    assert masks[0,5,5]==masks[3,5,5]==1
    assert masks[2].sum()>0 and masks[2].sum()<1
    assert masks[:,16:,:].sum()==0
    assert torch.equal(masks,semantic_masks(list(reversed(items)),[100,50],32))
    assert observation_mae(masks[None],masks[None],True)==0
    assert observation_mae(masks[None],torch.zeros_like(masks)[None],True)>0


def test_semantic_policy_ce_gradients_and_checkpoint(tmp_path):
    from framediff.visual import semantic_masks
    torch.set_num_threads(2)
    cfg=VisualConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,numeric_only=True,
        policy_head='autoregressive',training_scheme='balanced-v1',
        observation_contract='semantic-mask-pair-diff-v1')
    net=VisualPolicy(cfg,False)
    tree,boxes=fixture_tree()
    batch=visual_batch([current_features(tree,boxes,[320,240],16)],'cpu')
    target=semantic_masks([{'label':3,'box':[100,20,180,50]}],[320,240],32,boundaries=False)[None].requires_grad_()
    current=semantic_masks(elements(tree,boxes,[320,240]),[320,240],32,boundaries=False)[None].requires_grad_()
    logits,parts=net(batch,target,current,True)
    assert net.vision.stem[0].in_channels==12 and not hasattr(net,'operation')
    assert torch.isneginf(logits[0,-1])
    loss=-(parts['node'][0,1]+parts['property'][0,1,0]+parts['delta'][0,1,0,9])/3
    loss.backward()
    assert torch.isfinite(loss) and target.grad.abs().sum()>0 and current.grad.abs().sum()>0
    path=tmp_path/'semantic.pt'
    torch.save({'kind':'visual-policy-v6-semantic','config':asdict(cfg),'model':net.state_dict()},path)
    restored,_=load_policy(path,'cpu')
    assert torch.allclose(logits,restored(batch,target,current))


@pytest.mark.browser
def test_new_modalities_rollout_capture_counts(tmp_path,monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff.visual_experiment import rollout
    target=tmp_path/'target.png';calls=[]
    with HtmlBrowser() as browser:
        snapshot=browser.snapshot(HTML,[320,240],target,max_nodes=16)
        tree,boxes,_=dom_tree(snapshot);annotate(browser,tree)
        items=elements(tree,boxes,[320,240])
        def fake_detect(*args):calls.append(1);return items
        monkeypatch.setattr('framediff.visual_experiment.detect',fake_detect)
        for mode in ('screenshot','abstract'):
            cfg=VisualConfig(mode=mode,size=32,hidden=16,layers=1,heads=2,token_grid=2,numeric_only=True,
                policy_head='autoregressive',training_scheme='balanced-v1',observation_contract=(
                    'semantic-mask-pair-diff-v1' if mode=='abstract' else 'rgb-pair-diff-v1'))
            policy=VisualPolicy(cfg,False).eval()
            _,stats,_=rollout(browser,snapshot['html'],tree,[320,240],str(target),policy,
                parser=object(),steps=2)
            assert stats['actions']==2 and stats['stop_reason']=='steps'
            assert stats['pair_image_encodings']==2 and stats['dom_queries']==2
            assert stats['browser_screenshots']==(2 if mode=='screenshot' else 0)
            assert stats['abstraction_calls']==(0 if mode=='screenshot' else 1)
        assert len(calls)==1


def test_structured_action_contract_is_single_and_normalized():
    assert validate_action((1,'margin-left',.05))==(1,'margin-left',.05)
    assert validate_action((1,'justify-content','space-between'))==(1,'justify-content','space-between')
    assert action_size((1,'margin-left',.05))=={'declarations':1,'normalized_magnitude':.05}
    assert action_size((1,'align-items','center'))=={'declarations':1,'normalized_magnitude':None}
    assert ACTIONS==81
    with pytest.raises(ValueError,match='grammar'):validate_action((1,'padding-left',.05))
    with pytest.raises(ValueError,match='grammar'):validate_action((1,'column-gap',.05))
    with pytest.raises(ValueError,match='grammar'):validate_action((1,'unknown',.01))


def test_teacher_action_legality_matches_policy_mask():
    tree,boxes=fixture_tree();viewport=[320,240]
    assert action_is_legal(tree,boxes,viewport,(1,'width',-.05))
    tiny={**boxes,'card':[16,20,4,30]}
    assert not action_is_legal(tree,tiny,viewport,(1,'width',-.05))
    assert action_is_legal(tree,tiny,viewport,(1,'width',.05))
    feature=current_features(tree,tiny,viewport,16)
    invalid=action_index((1,'width',-.05),len(tree['nodes']))
    assert not feature['action_legal'].flatten()[invalid]


def test_policy_legality_filter_reports_bad_teacher_rows():
    from framediff.visual_train import filter_legal_policy_rows
    tree,boxes=fixture_tree();base={'current':tree,'current_boxes':boxes,'viewport':[320,240]}
    rows=[{**base,'id':'clean','teacher_edits':[]},
          {**base,'id':'valid','teacher_edits':[[1,'width',.05]]},
          {**base,'id':'invalid','current_boxes':{**boxes,'card':[16,20,4,30]},
           'teacher_edits':[[1,'width',-.05]]}]
    kept,rejected=filter_legal_policy_rows(rows)
    assert [r['id'] for r in kept]==['clean','valid']
    assert rejected==[{'id':'invalid','edits':[[1,'width',-.05]]}]


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


@pytest.mark.browser
def test_html_browser_recovers_context_and_disables_meta_refresh(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    target=tmp_path/'page.png'
    html='''<html><head><meta http-equiv="refresh" content="0; url=https://invalid.example"></head>
    <body><button>safe</button></body></html>'''
    with HtmlBrowser() as browser:
        first=browser.snapshot(html,[160,100],target,max_nodes=8)
        assert any(n['role']=='button' for n in first['nodes'])
        browser.page.close();browser.reset_context()
        second=browser.snapshot('<html><body><p>again</p></body></html>',[160,100],max_nodes=8)
        assert any(n['role']=='p' for n in second['nodes'])


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
        select = VisualPolicy.select
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
    assert len(train)>=2 and 'teacher_edits' not in train[0]
    assert train[1]['teacher_strategy']=='unlabeled-css-corruption-v1'
    assert validate_action(train[1]['corruption_edit'])
    main(['visual-tree-prepare','--rendered',str(tmp_path/'data'),'--out',str(tmp_path/'tree')])
    common=['--train',str(tmp_path/'tree/policy-train.jsonl'),'--val',str(tmp_path/'tree/policy-val.jsonl'),
            '--out',str(tmp_path/'policy'),'--mode','abstract','--size','64','--hidden','16','--layers','1',
            '--heads','2','--max-nodes','16','--token-grid','3','--device','cpu','--cpu-threads','2',
            '--batch-size','1','--accumulation','1','--val-samples','1','--eval-every','1','--no-pretrained']
    main(['visual-tree-train',*common,'--steps','1'])
    policy_log=list(read_jsonl(tmp_path/'policy/validation.jsonl'))
    assert policy_log[-1]['validation_n']==1 and 'action_accuracy' not in policy_log[-1]
    main(['visual-tree-train',*common,'--steps','2','--resume',str(tmp_path/'policy/last.pt')])
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
        main(['visual-cache-targets','--data',str(tmp_path/f'tree/policy-{split}.jsonl'),
              '--checkpoint',str(tmp_path/'detector/best.pt'),'--out',str(tmp_path/f'cached-{split}'),
              '--device','cpu','--cpu-threads','2','--size','64','--threshold','0'])
    fine=list(common)
    for flag,value in [('--train','cached-train/data.jsonl'),('--val','cached-val/data.jsonl'),('--out','fine')]:
        fine[fine.index(flag)+1]=str(tmp_path/value)
    main(['visual-tree-train',*fine,'--steps','1','--init-checkpoint',str(tmp_path/'policy/best.pt'),
          '--predicted-targets'])
    raw=list(common);raw[raw.index('--mode')+1]='screenshot';raw[raw.index('--out')+1]=str(tmp_path/'raw')
    main(['visual-tree-train',*raw,'--steps','1'])
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
            {'step':50,'validation_loss':3.5,'best':3.5,'improving_action_rate':.4,'node_accuracy':.7,
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

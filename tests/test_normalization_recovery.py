from types import SimpleNamespace

import pytest

from framediff.ir import read_json, read_jsonl, write_jsonl
from framediff.visual_data import build
from framediff.tree_normalize import prepare
from framediff.tree_freeze import freeze


def test_changed_normalization_keeps_original_goal_and_learns_repair(tmp_path, monkeypatch):
    from framediff import normalization_recovery as recovery
    from framediff import explicit_html
    from framediff.tree_online import OnlineSampler,target_pool,abstraction
    from framediff.html_bridge import HtmlBrowser
    from framediff import css_owners
    from framediff.visual import semantic_masks
    manifest=tmp_path/'source.jsonl';sources=[]
    for split in ('train','val','test'):
        path=tmp_path/f'{split}.html'
        path.write_text(f'<html><body style="margin:0"><div style="width:120px;height:50px;background:red">{split}</div></body></html>')
        sources.append(dict(id=split,group=split,split=split,html=str(path),viewport=[300,200]))
    write_jsonl(manifest,sources)
    rendered=tmp_path/'rendered'
    build(SimpleNamespace(manifest=str(manifest),out=str(rendered),resume=True,seed=42,max_nodes=16,
          abstract_size=64,min_elements=1,max_source_mae=1.,clean_only=True,trajectories=1,max_noise=1))
    original=explicit_html.convert
    def damaged(browser,*args,**kwargs):
        report=original(browser,*args,**kwargs)
        folder=args[2]
        path=folder/'normalized.html'
        assert report['accepted']
        browser.load(path.read_text(),[300,200])
        browser.page.evaluate("document.querySelector('div').style.setProperty('width','60px','important')")
        (folder/'rejected.html').write_text(browser.page.content())
        return {**report,'accepted':False,'failure_stage':'normalization_preservation'}
    monkeypatch.setattr(explicit_html,'convert',damaged)
    labels=tmp_path/'labels'
    args=SimpleNamespace(rendered=str(rendered),out=str(labels),resume=True,seed=42,max_css_owners=64,
          observation_size=64,max_pixel_mae=.01,max_box_error=1.,recover_normalization=True,
          recovery_fit_steps=3,reuse_normalized=None)
    prepare(args)
    row=list(read_jsonl(labels/'policy-train.jsonl'))[0]
    reference=list(read_jsonl(rendered/'policy-train.jsonl'))[0]
    assert row['target_image']==reference['target_image']
    assert row['target_elements']==reference['target_elements']
    assert row['recovery_report']['edits']
    assert row['recovery_report']['final_distance']<row['recovery_report']['initial_distance']
    pool=target_pool([{**row,'split':'train'}],observation_size=64)
    assert pool[0]['target_elements']==reference['target_elements']
    assert pool[0]['target_declaration_state']==row['target_declaration_state']
    sampler=OnlineSampler(pool,'screenshot',seed=42,max_noise=2)
    try:
        sample=sampler.generate(0)
        assert sample['online']['corruptions']==[]
        sampler.close()
        with HtmlBrowser() as browser:
            browser.load(sample['current_html_text'],sample['viewport'])
            before,_,_=abstraction(browser,sample['current'],sample['viewport'],64)
            target=semantic_masks(sample['target_elements'],sample['viewport'],64)
            css_owners.execute(browser,sample['css_owners'],sample['replacement_edit'])
            after,_,_=abstraction(browser,sample['current'],sample['viewport'],64)
            assert (after-target).abs().mean()<(before-target).abs().mean()
    finally:sampler.close()
    with_combined=OnlineSampler(pool,'screenshot',seed=42,max_noise=2)
    try:
        combined=with_combined.generate(1)
        assert combined['online']['corruptions']
        assert combined['target_image']==reference['target_image']
        assert combined['online']['recovery_teacher_gain']>0
    finally:with_combined.close()
    # Both RGB and abstract CE training consume the original target, including
    # stage-2 cached detector conditioning; teacher state is not an input.
    import torch
    from framediff.tree_policy import TreeConfig,TreePolicy,batch_inputs
    from framediff.ir import write_json
    torch.set_num_threads(2)
    prediction=tmp_path/'prediction.json';write_json(prediction,row['target_elements'])
    row['predicted_target_elements']=str(prediction)
    for mode,predicted in [('screenshot',False),('abstract',False),('abstract',True)]:
        cfg=TreeConfig(mode=mode,size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=16,
                       stylesheets=True,action_contract=css_owners.CONTRACT)
        net=TreePolicy(cfg,False)
        batch,target,current,states=batch_inputs(net,[row],'cpu',predicted)
        loss,count=net.loss(batch,target,current,states,[row['replacement_edit']])
        assert torch.isfinite(loss) and count>0
        loss.backward()
        assert net.output.weight.grad.abs().sum()>0
    saved=(labels/'policy-train.jsonl').read_bytes()
    monkeypatch.setattr(explicit_html,'convert',lambda *a,**k:pytest.fail('should reuse page cache'))
    prepare(args)
    assert saved==(labels/'policy-train.jsonl').read_bytes()
    # An old accepted-normalization cache is read-only reusable. The recovery
    # labels are regenerated against the original goal, not copied blindly.
    from framediff.ir import write_json
    from framediff.web_experiment import digest
    before_assets={}
    for folder in (labels/'normalized-pages').glob('*/*'):
        record=read_json(folder/'record.json')
        write_json(folder/'normalization.json',{'source_sha':record['signature']['html'],
                   'sha':digest(folder/'normalized.html')})
        before_assets[str(folder/'normalized.html')]=digest(folder/'normalized.html')
    migrated=SimpleNamespace(**{**vars(args),'out':str(tmp_path/'migrated'),'reuse_normalized':str(labels)})
    prepare(migrated)
    assert list(read_jsonl(tmp_path/'migrated/policy-train.jsonl'))[0]['normalization_stats']['reused_normalization']
    assert all(digest(path)==sha for path,sha in before_assets.items())
    out=tmp_path/'frozen'
    freeze(SimpleNamespace(rendered=str(labels),out=str(out),resume=True,workers=1,seed=42,
           samples_per_page=1,max_noise=2,observation_size=64))
    test=list(read_jsonl(out/'policy-test.jsonl'))[0]
    test_reference=list(read_jsonl(rendered/'policy-test.jsonl'))[0]
    page=list(read_jsonl(out/'pages-test.jsonl'))[0]
    assert test['target_image']==test_reference['target_image']
    assert page['evaluation_target_boxes']==test_reference['current_boxes']


def test_lazy_train_carries_recovery_configuration(tmp_path):
    from framediff.tree_normalize import _prepare_page
    row={'id':'a/clean','lazy_contract':'lazy-webui-train-view-v1','lazy_settings':{}}
    args=SimpleNamespace(max_css_owners=64,observation_size=64,max_pixel_mae=.01,
                         max_box_error=1.,recover_normalization=True,recovery_fit_steps=3)
    new,_,error=_prepare_page((args,tmp_path,'train',0,'a',row,{'a':{'id':'a'}}))
    assert error is None
    assert new['lazy_settings']['recover_normalization']
    assert new['lazy_settings']['recovery_fit_steps']==3

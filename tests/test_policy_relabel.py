from pathlib import Path
from types import SimpleNamespace
import copy
import json

import pytest

from framediff import css_owners
from framediff.html_bridge import HtmlBrowser
from framediff.plans import dom_tree
from framediff.visual import annotate, VISUAL_GEOMETRY_CONTRACT
from framediff.policy_relabel import refresh_row, relabel
from framediff.ir import write_jsonl, read_jsonl


def test_live_failure_reporting(tmp_path,monkeypatch,capsys):
    from framediff import policy_relabel as module
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out'
    for split in ('train','val','test'):
        write_jsonl(source/f'policy-{split}.jsonl',[{'id':split+'/bad'},{'id':split+'/good'}])
        write_jsonl(source/f'pages-{split}.jsonl',[])
    def mapped(fn,tasks,workers):
        tasks=list(tasks);split=tasks[0][0]['id'].split('/')[0]
        error={'sample_id':split+'/bad','reason':'screenshot_mismatch','error':'example'}
        yield None,error
        # Verify persistence and flushed output before the next result arrives.
        assert list(read_jsonl(out/f'rejections-{split}.jsonl'))==[error]
        state=json.loads((out/f'relabel-progress-{split}.json').read_text())
        assert state['processed']==1 and state['failure_reasons']=={'screenshot_mismatch':1}
        assert 'relabel_failure' in capsys.readouterr().out
        yield {'id':split+'/good'},None
    monkeypatch.setattr(module,'page_map',mapped)
    report=relabel(SimpleNamespace(rendered=source,out=out,workers=1,observation_size=64,resume=True))
    assert all(r['failure_reasons']=={'screenshot_mismatch':1} for r in report.values())
    assert json.loads((out/'relabel-progress-test.json').read_text())['status']=='complete'


@pytest.mark.parametrize('error,reason',[
    (ValueError('target: saved screenshot differs from measured HTML'),'screenshot_mismatch'),
    (ValueError('Existing corruption has no new abstraction effect'),'corruption_no_visual_effect'),
    (ValueError('Existing teacher has no new abstraction effect'),'teacher_no_visual_effect'),
    (ValueError('Render stabilization timed out'),'render_unstable'),
    (FileNotFoundError('missing'),'missing_asset'),
    (RuntimeError('unknown'),'other'),
])
def test_failure_categories(error,reason):
    from framediff.policy_relabel import failure_reason
    assert failure_reason(error)==reason


def fixture(tmp_path):
    with HtmlBrowser() as b:
        dom=b.snapshot('<body style="margin:0"><div class="card" style="width:200px;height:50px;margin-left:10px;background:red">Hello</div></body>',[400,200],max_nodes=32)
        tree,boxes,_=dom_tree(dom);annotate(b,tree)
        target=b.page.content();parsed=css_owners.read(b,tree)
        nid=b.page.locator('.card').get_attribute('data-fd-id')
        owner=next(i for i,o in enumerate(parsed['owners']) if o.get('id')==nid)
        target_png,_=b.stable_capture()
        css_owners.execute(b,parsed['owners'],[owner,'margin-left','60px',''])
        current=b.page.content();state=css_owners.read(b,tree)['state']
        current_png,_=b.stable_capture()
    row=dict(id='page/online-0',group='page',source_sha='page',split='train',viewport=[400,200],
        current=tree,current_boxes=boxes,css_owners=parsed['owners'],declaration_state=state,
        target_declaration_state=parsed['state'],replacement_edit=[owner,'margin-left','10px',''],
        teacher_strategy=css_owners.CONTRACT,symbolic_distance=1)
    for key,value in [('target_html',target),('current_html',current),('target_image',target_png),('current_image',current_png)]:
        path=tmp_path/(key+('.html' if isinstance(value,str) else '.png'))
        path.write_bytes(value.encode() if isinstance(value,str) else value);row[key]=str(path)
    return row


def test_relabel_preserves_assets_teacher_and_both_training_modes(tmp_path,monkeypatch):
    row=fixture(tmp_path);before=copy.deepcopy(row)
    assets={row[k]:Path(row[k]).read_bytes() for k in ('target_html','current_html','target_image','current_image')}
    updated=refresh_row(row,tmp_path/'new',64)
    assert row==before and updated['replacement_edit']==row['replacement_edit']
    assert updated['visual_geometry_contract']==VISUAL_GEOMETRY_CONTRACT
    assert any(n.get('visual_boxes') for n in updated['current']['nodes'])
    for path,data in assets.items():assert Path(path).read_bytes()==data
    for key in ('target_html','current_html','target_image','current_image'):assert updated[key]==row[key]
    import torch
    from framediff.tree_policy import TreePolicy,TreeConfig,batch_inputs
    torch.set_num_threads(1)
    for mode in ('abstract','screenshot'):
        net=TreePolicy(TreeConfig(mode=mode,size=64,hidden=32,layers=1,heads=4,max_nodes=64,
            stylesheets=True,action_contract=css_owners.CONTRACT,visual_geometry_contract=VISUAL_GEOMETRY_CONTRACT),pretrained=False)
        loss,_=net.loss(*batch_inputs(net,[updated],'cpu'),[updated['replacement_edit']])
        assert torch.isfinite(loss);loss.backward()
    wrong={**row,'current_image':row['target_image']}
    with pytest.raises(ValueError,match='saved screenshot differs'):
        refresh_row(wrong,tmp_path/'bad',64)


def test_manifest_relabel_resume_and_failure_reporting(tmp_path,monkeypatch):
    row=fixture(tmp_path);source=tmp_path/'source';source.mkdir()
    for split in ('train','val','test'):
        sample={**row,'id':split+'/online-0','group':split,'source_sha':split,'split':split}
        bad={**sample,'id':split+'bad/online-0','current_image':row['target_image']}
        write_jsonl(source/f'policy-{split}.jsonl',[sample,bad])
        write_jsonl(source/f'pages-{split}.jsonl',[{'id':split,'split':split,'html':row['target_html']},
                                                  {'id':split+'bad','split':split}])
    args=SimpleNamespace(rendered=str(source),out=str(tmp_path/'out'),workers=2,observation_size=64,resume=True)
    report=relabel(args)
    assert all(r['kept']==1 and r['failed']==1 for r in report.values())
    from framediff import policy_relabel as module
    original=module.refresh_row
    def guarded(row,*a,**kw):
        assert 'bad/' in row['id'], 'successful row was recomputed'
        return original(row,*a,**kw)
    monkeypatch.setattr(module,'refresh_row',guarded);args.workers=1
    assert relabel(args)==report
    for split in ('train','val','test'):
        saved=list(read_jsonl(Path(args.out)/f'policy-{split}.jsonl'))
        assert saved[0]['visual_geometry_contract']==VISUAL_GEOMETRY_CONTRACT


def test_lazy_reuses_existing_normalization_without_render_or_corruption(tmp_path,monkeypatch):
    import json
    from framediff.lazy_webui import TrainViewCache, CONTRACT, DISK_CACHE_CONTRACT
    from framediff.policy_relabel import _task
    from framediff.web_experiment import digest
    row=fixture(tmp_path)
    source=dict(id='page',split='train',html=row['target_html'],source_sha=digest(row['target_html']))
    descriptor=dict(id='page/clean',lazy_contract=CONTRACT,lazy_source=source,lazy_settings={},split='train')
    monkeypatch.setenv('TUIDE_SHARED_CACHE',str(tmp_path/'shared'))
    cache=TrainViewCache(cache_dir=tmp_path/'legacy')
    folder=cache.cache_dir/cache._key(descriptor,True);folder.mkdir()
    assets={row[k]:digest(row[k]) for k in ('target_html','current_html','target_image','current_image')}
    (folder/'result.json').write_text(json.dumps({'contract':DISK_CACHE_CONTRACT,'result':row,'assets':assets}))
    def forbidden(*a,**kw):raise AssertionError('old cached HTML was regenerated')
    monkeypatch.setattr('framediff.visual_data._render_page',forbidden)
    upgraded,error=_task((descriptor,tmp_path/'new',64,True,tmp_path/'legacy'))
    assert error is None
    result=cache.get(upgraded,policy=True)
    assert result['visual_geometry_contract']==VISUAL_GEOMETRY_CONTRACT
    assert 'current_image' not in result and 'current_html' not in result
    assert Path(row['current_image']).is_file()  # Published legacy cache is untouched.
    assert result['replacement_edit']==row['replacement_edit']
    cache.close()
    monkeypatch.setattr('framediff.policy_relabel.refresh_row',forbidden)
    cache=TrainViewCache(cache_dir=tmp_path/'legacy')
    assert cache.get(upgraded,policy=True)==result
    cache.close()


def test_rebuild_heldout_reuses_clean_assets_and_new_teachers(tmp_path,monkeypatch):
    from framediff import policy_relabel as module
    from framediff.tree_freeze import freeze
    from framediff.tree_edits import repair_path
    row=fixture(tmp_path);source=tmp_path/'source';source.mkdir()
    original={row[k]:Path(row[k]).read_bytes() for k in ('target_html','target_image')}
    for split in ('train','val','test'):
        sample={**row,'id':split+'/seed','group':split,'split':split}
        if split!='train':
            sample.update(current_html='/missing/obsolete.html',current_image='/missing/obsolete.png',replacement_edit=None)
        write_jsonl(source/f'policy-{split}.jsonl',[sample])
        write_jsonl(source/f'pages-{split}.jsonl',[{'id':split,'split':split,'html':row['target_html']}])
    args=SimpleNamespace(rendered=source,out=tmp_path/'labels',workers=1,observation_size=64,resume=True,clean_heldout=True)
    report=relabel(args)
    assert all(r['kept']==1 and r['failed']==0 for r in report.values())
    def forbidden(*a,**kw):raise AssertionError('successful clean cache remeasured')
    monkeypatch.setattr(module,'refresh_row',forbidden)
    assert relabel(args)==report
    options=SimpleNamespace(rendered=args.out,out=tmp_path/'css',workers=1,observation_size=64,
        resume=True,seed=90210,max_noise=1,samples_per_page=1)
    freeze(options)
    import torch
    from framediff.tree_policy import TreePolicy,TreeConfig,batch_inputs
    torch.set_num_threads(1)
    for split in ('val','test'):
        rows=list(read_jsonl(options.out/f'policy-{split}.jsonl'));assert len(rows)==1
        sample=rows[0]
        assert sample['replacement_edit'] in repair_path(sample['declaration_state'],sample['target_declaration_state'])
        assert sample['visual_geometry_contract']==VISUAL_GEOMETRY_CONTRACT
        assert Path(sample['current_html']).is_file() and Path(sample['current_image']).is_file()
        for mode in ('abstract','screenshot'):
            net=TreePolicy(TreeConfig(mode=mode,size=64,hidden=32,layers=1,heads=4,max_nodes=64,
                stylesheets=True,action_contract=css_owners.CONTRACT,visual_geometry_contract=VISUAL_GEOMETRY_CONTRACT),pretrained=False)
            loss,_=net.loss(*batch_inputs(net,rows,'cpu'),[sample['replacement_edit']])
            assert torch.isfinite(loss)
    before={p:p.read_bytes() for p in options.out.rglob('*') if p.is_file()}
    monkeypatch.setattr('framediff.tree_freeze.OnlineSampler',forbidden)
    freeze(options)
    assert all(p.read_bytes()==data for p,data in before.items())
    assert all(Path(p).read_bytes()==data for p,data in original.items())

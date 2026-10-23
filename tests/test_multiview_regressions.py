import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from framediff.ir import read_jsonl, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[1]


def test_shell_view_names_preserve_spaces():
    command='source scripts/webui_views.sh; printf "%s\\n" "${views[@]}"'
    env={k:v for k,v in os.environ.items() if k not in ('WEBUI_VIEW','WEBUI_VIEWS')}
    result=subprocess.run(['bash','-c',command],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    assert len(result.stdout.splitlines())==6
    assert result.stdout.splitlines()[-1]=='iPhone-13 Pro'
    env['WEBUI_VIEWS']='default_1280-720,iPhone-13 Pro'
    result=subprocess.run(['bash','-c',command],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    assert result.stdout.splitlines()==['default_1280-720','iPhone-13 Pro']
    env['WEBUI_VIEWS']='default_1280-720 iPhone-13 Pro'
    assert subprocess.run(['bash','-c',command],cwd=ROOT,env=env,capture_output=True).returncode==2


def test_preindexed_sampler_and_unlimited_validation():
    from framediff.visual_train import page_view_pools,sample_page_views,validation_sample
    rows=[dict(id=f'p{i}/v{j}',page_id=f'p{i}',source_sha=str(i)) for i in range(1001) for j in range(3)]
    pools=page_view_pools(iter(rows))
    assert len(pools)==1001
    assert sample_page_views(pools,8,random.Random(42))==sample_page_views(pools,8,random.Random(42))
    assert validation_sample(rows,0)==rows
    assert len(validation_sample(rows,2000))==2000
    with pytest.raises(ValueError):validation_sample(rows,-1)


def test_coverage_unique_pages_not_views_and_partial_views_not_success(tmp_path):
    from framediff.fixed_test import coverage_provenance
    labels=tmp_path/'labels';css=tmp_path/'css'
    write_json(labels/'prepare-report.json',{'test':dict(selected_pages=1,selected_views=2,
        source_pages=1,source_views=2,clean_pages=1,clean_views=2,kept=1,kept_views=2,rejected=0,rejected_views=0)})
    write_json(css/'config.json',{'data':{str(labels/'policy-test.jsonl'):'unused'}})
    write_json(css/'freeze-report.json',{'test':{'pages':1,'page_views':2}})
    rows=[dict(id=f'p/{v}/online-0',page_id='p') for v in ('desktop','mobile')]
    write_jsonl(tmp_path/'source/manifest-test.jsonl',[dict(id=f'p/{v}',page_id='p') for v in ('desktop','mobile')])
    results=[dict(method='abstract',page_id=f'p/{v}',logical_page_id='p',failed=False) for v in ('desktop','mobile')]
    report=coverage_provenance(css/'policy-test.jsonl',rows,results,['abstract'])
    assert report['methods']['abstract']['successful_pages']==1
    assert report['methods']['abstract']['end_to_end_success_rate']==1
    assert report['page_views']['evaluated']==2
    partial=coverage_provenance(css/'policy-test.jsonl',rows[:1],results[:1],['abstract'])
    assert partial['methods']['abstract']['successful_pages']==0


def lazy_descriptor(tmp_path):
    from framediff.visual_data import build
    from framediff.tree_normalize import _prepare_page
    html=tmp_path/'page.html'
    html.write_text('<html><body style="margin:0"><div style="width:100px;height:40px;background:red">Test</div></body></html>')
    manifest=tmp_path/'manifest.jsonl'
    write_jsonl(manifest,[dict(id='page/desktop',page_id='page',view='desktop',group='example',split='train',html=str(html),viewport=[400,200])])
    out=tmp_path/'rendered'
    args=SimpleNamespace(manifest=str(manifest),out=str(out),resume=True,trajectories=1,max_noise=1,
        seed=42,max_nodes=32,abstract_size=64,min_elements=1,max_source_mae=1.,clean_only=True,lazy_train=True)
    build(args)
    assert not list(out.rglob('*.png'))
    args=SimpleNamespace(resume=True,seed=42,max_css_owners=32,observation_size=64,max_pixel_mae=.01,max_box_error=1.)
    clean=list(read_jsonl(out/'policy-train.jsonl'))[0]
    page=list(read_jsonl(out/'pages-train.jsonl'))[0]
    row,_,error=_prepare_page((args,tmp_path/'labels','train',0,page['id'],clean,{page['id']:page}))
    assert error is None
    assert not list((tmp_path/'labels').rglob('*.png'))
    return row


def test_lazy_preparation_does_not_render_train(tmp_path):
    row=lazy_descriptor(tmp_path)
    assert row['lazy_source']['viewport']==[400,200]
    assert row['normalization_contract']


def test_lazy_cache_invalidates_when_preparation_changes(tmp_path,monkeypatch):
    import framediff.lazy_webui as lazy
    row=lazy_descriptor(tmp_path);cache=lazy.TrainViewCache()
    before=cache._key(row,True)
    monkeypatch.setattr(lazy,'preparation_fingerprint',lambda:{'version':'changed'})
    assert cache._key(row,True)!=before


def test_compact_unpublished_cache_preserves_inputs_and_links_duplicates(tmp_path):
    from framediff.lazy_webui import compact_new_cache
    work=tmp_path/'stage';work.mkdir()
    paths={name:work/name for name in ('target.png','current.png','current.html','unused.png','report.json')}
    for name,path in paths.items():path.write_text('pixels' if name.endswith('.png') else name)
    result={'target_image':str(paths['target.png']),'current_image':str(paths['current.png']),
            'current_html':str(paths['current.html'])}
    compact_new_cache(work,result)
    assert not paths['unused.png'].exists()
    assert all(Path(path).is_file() for path in result.values())
    assert paths['report.json'].is_file()
    assert os.path.samefile(paths['target.png'],paths['current.png'])


def test_stream_only_changes_do_not_invalidate_preparation(monkeypatch):
    from framediff.lazy_webui import preparation_fingerprint
    original=Path.read_text
    preparation_fingerprint.cache_clear()
    before=preparation_fingerprint()
    def altered(path,*args,**kwargs):
        text=original(path,*args,**kwargs)
        return text.replace('self.timeout_skips=0','self.timeout_skips=42') if path.name=='tree_online.py' else text
    try:
        monkeypatch.setattr(Path,'read_text',altered)
        preparation_fingerprint.cache_clear()
        assert preparation_fingerprint()==before
    finally:preparation_fingerprint.cache_clear()


def test_sampler_memory_fix_preserves_existing_cache_fingerprint():
    from framediff.lazy_webui import preparation_fingerprint
    assert preparation_fingerprint()['online_preparation']=='e598fecb8a9bb423bba5e9cf118b338f89be0bbe3d83195d5512e4e7c2fbf53d'
    assert preparation_fingerprint()['tree_normalize.py']=='99abfa568147b5581248b90f477b920c26202fd5dffa02f9059135ea5eb5cc7f'


@pytest.mark.browser
def test_corrupt_lazy_cache_is_rebuilt(tmp_path):
    from framediff.lazy_webui import TrainViewCache
    row=lazy_descriptor(tmp_path);root=tmp_path/'persistent'
    cache=TrainViewCache(cache_dir=root)
    try:first=cache.get(row,policy=True)
    finally:cache.close()
    stale=root/f'.{cache._key(row,True)}.interrupted';stale.mkdir()
    (stale/'partial').write_text('interrupted writer')
    Path(first['target_image']).write_bytes(b'corrupt')
    cache=TrainViewCache(cache_dir=root)
    try:
        second=cache.get(row,policy=True)
        assert Path(second['target_image']).read_bytes().startswith(b'\x89PNG')
        assert not stale.exists()
    finally:cache.close()


def shared_lazy_cache_task(task):
    import framediff.visual_data as data
    from framediff.lazy_webui import TrainViewCache
    row,root,markers=task
    original=data._render_page
    def counted(*args,**kwargs):
        Path(markers,f'{os.getpid()}.render').touch()
        return original(*args,**kwargs)
    data._render_page=counted
    cache=TrainViewCache(cache_dir=root)
    try:return cache.get(row,policy=True)['target_image']
    finally:cache.close();data._render_page=original


@pytest.mark.browser
def test_concurrent_lazy_cache_materializes_once(tmp_path):
    from framediff.preparation_workers import page_map
    row=lazy_descriptor(tmp_path);root=tmp_path/'shared'
    task=(row,str(root),str(tmp_path))
    results=list(page_map(shared_lazy_cache_task,[task,task],2))
    assert results[0]==results[1]
    assert len(list(tmp_path.glob('*.render')))==1


def test_detector_prefetch_resume_preserves_consumed_rng(tmp_path):
    from framediff.detector_support import DetectorStream
    from framediff.visual_train import page_view_pools
    pools=page_view_pools([{'id':f'p{i}/v{v}','page_id':f'p{i}'} for i in range(4) for v in range(2)])
    rng=random.Random(42)
    with DetectorStream(pools,rng,2,tmp_path/'cache',tmp_path/'errors',workers=2,prefetch=4) as stream:
        stream.take();saved=rng.getstate()
        expected=stream.take();final=rng.getstate()
    resumed=random.Random();resumed.setstate(saved)
    with DetectorStream(pools,resumed,2,tmp_path/'cache',tmp_path/'errors',workers=1,prefetch=1) as stream:
        assert stream.take()==expected
    assert resumed.getstate()==final


@pytest.mark.browser
def test_lazy_online_worker_reuses_browser_across_cache_hits(tmp_path,monkeypatch):
    from framediff.browser import enable_worker_browser,close_worker_browser
    from framediff.tree_online import OnlineSampler,target_pool
    monkeypatch.setenv('ONLINE_PROFILE','1')
    row=lazy_descriptor(tmp_path)
    enable_worker_browser()
    sampler=OnlineSampler(target_pool([row],cache_dir=tmp_path/'cache'),'screenshot',max_noise=1)
    try:
        first=sampler.generate(0)
        browser=sampler.browser.browser
        second=sampler.generate(1)
        assert sampler.browser.browser is browser and browser.is_connected()
        assert first['replacement_edit'] and second['replacement_edit']
        assert first['online']['cache_profile']['status']=='disk_miss'
        assert second['online']['cache_profile']['status']=='disk_hit'
    finally:
        sampler.close();sampler.view_cache.close();close_worker_browser()
    assert not browser.is_connected()


@pytest.mark.browser
def test_lazy_policy_disk_cache_survives_process_cache(tmp_path,monkeypatch):
    from framediff.lazy_webui import TrainViewCache
    row=lazy_descriptor(tmp_path);root=tmp_path/'persistent-lazy-cache'
    first=TrainViewCache(capacity=1,cache_dir=root)
    try:
        prepared=first.get(row,policy=True)
        assert first.last_profile['status']=='disk_miss'
        assert first.last_profile['render_s']>0
        assert first.last_profile['normalize_s']>0
    finally:first.close()
    assert prepared['visible_candidates'] and prepared['target_declaration_state']
    assert Path(prepared['target_html']).is_file() and Path(prepared['target_image']).is_file()
    entry=root/first._key(row,True)
    assert not (entry/'rendered').exists()
    assert list(entry.rglob('report.json'))
    assert 'current_html' not in prepared and 'current_image' not in prepared
    assert not list(entry.rglob('initial.html')) and not list(entry.rglob('initial.png'))
    import framediff.visual_data as visual_data
    monkeypatch.setattr(visual_data,'_render_page',lambda *a,**k: (_ for _ in ()).throw(
        AssertionError('persistent lazy cache rerendered the page')))
    second=TrainViewCache(capacity=1,cache_dir=root)
    try:
        assert second.get(row,policy=True)==prepared
        assert second.last_profile['status']=='disk_hit'
        assert second.last_profile['disk_validation_s']>0
    finally:second.close()


@pytest.mark.browser
def test_shared_cache_imports_existing_without_rendering(tmp_path,monkeypatch):
    import shutil
    from framediff.lazy_webui import TrainViewCache
    import framediff.visual_data as data
    row=lazy_descriptor(tmp_path);old=tmp_path/'old-cache'
    # Simulate a published legacy cache retaining the seed assets.
    monkeypatch.setenv('TUIDE_SAVE_DIAGNOSTICS','1')
    first=TrainViewCache(cache_dir=old)
    try:before=first.get(row,policy=True)
    finally:first.close()
    monkeypatch.delenv('TUIDE_SAVE_DIAGNOSTICS')
    monkeypatch.setenv('TUIDE_SHARED_CACHE',str(tmp_path/'shared'))
    monkeypatch.setattr(data,'_render_page',lambda *a,**k: (_ for _ in ()).throw(AssertionError('rerender')))
    shared=TrainViewCache(cache_dir=old)
    try:after=shared.get(row,policy=True)
    finally:shared.close()
    for key in ('target_html','target_image','current_html','current_image'):
        assert os.path.samefile(before[key],after[key])
        assert Path(after[key]).is_relative_to(tmp_path/'shared')
    for key in ('replacement_edit','visible_candidates','target_declaration_state','current'):
        assert before[key]==after[key]
    shutil.rmtree(old)  # Generated test fixture only.
    other=TrainViewCache(cache_dir=tmp_path/'another-scale-cache')
    try:assert other.get(row,policy=True)==after
    finally:other.close()


@pytest.mark.browser
def test_lazy_detector_and_policy_materialization(tmp_path,capsys,monkeypatch):
    from framediff.lazy_webui import TrainViewCache,portable_images
    from framediff.tree_online import target_pool,OnlineSampler
    monkeypatch.setenv('TUIDE_VERBOSE_INSPECTION','1')
    row=lazy_descriptor(tmp_path)
    cache=TrainViewCache(capacity=1)
    try:
        detector=cache.get(row)
        path=Path(detector['image'])
        assert path.is_file() and detector['elements']
        portable=portable_images(detector,('image',))
        assert isinstance(portable['image'],bytes)
        policy=cache.get(row,policy=True)
        assert policy['replacement_edit'] and policy['page_id']=='page'
        assert not path.exists() # eviction only removes this cache's generated files
        assert Path(row['lazy_source']['html']).is_file()
        assert cache.get(row,policy=True)==policy
    finally:cache.close()
    capsys.readouterr()
    sampler=OnlineSampler(target_pool([row]),'screenshot',max_noise=1)
    try:
        result=sampler.generate(0)
        assert isinstance(result['target_image'],bytes)
        assert isinstance(result['current_image'],bytes)
        assert result['replacement_edit'] and result['page_id']=='page'
    finally:
        sampler.close();sampler.view_cache.close()
    # Lazy normalization performs the expensive witness probe once. The
    # online sampler consumes that enriched result without a second probe.
    assert capsys.readouterr().out.count("'online_target_filter'")==1


@pytest.mark.browser
def test_lazy_training_cache_stage2_and_resume(tmp_path,monkeypatch):
    import torch
    from framediff.cli import main
    from framediff.lazy_webui import TrainViewCache
    from framediff.web_experiment import digest
    row=lazy_descriptor(tmp_path)
    # Independent held-out source, eagerly prepared at a different viewport.
    html=tmp_path/'heldout.html'
    html.write_text('<html><body style="margin:0"><div style="width:120px;height:50px;background:blue">Validation</div></body></html>')
    val_source={**row['lazy_source'],'id':'val/mobile','page_id':'val','group':'heldout',
                'view':'mobile','html':str(html),'source_sha':digest(html),'viewport':[220,300]}
    val_descriptor={**row,'lazy_source':val_source}
    cache=TrainViewCache()
    try:
        val_detection={**cache.get(val_descriptor),'split':'val'}
        # This fixture uses a train materializer to construct a fixed held-out
        # sample. Explicitly retain that sample's assets, unlike online targets.
        with monkeypatch.context() as context:
            context.setenv('TUIDE_SAVE_DIAGNOSTICS','1')
            val_policy={**cache.get(val_descriptor,policy=True),'split':'val'}
        write_jsonl(tmp_path/'detector-train.jsonl',[row])
        write_jsonl(tmp_path/'detector-val.jsonl',[val_detection])
        write_jsonl(tmp_path/'policy-train.jsonl',[row])
        write_jsonl(tmp_path/'policy-val.jsonl',[val_policy])
        main(['visual-train-detector','--train',str(tmp_path/'detector-train.jsonl'),
              '--val',str(tmp_path/'detector-val.jsonl'),'--out',str(tmp_path/'detector'),
              '--device','cpu','--cpu-threads','2','--steps','1','--batch-size','1','--accumulation','1',
              '--val-samples','0','--eval-every','1','--min-size','64','--max-size','96',
              '--max-detections','8','--no-pretrained'])
        for split in ('train','val'):
            main(['visual-cache-targets','--data',str(tmp_path/f'policy-{split}.jsonl'),
                  '--checkpoint',str(tmp_path/'detector/best.pt'),'--out',str(tmp_path/f'cached-{split}'),
                  '--device','cpu','--cpu-threads','2','--size','64'])
        assert not list((tmp_path/'cached-train').glob('*.png'))
        assert list(read_jsonl(tmp_path/'cached-train/data.jsonl'))[0]['lazy_detector']
        for mode,predicted in [('screenshot',False),('abstract',False),('abstract',True)]:
            out=tmp_path/f'run-{mode}-{predicted}'
            args=['visual-tree-train','--train',str(tmp_path/('cached-train/data.jsonl' if predicted else 'policy-train.jsonl')),
                  '--val',str(tmp_path/('cached-val/data.jsonl' if predicted else 'policy-val.jsonl')),
                  '--out',str(out),'--mode',mode,'--stylesheets','--online-corruption',
                  '--online-workers','1','--online-prefetch','1','--online-max-noise','1',
                  '--device','cpu','--no-pretrained','--size','32','--hidden','16','--layers','1',
                  '--heads','2','--token-grid','2','--max-nodes','32','--batch-size','1','--accumulation','1',
                  '--eval-every','1','--val-samples','0','--policy-metric-samples','0','--cpu-threads','2']
            if predicted:args.append('--predicted-targets')
            main([*args,'--steps','1'])
            checkpoint=out/'last.pt'
            main([*args,'--steps','2','--resume',str(checkpoint)])
            assert torch.load(checkpoint,weights_only=True)['online_cursor']==2
            assert list(read_jsonl(out/'validation.jsonl'))[-1]['validation_complete']
    finally:cache.close()


@pytest.mark.browser
def test_multiview_prepare_pipeline_resume_and_subset(tmp_path):
    from PIL import Image
    from framediff.cli import main
    from framediff.ir import read_json
    raw=tmp_path/'raw';out=tmp_path/'scale'
    for i in range(12):
        page=raw/str(i);page.mkdir(parents=True)
        for view,viewport in [('default_1280-720',(1280,720)),('iPhone-13 Pro',(390,844))]:
            Image.new('RGB',viewport,'white').save(page/f'{view}-screenshot.png')
            (page/f'{view}-html.html').write_text(f'<html><body style="margin:0"><div style="width:100px;height:40px;background:red">Page {i}</div></body></html>')
            (page/f'{view}-url.txt').write_text(f'https://site{i}.example/')
    main(['visual-webui-scales-prepare','--root',str(raw),'--out',str(out),'--sizes','10',
          '--views','default_1280-720','iPhone-13 Pro','--resume'])
    master=out/'master'
    commands=[['visual-build-data','--manifest',str(master/'source/manifest.jsonl'),
               '--out',str(master/'rendered'),'--clean-only','--lazy-train',
               '--max-nodes','32','--abstract-size','64','--workers','2','--resume'],
              ['visual-tree-normalize','--rendered',str(master/'rendered'),'--out',str(master/'labels'),
               '--observation-size','64','--max-css-owners','32','--workers','2','--resume'],
              ['visual-tree-freeze','--rendered',str(master/'labels'),'--out',str(master/'css'),
               '--observation-size','64','--max-noise','1','--workers','2','--resume'],
              ['visual-webui-scales-subset','--data',str(out)]]
    for command in commands:main(command)
    before=(out/'10/css/policy-test.jsonl').read_bytes()
    for command in commands:main(command)
    assert before==(out/'10/css/policy-test.jsonl').read_bytes()
    train=list(read_jsonl(out/'10/css/policy-train.jsonl'))
    test=list(read_jsonl(out/'10/css/policy-test.jsonl'))
    assert len(train)==16 and all(r.get('lazy_contract') for r in train)
    assert len(test)==2 and all(not r.get('lazy_contract') for r in test)
    assert len({r['page_id'] for r in train})==8
    assert {r['view'] for r in test}=={'default_1280-720','iPhone-13 Pro'}
    assert all(Path(r['current_image']).is_file() for r in test)
    assert read_json(out/'10/coverage.json')['rendering']['train']['status']=='deferred'

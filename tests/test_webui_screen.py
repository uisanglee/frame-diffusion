import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from framediff.ir import read_jsonl
from framediff.preparation_workers import write_jsonl
from framediff.webui_screen import RowIndex,page_inventory,publish,result_record,select_split,screen_view


def test_geometry_retry_preserves_successes_and_refuses_active_or_published_run(tmp_path):
    import fcntl
    import runpy
    requeue=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/retry_screen_geometry_failures.py'))['requeue']
    folder=tmp_path/'screening/train';folder.mkdir(parents=True)
    (folder.parent/'config.json').write_text('{}')
    records=[dict(status='accepted'),dict(status='rejected',error='timeout'),
             dict(status='rejected',error='target: label geometry mismatch (element geometry changed)')]
    for i,r in enumerate(records):(folder/f'{i}.json').write_text(json.dumps(r))
    with (tmp_path/'.screening.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='Stop screening'):requeue(tmp_path)
    result=requeue(tmp_path)
    assert result['requeued']==1
    assert json.loads((Path(result['backup'])/'train/2.json').read_text())==records[2]
    assert not (folder/'2.json').exists()
    assert all(json.loads((folder/f'{i}.json').read_text())==records[i] for i in (0,1))
    assert requeue(tmp_path)['requeued']==0
    (folder/'3.json').write_text(json.dumps(dict(status='rejected',error='Lazy render failed: Only 0 visible abstraction elements')))
    assert requeue(tmp_path)['requeued']==0
    assert requeue(tmp_path,include_visible=True)['requeued']==1
    assert (folder/'0.json').exists() and (folder/'1.json').exists()
    (tmp_path/'READY.json').write_text('{}')
    with pytest.raises(ValueError,match='already published'):requeue(tmp_path)


def test_existing_input_or_unrelated_output_is_not_overwritten(tmp_path):
    from framediff.webui_screen import run
    source=tmp_path/'source';detector=tmp_path/'detector'
    args=SimpleNamespace(data=str(source),detector_rendered=str(detector),out=str(detector))
    with pytest.raises(ValueError,match='separate output'):run(args)
    out=tmp_path/'out';out.mkdir();marker=out/'keep.txt';marker.write_text('keep')
    args.out=str(out)
    with pytest.raises(ValueError,match='unrelated files'):run(args)
    assert marker.read_text()=='keep'


class Job:
    def __init__(self,value=None,active_s=.01,stage='working',timeout=False):
        self.value=value;self.active_s=active_s;self.stage=stage;self.timeout=timeout

    def result(self,timeout):
        if self.timeout:raise TimeoutError('slow')
        return self.value


class Pool:
    def __init__(self):self.calls=[];self.restarted=[]
    def submit(self,function,policy,detector,options):
        self.calls.append(policy['id'])
        return Job(dict(status='accepted'),active_s=200 if policy.get('slow') else .01)
    def restart(self,job,retry):self.restarted.append((job,retry))


def test_timeout_kills_only_its_worker_and_lock_wait_is_not_a_rejection():
    pool=Pool();job=Job(timeout=True)
    assert result_record(pool,job,180)['reason']=='timeout'
    assert pool.restarted==[(job,False)]
    assert result_record(pool,Job({'status':'accepted'},active_s=181),180)['reason']=='timeout'
    with pytest.raises(RuntimeError,match='Shared cache lock'):
        result_record(pool,Job(timeout=True,stage='cache_wait'),180)


def _delayed_result(delay):
    import time
    time.sleep(delay)
    return dict(status='accepted')


def test_real_worker_timeout_skips_and_next_candidate_runs():
    from framediff.online_workers import OnlineWorkerPool
    pool=OnlineWorkerPool(1)
    try:
        pool.submit(_delayed_result,0).result(timeout=20)
        previous=pool.slots[0].process.pid
        assert result_record(pool,pool.submit(_delayed_result,1),.05)['reason']=='timeout'
        assert pool.slots[0].process.pid!=previous
        assert pool.submit(_delayed_result,0).result(timeout=20)['status']=='accepted'
    finally:pool.shutdown()


def test_whole_page_replacement_and_resume(tmp_path):
    rows=[dict(id=f'{p}/v{v}/clean',group=p,split='train',slow=p=='bad' and v==1)
          for p in ('bad','good') for v in range(2)]
    path=tmp_path/'rows.jsonl';write_jsonl(path,rows)
    index=RowIndex(path)
    pages=[dict(id=p,split='train',group=p,views=[f'{p}/v{v}' for v in range(2)]) for p in ('bad','good')]
    folder=tmp_path/'records';folder.mkdir();pool=Pool()
    args=(pages,1,index,index,folder,pool,{'workers':2},180)
    chosen,report=select_split(*args)
    assert chosen==['good'] and report['reasons']=={'timeout':1}
    assert len(pool.calls)==4
    pool.calls.clear()
    assert select_split(*args)==(chosen,report)
    assert pool.calls==[]


def test_missing_view_does_not_silently_reduce_view_coverage(tmp_path):
    path=tmp_path/'rows.jsonl';write_jsonl(path,[])
    index=RowIndex(path);folder=tmp_path/'records';folder.mkdir()
    page=dict(id='p',split='test',group='g',views=['p/mobile'])
    with pytest.raises(ValueError,match='Only 0/1'):
        select_split([page],1,index,index,folder,Pool(),{'workers':1},180)
    assert next(folder.glob('*.json')).read_text().find('missing_prepared_view')>=0


def test_source_domains_and_duplicate_html_cannot_cross_splits(tmp_path):
    rows=[dict(id='a/desktop',page_id='a',split='train',group='same',source_sha='a'),
          dict(id='b/desktop',page_id='b',split='test',group='same',source_sha='b')]
    path=tmp_path/'source.jsonl';write_jsonl(path,rows)
    with pytest.raises(ValueError,match='domain crosses'):page_inventory(path)
    rows[1].update(group='different',source_sha='a');write_jsonl(path,rows)
    with pytest.raises(ValueError,match='HTML shared'):page_inventory(path)


def test_publication_keeps_fixed_labels_and_counts_unique_pages(tmp_path):
    source=tmp_path/'source';detectors=tmp_path/'detectors';out=tmp_path/'new'
    chosen={};sources=[];expected={}
    for split,n in (('train',8),('val',1),('test',1)):
        chosen[split]=[f'{split}-{i}' for i in range(n)]
        local=[dict(id=f'{split}-{i}/v{v}',page_id=f'{split}-{i}',split=split,
                    group=f'{split}-{i}.example',source_sha=f'{split}-{i}')
               for i in range(n+1) for v in range(2)]
        sources+=local
        rows=[dict(r,id=r['id']+'/clean',target_html='/existing/target.html',
                   replacement_edit=[0,'width','100px','important']) for r in local]
        expected[split]=rows[:2*n]
        for name,kinds in (('rendered',('policy','pages')),('labels',('policy','pages')),('css',('policy','pages'))):
            for kind in kinds:write_jsonl(source/'master'/name/f'{kind}-{split}.jsonl',rows if kind=='policy' else local)
        write_jsonl(detectors/f'detector-{split}.jsonl',rows)
    write_jsonl(source/'master/source/manifest.jsonl',sources)
    publish(source,detectors,out,10,chosen,{'seed':42})
    assert json.loads((out/'READY.json').read_text())['counts']==dict(train=8,val=1,test=1)
    for split in chosen:
        assert list(read_jsonl(out/'10/css'/f'policy-{split}.jsonl'))==expected[split]
        assert len(list(read_jsonl(out/'10/source'/f'manifest-{split}.jsonl')))==2*len(chosen[split])
    assert not list(out.rglob('*.png')) and not list(out.rglob('*.html'))


@pytest.mark.browser
def test_screening_reuses_lazy_cache_and_does_not_mutate_heldout(tmp_path,monkeypatch):
    from test_multiview_regressions import lazy_descriptor
    from framediff.lazy_webui import TrainViewCache
    from framediff.online_exclusions import TimeoutExclusions
    from framediff.tree_online import OnlineSampler,target_pool
    row=lazy_descriptor(tmp_path);shared=tmp_path/'shared'
    monkeypatch.setenv('TUIDE_SHARED_CACHE',str(shared));monkeypatch.setenv('TUIDE_SAVE_DIAGNOSTICS','0')
    options=dict(shared_cache=str(shared),observation_size=64,samples=2,max_noise=1,seed=42)
    original=copy.deepcopy(row)
    assert screen_view(row,row,options)['status']=='accepted'
    assert row==original
    from framediff import visual_data
    def no_render(*a,**kw):raise AssertionError('Repeated raw rendering')
    monkeypatch.setattr(visual_data,'_render_page',no_render)
    assert screen_view(row,row,options)['status']=='accepted'
    sampler=OnlineSampler(target_pool([row],cache_dir=shared),'screenshot',max_noise=1)
    try:heldout=sampler.generate(7)
    finally:sampler.close();sampler.view_cache.close()
    # Portable bytes are not file paths; use the prepared clean image for the
    # held-out fixture, as the real fixed manifests do.
    cache=TrainViewCache(cache_dir=shared)
    try:heldout['target_image']=cache.get(row,policy=True)['target_image']
    finally:cache.close()
    heldout['split']='val';before=copy.deepcopy(heldout)
    detector={'image':heldout['target_image']}
    assert screen_view(heldout,detector,options)['status']=='accepted'
    assert heldout==before
    exclusions=TimeoutExclusions([row]);exclusions.exclude(exclusions.describe(row),180)
    assert screen_view(row,row,options)['reason']=='known_timeout'


@pytest.mark.browser
def test_full_dom_screen_cache_connects_to_new_training_paths(tmp_path,monkeypatch):
    from test_multiview_regressions import lazy_descriptor
    from framediff.lazy_webui import TrainViewCache
    from framediff.policy_relabel import _task
    from framediff.visual import VISUAL_GEOMETRY_CONTRACT,DETECTOR_LABEL_CONTRACT
    from framediff.tree_online import OnlineSampler,target_pool
    from framediff import visual_data,tree_normalize,tree_online,policy_relabel
    original=lazy_descriptor(tmp_path);shared=tmp_path/'shared'
    monkeypatch.setenv('TUIDE_SHARED_CACHE',str(shared));monkeypatch.setenv('TUIDE_SAVE_DIAGNOSTICS','0')
    policy,error=_task((original,tmp_path/'relabel',64,True,tmp_path/'old-cache'))
    assert error is None
    detector={**original,'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT,
              'detector_label_contract':DETECTOR_LABEL_CONTRACT,
              'lazy_settings':{**original['lazy_settings'],'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT,
                               'detector_label_contract':DETECTOR_LABEL_CONTRACT,'min_elements':0}}
    options=dict(shared_cache=str(shared),observation_size=64,samples=1,max_noise=1,seed=42)
    assert screen_view(policy,detector,options)['status']=='accepted'
    # Publication preserves descriptor dictionaries, while their containing
    # manifests and the run directory move to the newly selected 7k dataset.
    data=tmp_path/'screened/7000';css=data/'css';rendered=data/'rendered'
    write_jsonl(css/'policy-train.jsonl',[policy])
    write_jsonl(css/'pages-train.jsonl',[{**policy,'id':policy['lazy_source']['id']}])
    write_jsonl(rendered/'detector-train.jsonl',[detector])
    def forbidden(*a,**kw):raise AssertionError('Repeated common preparation during training')
    for module,name in ((visual_data,'_render_page'),(tree_normalize,'_prepare_page'),
                        (policy_relabel,'refresh_row'),(tree_online,'inspect_candidates')):
        monkeypatch.setattr(module,name,forbidden)
    detection_cache=TrainViewCache(cache_dir=rendered/'.lazy-detector-cache')
    try:assert detection_cache.get(next(read_jsonl(rendered/'detector-train.jsonl')))['elements']
    finally:detection_cache.close()
    outputs=[]
    for mode,predicted in (('abstract',False),('screenshot',False),('abstract',True)):
        rows=list(read_jsonl(css/'policy-train.jsonl'))
        if predicted:
            rows=[{**r,'lazy_detector':{'checkpoint':'separate-detector.pt','sha':'test','threshold':.4}}
                  for r in rows]
        targets=target_pool(rows,css/'pages-train.jsonl',cache_dir=css/'.online-target-cache')
        sampler=OnlineSampler(targets,mode,max_noise=1)
        try:
            # A fresh online index outside the preprocessing probe sequence.
            sample=sampler.generate(9);outputs.append(sample)
            assert sample['online']['index']==9
            if predicted:assert sample['lazy_detector']==rows[0]['lazy_detector']
        finally:sampler.close();sampler.view_cache.close()
    assert all(r['replacement_edit']==outputs[0]['replacement_edit'] for r in outputs)
    assert all(r['current_boxes']==outputs[0]['current_boxes'] for r in outputs)
    assert 'current_image' not in outputs[0] and outputs[1]['current_image'].startswith(b'\x89PNG')


@pytest.mark.browser
def test_complete_pipeline_and_record_resume(tmp_path,monkeypatch):
    from test_multiview_regressions import lazy_descriptor
    from framediff.lazy_webui import TrainViewCache
    from framediff.tree_online import OnlineSampler,target_pool
    from framediff.webui_screen import run
    from framediff.online_workers import OnlineWorkerPool
    shared=tmp_path/'shared';monkeypatch.setenv('TUIDE_SHARED_CACHE',str(shared))
    monkeypatch.setenv('TUIDE_SAVE_DIAGNOSTICS','0')
    row=lazy_descriptor(tmp_path);cache=TrainViewCache(cache_dir=shared)
    try:clean=cache.get(row,policy=True)
    finally:cache.close()
    sampler=OnlineSampler(target_pool([row],cache_dir=shared),'screenshot',max_noise=1)
    try:sample=sampler.generate(0)
    finally:sampler.close();sampler.view_cache.close()
    image=tmp_path/'fixed.png';image.write_bytes(sample['current_image'])
    html=tmp_path/'fixed.html';html.write_text(sample.pop('current_html_text'))
    sample.update(target_image=clean['target_image'],current_image=str(image),current_html=str(html))
    source=tmp_path/'source';detectors=tmp_path/'detectors';out=tmp_path/'output';sources=[]
    for split,n in (('train',8),('val',1),('test',1)):
        policies=[];pages=[];detections=[]
        for i in range(n):
            page=f'{split}-{i}';view=page+'/desktop'
            identity=dict(page_id=page,split=split,group=page+'.example',source_sha=page)
            sources.append(dict(identity,id=view,html=row['lazy_source']['html'],viewport=[400,200]))
            policies.append(dict(sample,**identity,id=view+'/online-0'))
            pages.append(dict(identity,id=view,html=clean['target_html']))
            detections.append(dict(identity,id=view+'/clean',image=clean['target_image']))
        for name,kinds in (('rendered',('policy','pages')),('labels',('policy','pages')),('css',('policy','pages'))):
            for kind in kinds:write_jsonl(source/'master'/name/f'{kind}-{split}.jsonl',policies if kind=='policy' else pages)
        write_jsonl(detectors/f'detector-{split}.jsonl',detections)
    write_jsonl(source/'master/source/manifest.jsonl',sources)
    pidfile=tmp_path/'screen.pid';monkeypatch.setenv('SCREEN_PID_FILE',str(pidfile))
    args=SimpleNamespace(data=str(source),detector_rendered=str(detectors),out=str(out),
        workers=2,samples=1,max_noise=1,timeout=60,observation_size=64,seed=42,
        shared_cache=str(shared),size=10,resume=True)
    run(args)
    assert (out/'READY.json').is_file() and not pidfile.exists()
    before=(out/'10/css/policy-val.jsonl').read_bytes()
    (out/'READY.json').unlink()
    def forbidden(*a,**kw):raise AssertionError('Re-screened completed records')
    monkeypatch.setattr(OnlineWorkerPool,'submit',forbidden)
    run(args)
    assert (out/'READY.json').is_file() and not pidfile.exists()
    assert (out/'10/css/policy-val.jsonl').read_bytes()==before

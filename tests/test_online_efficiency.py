import copy
import json
import os
import time
from pathlib import Path

import pytest
import torch


def _isolated_task(value, marker, delay=0, lock=None):
    from framediff.cache_lock import cache_lock
    Path(marker).write_text(str(os.getpid()))
    if lock:
        with open(lock,'a') as handle:
            with cache_lock(handle):time.sleep(delay)
    else:time.sleep(delay)
    return value,os.getpid()


def _retry_task(marker):
    path=Path(marker)
    if not path.exists():
        path.write_text(str(os.getpid()));time.sleep(60)
    return 7,os.getpid()


def _gate_task(marker,release):
    Path(marker).write_text(str(os.getpid()))
    while not Path(release).exists():time.sleep(.02)
    return os.getpid()


def test_restart_does_not_interrupt_running_worker(tmp_path):
    from framediff.online_workers import OnlineWorkerPool
    pool=OnlineWorkerPool(2)
    try:
        slow=pool.submit(_retry_task,str(tmp_path/'slow'))
        healthy=pool.submit(_gate_task,str(tmp_path/'healthy'),str(tmp_path/'release'))
        deadline=time.monotonic()+20
        while (not (tmp_path/'slow').exists() or not (tmp_path/'healthy').exists()) and time.monotonic()<deadline:
            time.sleep(.02)
        expected=int((tmp_path/'healthy').read_text())
        with pytest.raises(TimeoutError):slow.result(timeout=.1)
        pool.restart(slow)
        assert any(s.process.pid==expected and s.process.is_alive() for s in pool.slots)
        (tmp_path/'release').touch()
        assert healthy.result(timeout=20)==expected
        assert slow.result(timeout=20)[0]==7
    finally:pool.shutdown()


def test_isolated_timeout_preserves_other_jobs(tmp_path):
    from framediff.online_workers import OnlineWorkerPool
    pool=OnlineWorkerPool(2)
    try:
        slow=pool.submit(_retry_task,str(tmp_path/'slow'))
        healthy=pool.submit(_isolated_task,11,str(tmp_path/'healthy'),.5)
        # Drain both pipes before forcing the slow worker's deadline.
        expected=healthy.result(timeout=20)
        old_pid=int((tmp_path/'slow').read_text())
        with pytest.raises(TimeoutError):slow.result(timeout=.1)
        pool.restart(slow)
        assert healthy.result(timeout=20)==expected
        value,new_pid=slow.result(timeout=20)
        assert value==7 and new_pid!=old_pid
        assert not any(slot.process.pid==old_pid for slot in pool.slots)
    finally:pool.shutdown()


def test_shared_lock_wait_does_not_consume_work_timeout(tmp_path):
    import fcntl
    from framediff.online_workers import OnlineWorkerPool
    path=tmp_path/'shared.lock';pool=OnlineWorkerPool(2,cache_wait_timeout=10)
    try:
        # Warm the spawned workers before testing subsecond deadlines.
        pool.submit(_isolated_task,0,str(tmp_path/'warm')).result(timeout=20)
        with path.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            blocked=pool.submit(_isolated_task,1,str(tmp_path/'wait'),0,str(path))
            pool.submit(_isolated_task,2,str(tmp_path/'other'),.8).result(timeout=20)
            assert blocked.stage=='cache_wait'
            # Release from a thread while result() waits beyond its active budget.
            import threading
            releaser=threading.Thread(target=lambda:(time.sleep(.6),fcntl.flock(handle,fcntl.LOCK_UN)))
            releaser.start()
            assert blocked.result(timeout=.3)[0]==1
            releaser.join()
    finally:pool.shutdown()


def test_only_definite_eligibility_failures_are_cached(tmp_path,monkeypatch):
    from framediff.lazy_webui import TrainViewCache
    from framediff.cache_lock import UnusableTrainView
    from framediff.tree_online import EmptyTargetPoolError
    cache=TrainViewCache(cache_dir=tmp_path/'cache')
    source=tmp_path/'source.html';source.write_text('<html></html>')
    from framediff.web_experiment import digest
    row={'lazy_source':{'html':str(source),'source_sha':digest(source)}}
    monkeypatch.setattr(cache,'_key',lambda *a:'entry')
    calls=[]
    def fail(*args):
        calls.append(1)
        raise EmptyTargetPoolError([{'reason':'no_visible_candidate'}])
    monkeypatch.setattr(cache,'_get',fail)
    with pytest.raises(EmptyTargetPoolError):cache.get(row,policy=True)
    with pytest.raises(UnusableTrainView):cache.get(row,policy=True)
    assert len(calls)==1
    assert json.loads((cache.cache_dir/'entry.failure.json').read_text())['key']=='entry'
    monkeypatch.setattr(cache,'_key',lambda *a:'transient')
    def transient(*args):
        raise EmptyTargetPoolError([{'reason':'inspection_error'}])
    monkeypatch.setattr(cache,'_get',transient)
    for _ in range(2):
        with pytest.raises(EmptyTargetPoolError):cache.get(row,policy=True)
    assert not (cache.cache_dir/'transient.failure.json').exists()


def _browser_retry(marker):
    from framediff.browser import Browser
    from framediff.preparation_workers import process_table
    with Browser() as browser:
        browser.page.set_content('<div>isolated restart</div>')
        path=Path(marker)
        if not path.exists():
            table=process_table();children=set();front={os.getpid()}
            while front:
                front={pid for pid,(parent,_) in table.items() if parent in front}-children
                children.update(front)
            path.write_text(json.dumps({'pid':os.getpid(),'children':list(children)}))
            time.sleep(60)
        return os.getpid()


@pytest.mark.browser
def test_isolated_restart_reaps_browser_descendants(tmp_path):
    from framediff.online_workers import OnlineWorkerPool
    from framediff.preparation_workers import process_table
    pool=OnlineWorkerPool(2)
    try:
        slow=pool.submit(_browser_retry,str(tmp_path/'browser'))
        healthy=pool.submit(_isolated_task,8,str(tmp_path/'healthy'))
        healthy_result=healthy.result(timeout=20)
        deadline=time.monotonic()+20
        while not (tmp_path/'browser').exists() and time.monotonic()<deadline:time.sleep(.05)
        tracked=json.loads((tmp_path/'browser').read_text())
        assert tracked['children']
        with pytest.raises(TimeoutError):slow.result(timeout=.1)
        pool.restart(slow)
        assert healthy.result(timeout=20)==healthy_result
        assert slow.result(timeout=20)!=tracked['pid']
        assert not set(tracked['children']).intersection(process_table())
    finally:pool.shutdown()


def _original_observation(browser,tree,viewport,size):
    from framediff.visual import refresh_visual_geometry,elements,semantic_masks
    from framediff.explicit_html import current_clip_boxes
    ids=[n['id'] for n in tree['nodes'][1:]]
    boxes=browser.tagged_boxes(ids);boxes[tree['nodes'][0]['id']]=[0,0,*viewport]
    refresh_visual_geometry(browser,tree)
    items=elements(tree,boxes,viewport,current_clip_boxes(browser,ids))
    return semantic_masks(items,viewport,size),boxes,items


@pytest.mark.browser
def test_fused_observation_and_candidate_lists_match_original(monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff import css_owners,tree_online
    html='''<html><body style="margin:0"><section style="width:190px;height:90px;overflow:hidden">
      <div style="width:240px;height:45px;background:red;margin:3px">Hello</div>
      <button style="width:80px;height:30px"><span>Button</span></button>
      <div style="opacity:0">Hidden</div><svg width="30" height="20"><rect width="30" height="20"/></svg>
      </section><div style="position:absolute;left:-15px;top:120px;border:2px solid;width:60px">Clipped</div></body></html>'''
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[300,200],max_nodes=5,visual_geometry=True)
        tree,_,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        source=dom['html'];parsed=css_owners.read(browser,tree)
        target={'current':tree,'viewport':[300,200],'css_owners':parsed['owners'],'corruption_observation_size':64}
        original=copy.deepcopy(tree);fused=copy.deepcopy(tree)
        expected=_original_observation(browser,original,[300,200],64)
        calls=[];evaluate=browser.page.evaluate
        monkeypatch.setattr(browser.page,'evaluate',lambda *a,**kw:(calls.append(1),evaluate(*a,**kw))[1])
        actual=tree_online.abstraction(browser,fused,[300,200],64)
        assert len(calls)==1
        assert torch.equal(actual[0],expected[0]) and actual[1:]==expected[1:]
        assert fused==original
        with monkeypatch.context() as patch:
            patch.setattr(tree_online,'abstraction',_original_observation)
            state,old=tree_online.inspect_candidates(browser,target,source)
        new_state,new=tree_online.inspect_candidates(browser,target,source)
        assert state==new_state and old==new


@pytest.mark.browser
def test_observation_clip_cache_refreshes_after_reflow_and_scroll():
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff.tree_online import abstraction
    html='''<html style="overflow:hidden"><body style="margin:0">
      <section id="outer" style="width:150px;height:90px;overflow:auto;border:3px solid">
        <div id="inner" style="width:240px;height:160px;overflow:hidden;padding:5px">
          <div style="margin-left:100px;width:200px;height:100px;background:red">Text</div>
          <button><span>Nested</span></button></div></section></body></html>'''
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[300,200],max_nodes=32,visual_geometry=True)
        tree,_,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        for edit in ('', 'outer.style.width="110px";outer.scrollLeft=35;outer.scrollTop=20',
                     'inner.style.visibility="hidden"', 'inner.style.visibility="visible";outer.style.overflow="visible"'):
            if edit:browser.page.evaluate('() => {'+edit+'}')
            expected=_original_observation(browser,copy.deepcopy(tree),[300,200],64)
            actual=abstraction(browser,copy.deepcopy(tree),[300,200],64)
            assert torch.equal(actual[0],expected[0]) and actual[1:]==expected[1:]


def test_cached_asset_hash_reused_and_content_change_detected(tmp_path,monkeypatch):
    from framediff.lazy_webui import TrainViewCache,_verified_asset_digest,DISK_CACHE_CONTRACT
    import hashlib
    asset=tmp_path/'target.png';asset.write_bytes(b'original')
    (tmp_path/'result.json').write_text(json.dumps(dict(contract=DISK_CACHE_CONTRACT,
        result={'target_image':str(asset)},assets={str(asset):hashlib.sha256(b'original').hexdigest()})))
    _verified_asset_digest.cache_clear()
    calls=[];digest=hashlib.file_digest
    monkeypatch.setattr(hashlib,'file_digest',lambda *a,**kw:(calls.append(1),digest(*a,**kw))[1])
    assert TrainViewCache._load_disk(tmp_path)
    assert TrainViewCache._load_disk(tmp_path)
    assert len(calls)==1
    stat=asset.stat();asset.write_bytes(b'modified')
    os.utime(asset,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert TrainViewCache._load_disk(tmp_path) is None
    assert len(calls)==2

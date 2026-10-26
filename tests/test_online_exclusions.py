from collections import deque
from concurrent.futures import TimeoutError
from types import SimpleNamespace

import pytest

from framediff.online_exclusions import TimeoutExclusions
from framediff.tree_online import OnlineGenerationError, OnlineSampler, OnlineStream


def view(name='page/desktop', sha='original'):
    return dict(id=name, source_sha=sha, viewport=[1280, 720])


def identified_task(target, delay):
    import time
    from framediff.cache_lock import report_online_target
    report_online_target(target)
    time.sleep(delay)
    return target


def test_worker_reports_page_before_timeout():
    from framediff.online_workers import OnlineWorkerPool
    pool = OnlineWorkerPool(1)
    identity = {'key':'test', 'page':'page/desktop', 'view':'desktop'}
    try:
        assert pool.submit(identified_task, identity, 0).result(timeout=20)==identity
        stalled = pool.submit(identified_task, identity, 5)
        with pytest.raises(TimeoutError):stalled.result(timeout=.2)
        assert stalled.target==identity and stalled.stage=='working'
    finally:pool.shutdown()


def test_exclusion_survives_restart_and_is_shared_between_modes(tmp_path, monkeypatch):
    monkeypatch.setenv('TUIDE_SHARED_CACHE', str(tmp_path))
    cache = TimeoutExclusions([])
    target = cache.describe(view())
    cache.exclude(target, 900)
    for mode in ('abstract', 'screenshot'):
        sampler = OnlineSampler([view()], mode, attempts=2)
        assert sampler.timeout_exclusions.contains(sampler.timeout_exclusions.describe(view()))
        # Excluded targets never reach browser/cache preparation.
        with pytest.raises(OnlineGenerationError) as exc:
            sampler.generate(5)
        assert all(e['error_type']=='ExcludedOnlineView' for e in exc.value.errors)
    assert not cache.contains(cache.describe(view('page/mobile')))
    assert not cache.contains(cache.describe(view(sha='changed')))
    assert not cache.contains(cache.describe({**view(), 'viewport':[800, 600]}))
    assert len(list(cache.directory.glob('*.json')))==1


@pytest.mark.parametrize('stages,same_view,excluded', [
    (['working', 'working'], True, True),
    (['cache_wait', 'working'], True, False),
    (['working', 'cache_wait'], True, False),
    (['working', 'working'], False, False),
])
def test_only_repeated_work_timeout_excludes_identified_view(tmp_path, monkeypatch, stages, same_view, excluded):
    monkeypatch.setenv('TUIDE_SHARED_CACHE', str(tmp_path))
    stream = OnlineStream([], 'screenshot', workers=1, prefetch=1, timeout=1)
    first = stream.timeout_exclusions.describe(view())
    second = first if same_view else stream.timeout_exclusions.describe(view('other/desktop'))

    class Stalled:
        def result(self, timeout=None):
            self.stage = stages.pop(0)
            self.target = first if stages else second
            raise TimeoutError('stalled')

    stalled = Stalled()
    ready = SimpleNamespace(result=lambda timeout: {'online':{'index':1}})
    stream.pending = deque([stalled, ready])
    stream.pool = SimpleNamespace(restart=lambda *args, **kwargs: None)
    monkeypatch.setattr(stream, '_fill', lambda: None)
    assert stream.take(1)[0]['online']['index']==1
    assert stream.timeout_exclusions.contains(second)==excluded
    assert stream.timeout_skips==1 and stream.timeout_retries==1


def test_target_progress_preserves_cache_wait_clock():
    from framediff.online_workers import OnlineWorkerPool, _Job
    pool = object.__new__(OnlineWorkerPool)
    job = _Job(pool, 0, None, ())
    job.stage = 'cache_wait'
    mark = job.mark
    target = {'key':'key', 'page':'page/desktop'}
    connection = SimpleNamespace(recv=lambda: ('progress', 0, mark+10, 'target', target))
    assert pool._receive(SimpleNamespace(connection=connection, job=job))
    assert job.target==target and job.stage=='cache_wait' and job.mark==mark


def test_transient_generation_errors_do_not_exclude_page(tmp_path, monkeypatch):
    monkeypatch.setenv('TUIDE_SHARED_CACHE', str(tmp_path))
    target = dict(view(), lazy_contract='lazy', lazy_source=view())
    sampler = OnlineSampler([target], 'abstract', attempts=2)
    attempts = []

    def transient(*args, **kwargs):
        attempts.append(1)
        raise ValueError('temporary browser failure')

    monkeypatch.setattr(sampler.view_cache, 'get', transient)
    for index in (1, 2):
        with pytest.raises(OnlineGenerationError):sampler.generate(index)
    assert len(attempts)==4
    assert not sampler.timeout_exclusions.contains(sampler.timeout_exclusions.describe(target))


def test_existing_prepared_cache_fingerprint_is_preserved():
    from framediff.lazy_webui import preparation_fingerprint
    assert preparation_fingerprint()['online_preparation']=='e598fecb8a9bb423bba5e9cf118b338f89be0bbe3d83195d5512e4e7c2fbf53d'

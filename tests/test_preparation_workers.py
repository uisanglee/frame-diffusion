import time

import pytest

from framediff.preparation_workers import page_map, write_json, write_jsonl


def delayed_square(value):
    time.sleep((4-value)*.01)
    return value*value


def test_spawn_preserves_order():
    assert list(page_map(delayed_square, range(4), 2)) == [0, 1, 4, 9]


def worker_identity(value):
    import os
    time.sleep(.1)
    return value,os.getpid()


def test_multiple_workers_execute_tasks():
    results=list(page_map(worker_identity,range(12),2))
    assert [value for value,_ in results]==list(range(12))
    assert len({pid for _,pid in results})==2


def test_frozen_candidate_inspection_reports_exclusion(monkeypatch,tmp_path):
    from framediff import tree_freeze
    from framediff.tree_online import EmptyTargetPoolError
    def excluded(rows,**kwargs):
        detail={'reason':'no_visible_candidate'}
        kwargs['diagnostics'].append(detail)
        raise EmptyTargetPoolError([detail])
    monkeypatch.setattr(tree_freeze,'target_pool',excluded)
    pool,failure=tree_freeze._inspect_page(({'id':'page/seed'},tmp_path,64))
    assert pool==[] and failure['id']=='page'
    assert failure['diagnostics']==[{'reason':'no_visible_candidate'}]


def test_interrupted_manifest_keeps_previous_file(tmp_path):
    path=tmp_path/'data.jsonl'
    write_jsonl(path,[{'id':'old'}])
    original=path.read_bytes()
    def interrupted():
        yield {'id':'new'}
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        write_jsonl(path,interrupted())
    assert path.read_bytes()==original
    write_jsonl(path,[{'id':'complete'}])
    assert 'complete' in path.read_text()


def test_interrupted_checkpoint_keeps_previous_file(tmp_path):
    path=tmp_path/'record.json'
    write_json(path,{'complete':True})
    original=path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path,{'bad':float('nan')})
    assert path.read_bytes()==original

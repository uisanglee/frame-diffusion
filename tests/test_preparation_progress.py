import json

import pytest

from framediff import preparation_progress as progress


@pytest.fixture
def clock(monkeypatch):
    now = [100.]
    monkeypatch.setattr(progress.time, 'monotonic', lambda: now[0])
    return now


def test_start_count_interval_and_completion_updates(tmp_path, clock, capsys):
    path = tmp_path/'nested/progress.json'
    reporter = progress.PreparationProgress('render', 100, path)
    first = reporter.update(0, kept=0, failures=0, cache_hits=0, workers=4)
    assert first['percent'] == 0 and first['eta_s'] is None
    assert 'eta=--' in capsys.readouterr().out
    initial_bytes = path.read_bytes()

    clock[0] = 110.
    assert reporter.update(49, kept=49) is None
    assert path.read_bytes() == initial_bytes
    assert capsys.readouterr().out == ''

    by_count = reporter.update(50, kept=48, failures=2, cache_hits=10)
    assert by_count['percent'] == 50 and by_count['elapsed_s'] == 10
    assert by_count['eta_s'] == 10
    assert '50/100 views (50.0%) kept=48 failures=2 cache_hits=10' in capsys.readouterr().out

    clock[0] = 140.
    by_time = reporter.update(51)
    assert by_time['elapsed_s'] == 40
    assert by_time['eta_s'] == pytest.approx(40*49/51, abs=.01)

    completed = reporter.update(100, kept=98, failures=2)
    assert completed['percent'] == 100 and completed['eta_s'] == 0
    assert json.loads(path.read_text()) == completed
    assert not path.with_name('progress.json.tmp').exists()


def test_force_and_empty_work(tmp_path, clock, capsys):
    reporter = progress.PreparationProgress('normalize-train', 2, tmp_path/'progress.json')
    reporter.update(0)
    clock[0] += 1
    assert reporter.update(1) is None
    assert reporter.update(1, force=True, kept=1)['eta_s'] == 1
    clock[0] += 90060
    forced = reporter.update(1, force=True, kept=1)
    assert forced['eta_s'] == 90061
    assert 'elapsed=25:01:01 eta=25:01:01' in capsys.readouterr().out
    empty = progress.PreparationProgress('normalize-val', 0, tmp_path/'empty.json').update(0)
    assert empty['percent'] == 100 and empty['eta_s'] == 0


@pytest.mark.parametrize('total', [-1, 1.5, True])
def test_invalid_total(total, tmp_path):
    with pytest.raises(ValueError, match='total'):
        progress.PreparationProgress('render', total, tmp_path/'progress.json')


@pytest.mark.parametrize('kwargs', [{'every': 0}, {'every': True}, {'interval': 0}, {'interval': float('inf')}])
def test_invalid_schedule(kwargs, tmp_path):
    with pytest.raises(ValueError):
        progress.PreparationProgress('render', 2, tmp_path/'progress.json', **kwargs)


@pytest.mark.parametrize('processed', [-1, 3, 1.5, True])
def test_invalid_count(processed, tmp_path, clock):
    reporter = progress.PreparationProgress('render', 2, tmp_path/'progress.json')
    with pytest.raises(ValueError, match='processed'):
        reporter.update(processed)
    assert not reporter.path.exists()


def test_counters_cannot_overwrite_progress_fields(tmp_path, clock):
    reporter = progress.PreparationProgress('render', 2, tmp_path/'progress.json')
    result = reporter.update(1, percent=999, stage='wrong', total=999, eta_s=-1)
    assert result['stage'] == 'render' and result['total'] == 2
    assert result['percent'] == 50 and result['eta_s'] == 0
    with pytest.raises(ValueError, match='decrease'):
        reporter.update(0)

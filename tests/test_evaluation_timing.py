import json
import pytest
from framediff.evaluation_timing import EvaluationTiming


def test_nested_timings_are_exclusive_and_failures_are_logged(monkeypatch, capsys):
    ticks = iter([0., 1., 4., 6.])
    monkeypatch.setattr('framediff.evaluation_timing.time.perf_counter', lambda: next(ticks))
    timing = EvaluationTiming(id='page', method='initial')
    with pytest.raises(ValueError):
        with timing.stage('clip'):
            with timing.stage('clip_image_preprocess'):
                raise ValueError('failed')
    assert timing.seconds == {'clip_image_preprocess': 3., 'clip': 3.}
    assert timing.inclusive_seconds['clip'] == 6.
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e['status'] for e in events] == ['started', 'started', 'failed', 'failed']
    assert all(e['id'] == 'page' for e in events)

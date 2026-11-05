import json
import pytest
from framediff.evaluation_report import summarize_logs, markdown


def write_log(path, rows):
    path.write_text('ordinary training log\n' + ''.join(json.dumps(r)+'\n' for r in rows))
    return path


def event(stage, status='completed', **kw):
    return dict(evaluation_stage=stage, status=status, id='a', method='initial', **kw)


def test_nested_summaries_and_caches_are_not_double_counted(tmp_path):
    path = write_log(tmp_path/'run.log', [
        event('clip', 'started'), event('clip_image_preprocess', 'started'),
        event('clip_image_preprocess', seconds=6, exclusive_seconds=6),
        event('clip_encode', 'started'), event('clip_encode', seconds=3, exclusive_seconds=3),
        event('clip', seconds=10, exclusive_seconds=1),
        event('block_matching', seconds=30, exclusive_seconds=30),
        event('method', stage_seconds={'clip': 1, 'block_matching': 30},
              stage_inclusive_seconds={'clip':10}),
        event('page', 'cache_hit'), event('method', 'cache_hit'), event('clip_encode', 'cache_hit'),
    ])
    result = summarize_logs([path])
    assert result['measured_stage_seconds'] == 40
    assert result['clip_total_seconds'] == 10 and result['clip_percent'] == 25
    assert result['stages'][0]['stage'] == 'block_matching'
    assert sum(r['percent'] for r in result['stages']) == pytest.approx(100)
    assert sum(r['seconds'] for r in result['groups']) == 40
    assert result['cache_reuse'] == dict(page_score_hits=1, method_score_hits=1,
                                      internal_operation_hits={'clip_encode':1})
    assert result['unfinished_stage_calls'] == 0


def test_failure_partial_and_old_logs(tmp_path):
    path = write_log(tmp_path/'run.log', [
        event('clip', 'started'), event('clip', 'failed', seconds=2, exclusive_seconds=2),
        event('block_matching', 'started'),
        event('method', stage_seconds={'clip':999}),  # aggregate summary ignored
        {'step':50, 'elapsed_s':1000},
        event('bad', exclusive_seconds=-1), event('bad', exclusive_seconds=float('nan')),
    ])
    with path.open('a') as f:
        f.write('{"evaluation_stage": "clip",')  # actively written partial line
    result = summarize_logs([path])
    assert result['measured_stage_seconds'] == 2
    assert result['stages'][0]['failed_calls'] == 1
    assert result['unfinished_stage_calls'] == 1
    assert result['invalid_event_lines'] == 2 and result['malformed_event_lines'] == 1


def test_empty_and_duplicate_paths(tmp_path):
    path = write_log(tmp_path/'old.log', [{'evaluation_seconds':500}, event('page', 'cache_hit')])
    result = summarize_logs([path])
    assert result['measured_stage_seconds'] == 0 and not result['stages']
    assert 'No completed timed stages' in markdown(result)
    with pytest.raises(ValueError, match='more than once'):
        summarize_logs([path, path])


def test_cli_writes_reports_without_touching_log(tmp_path, monkeypatch, capsys):
    from framediff.evaluation_report import main
    path = write_log(tmp_path/'run.log', [event('clip', exclusive_seconds=1, seconds=1)])
    original = path.read_bytes()
    out = tmp_path/'reports'
    monkeypatch.setattr('sys.argv', ['report', '--log', str(path), '--out', str(out)])
    main()
    assert json.loads((out/'evaluation-timing.json').read_text())['clip_total_seconds'] == 1
    assert 'CLIP total' in (out/'evaluation-timing.md').read_text()
    assert path.read_bytes() == original

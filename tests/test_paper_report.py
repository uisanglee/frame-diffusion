import pytest

from framediff.paper_report import report


def test_repeats_are_not_independent_pages_and_gains_are_paired(tmp_path):
    rows = []
    for page, repeats, before, after in [('a', 3, .2, .4), ('b', 1, .8, .6)]:
        for repeat in range(repeats):
            for method, score in [('initial', before), ('abstract-policy', after)]:
                rows.append(dict(id=f'{page}/repeat-{repeat+1}', method=method,
                                 dom_box_iou=score, pixel_mae=1-score,
                                 failed=False, evaluation_error=None,
                                 visual_timing={'seconds': 1., 'actions': 2}))
    result = report(rows, tmp_path)
    abstract = next(m for m in result['methods'] if m['method'] == 'abstract-policy')
    assert abstract['n_pages'] == 2 and abstract['n_trials'] == 4
    assert abstract['metrics']['dom_box_iou']['mean'] == pytest.approx(.5)
    gain = next(p for p in result['paired'] if p['metric'] == 'dom_box_iou')
    assert gain['mean'] == pytest.approx(0)
    assert gain['improved_page_rate'] == gain['worsened_page_rate'] == .5
    assert gain['ci95'][0] < 0 < gain['ci95'][1]
    assert (tmp_path/'paper-report.md').exists()


def test_missing_repeat_metric_not_zero_and_fallback_retained(tmp_path):
    rows = [dict(id='a/repeat-1', method='initial', dom_box_iou=.5, failed=False),
            dict(id='a/repeat-1', method='abstract-policy', dom_box_iou=.5, failed=True),
            dict(id='b/repeat-1', method='initial', dom_box_iou=.7, failed=False),
            dict(id='b/repeat-1', method='abstract-policy', dom_box_iou=.7, failed=False),
            dict(id='b/repeat-2', method='abstract-policy', evaluation_error='crash', failed=False)]
    result = report(rows, tmp_path)
    method = next(m for m in result['methods'] if m['method']=='abstract-policy')
    assert method['n_pages'] == 2
    assert method['metrics']['dom_box_iou']['n_pages'] == 1
    assert method['metrics']['dom_box_iou']['mean'] == .5
    assert method['metrics']['pipeline_failure_rate']['mean'] == .5

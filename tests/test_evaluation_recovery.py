import pytest
from framediff.evaluation_recovery import render_with_retry, reusable_metric


class Browser:
    resets = 0
    def reset_context(self): self.resets += 1


class Timing:
    def __init__(self): self.events = []
    def event(self, *args, **kwargs): self.events.append((args, kwargs))


def test_capture_retry_is_bounded_and_preserves_return(monkeypatch):
    monkeypatch.setattr('framediff.evaluation_recovery.time.sleep', lambda _: None)
    browser, timing = Browser(), Timing()
    calls = []
    def capture():
        calls.append(1)
        if len(calls) < 3: raise RuntimeError('Page.captureScreenshot: Unable to capture screenshot')
        return {'original': 42}
    assert render_with_retry(browser, capture, timing) == {'original': 42}
    assert len(calls) == 3 and browser.resets == 2 and len(timing.events) == 2
    def broken(): raise RuntimeError('Page.captureScreenshot: Unable to capture screenshot')
    with pytest.raises(RuntimeError): render_with_retry(browser, broken, timing)
    assert browser.resets == 4


def test_non_capture_errors_are_not_retried():
    browser = Browser()
    with pytest.raises(ValueError, match='invalid HTML'):
        render_with_retry(browser, lambda: (_ for _ in ()).throw(ValueError('invalid HTML')), Timing())
    assert browser.resets == 0


def test_resume_requires_finite_complete_metrics_but_keeps_failed_repair():
    row = dict(dom_box_iou=.5, official_position=.7, official_clip=.8, failed=True)
    assert reusable_metric(row, False, True)
    for key in ('evaluation_error', 'reference_error', 'geometry_error'):
        assert not reusable_metric(dict(row, **{key:'error'}), False, True)
    assert not reusable_metric(dict(row, official_position=None), False, True)
    assert not reusable_metric(dict(row, official_clip=float('nan')), False, True)
    assert reusable_metric({'evaluation_skipped':'initial_generation_failed'}, True, True)
    assert not reusable_metric(row, True, True)

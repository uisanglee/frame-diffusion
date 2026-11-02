from types import SimpleNamespace
from framediff import visual_experiment


def test_dead_browser_retains_completed_edit(monkeypatch):
    def fail(*args):
        stats = args[-1]
        stats['_last_success_html'] = '<html>one completed edit</html>'
        stats['_scroll_verification_pending'] = True
        stats['actions'] = 1
        raise RuntimeError('browser crashed during next edit')
    monkeypatch.setattr(visual_experiment, '_rollout', fail)
    html, stats, _ = visual_experiment.rollout(
        SimpleNamespace(executions=3, screenshots=2), '<html>initial</html>',
        {}, [100,100], 'unused', None)
    assert 'one completed edit' in html
    assert stats['failed'] and stats['actions'] == 1
    assert not any(key.startswith('_') for key in stats)

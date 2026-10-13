import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from framediff.browser import Browser, enable_worker_browser, close_worker_browser
from framediff.preparation_workers import page_map


@pytest.fixture
def runtime(monkeypatch):
    pw = Mock()
    pw.chromium.launch.return_value.is_connected.return_value = True
    api = SimpleNamespace(sync_playwright=lambda: SimpleNamespace(start=lambda: pw))
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', api)
    yield pw
    close_worker_browser()


@pytest.mark.parametrize('failure', ['launch', 'context', 'route', 'page'])
def test_initialization_failure_closes_runtime(runtime, failure):
    browser = runtime.chromium.launch.return_value
    context = browser.new_context.return_value
    operation = {'launch': runtime.chromium.launch, 'context': browser.new_context,
                 'route': context.route, 'page': context.new_page}[failure]
    operation.side_effect = RuntimeError('injected')
    with pytest.raises(RuntimeError, match='injected'):
        with Browser():pass
    assert runtime.stop.call_count == 1
    assert browser.close.call_count == (failure != 'launch')


def test_reset_close_failure_restarts_entire_runtime(runtime):
    with Browser() as lease:
        lease.context.close.side_effect = RuntimeError('close failed')
        lease.reset_context()
        assert runtime.chromium.launch.call_count == 2
        assert runtime.stop.call_count == 1
    assert runtime.stop.call_count == 2


def test_browser_close_failure_still_stops_driver(runtime):
    runtime.chromium.launch.return_value.close.side_effect = RuntimeError('close failed')
    with pytest.raises(RuntimeError, match='close failed'):
        with Browser():pass
    assert runtime.stop.call_count == 1


def test_shared_leases_reuse_browser_not_context(runtime):
    enable_worker_browser()
    for _ in range(4):
        with Browser():pass
    browser = runtime.chromium.launch.return_value
    assert runtime.chromium.launch.call_count == 1
    assert browser.new_context.call_count == 4
    assert browser.new_context.return_value.close.call_count == 4
    assert browser.close.call_count == 0
    close_worker_browser()
    assert browser.close.call_count == runtime.stop.call_count == 1


def test_shared_exit_close_failure_discards_runtime(runtime):
    enable_worker_browser()
    with pytest.raises(RuntimeError, match='close failed'):
        with Browser() as lease:
            lease.context.close.side_effect = RuntimeError('close failed')
    assert runtime.stop.call_count == 1
    runtime.chromium.launch.return_value.new_context.return_value.close.side_effect = None
    with Browser():pass
    assert runtime.chromium.launch.call_count == 2


def process_table():
    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True)
    return {int(parts[0]): (int(parts[1]), parts[2])
            for line in output.splitlines() if len(parts := line.strip().split(None, 2)) == 3}


def descendants(root):
    table = process_table()
    found = {root}
    while True:
        new = {pid for pid, (parent, _) in table.items() if parent in found}
        if new <= found:break
        found |= new
    return {pid: table[pid][1] for pid in found if pid != root
            and any(name in table[pid][1].lower() for name in ('playwright', 'chrome', 'chromium'))}


def browser_task(task):
    index, folder, fail = task
    with Browser() as lease:
        lease.page.set_content('<html><body>worker lifecycle</body></html>')
        # Include driver, browser and render subprocesses, not unrelated browsers.
        children = descendants(os.getpid())
        result = {'worker': os.getpid(), 'browser_identity': id(lease.browser), 'children': children}
        from pathlib import Path
        (Path(folder)/f'{index}.json').write_text(json.dumps(result))
        if fail:raise RuntimeError('task failed')
        return result


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('failure', [False, True])
def test_real_workers_leave_no_browser_processes(tmp_path, workers, failure):
    tasks = [(i, str(tmp_path), failure and i == 3) for i in range(6)]
    if failure:
        with pytest.raises(RuntimeError, match='task failed'):
            list(page_map(browser_task, tasks, workers))
    else:
        rows = list(page_map(browser_task, tasks, workers))
        for worker in {r['worker'] for r in rows}:
            assert len({r['browser_identity'] for r in rows if r['worker'] == worker}) == 1
    rows = [json.loads(path.read_text()) for path in tmp_path.glob('*.json')]
    assert rows and all(row['children'] for row in rows)
    tracked = {int(pid): command for row in rows for pid, command in row['children'].items()}
    assert_processes_gone(tracked)


def assert_processes_gone(tracked):
    remaining = tracked
    deadline = time.monotonic()+5
    while remaining and time.monotonic() < deadline:
        table = process_table()
        remaining = {pid: cmd for pid, cmd in tracked.items() if pid in table and table[pid][1] == cmd}
        if remaining:time.sleep(.1)
    assert not remaining, remaining


def test_real_context_initialization_failure_leaves_no_processes():
    tracked = {}
    class BrokenBrowser(Browser):
        def _new_context(self):
            super()._new_context()
            tracked.update(descendants(os.getpid()))
            raise RuntimeError('injected context failure')
    with pytest.raises(RuntimeError, match='injected context failure'):
        with BrokenBrowser():pass
    assert tracked
    assert_processes_gone(tracked)


def test_real_reset_context_failure_reaps_old_browser(monkeypatch):
    with Browser() as lease:
        old = descendants(os.getpid())
        def fail_close():raise RuntimeError('injected close failure')
        monkeypatch.setattr(lease.context, 'close', fail_close)
        lease.reset_context()
        assert_processes_gone(old)
        lease.page.set_content('<div>recovered</div>')
        current = descendants(os.getpid())
    assert current
    assert_processes_gone(current)


def test_early_generator_close_cleans_worker_browser(tmp_path):
    stream = page_map(browser_task, [(i, str(tmp_path), False) for i in range(3)], 1)
    row = next(stream)
    stream.close()
    assert_processes_gone({int(pid): cmd for pid, cmd in row['children'].items()})


def test_real_shared_browser_isolates_contexts():
    enable_worker_browser()
    tracked = {}
    try:
        with Browser() as first:
            first.context.add_cookies([{'name':'test','value':'secret','url':'https://example.org'}])
            browser = first.browser
            tracked.update(descendants(os.getpid()))
        assert browser.contexts == []
        with Browser() as second:
            assert second.browser is browser
            assert second.context.cookies() == []
    finally:
        close_worker_browser()
    assert_processes_gone(tracked)

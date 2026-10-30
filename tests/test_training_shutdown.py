import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from framediff.training_shutdown import cleanup_on_termination


def test_lifecycle_change_keeps_prepared_and_inspection_cache_keys():
    from framediff.browser_fingerprint import compatible_browser_digest
    from framediff.lazy_webui import preparation_fingerprint
    from framediff.tree_online import inspection_fingerprint
    legacy='4e3e43aef0e8deb0bae424d2dc5e8a438d86d7ec5149c09d836c48cdde3b8882'
    assert preparation_fingerprint()['browser.py'] == legacy
    assert inspection_fingerprint()['browser.py'] == legacy
    assert preparation_fingerprint()['online_preparation'] == 'e598fecb8a9bb423bba5e9cf118b338f89be0bbe3d83195d5512e4e7c2fbf53d'
    assert compatible_browser_digest('future-change') == 'future-change'


def test_handlers_restored_and_cleanup_survives_second_signal():
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    cleaned = []

    @cleanup_on_termination
    def training():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            os.kill(os.getpid(), signal.SIGTERM)
            cleaned.append(True)

    with pytest.raises(SystemExit) as exc:
        training()
    assert exc.value.code == 143
    assert cleaned == [True]
    assert before == {s: signal.getsignal(s) for s in before}


@pytest.mark.browser
@pytest.mark.parametrize('sig', [signal.SIGTERM, signal.SIGINT])
def test_training_signal_reaps_worker_and_browser(tmp_path, sig):
    from framediff.preparation_workers import process_table
    script = tmp_path / 'parent.py'
    script.write_text('''
import os, sys, json, time
from pathlib import Path
from framediff.browser import Browser
from framediff.online_workers import OnlineWorkerPool
from framediff.preparation_workers import process_table
from framediff.training_shutdown import cleanup_on_termination

def work(marker):
    with Browser() as browser:
        browser.page.set_content('<div>shutdown test</div>')
        table=process_table();children={os.getpid()};front=children
        while front:
            front={pid for pid,(parent,_) in table.items() if parent in front}-children
            children.update(front)
        Path(marker).write_text(json.dumps(sorted(children)))
        time.sleep(60)

@cleanup_on_termination
def main(marker):
    pool=OnlineWorkerPool(1)
    try:pool.submit(work,marker).result(timeout=90)
    finally:pool.shutdown()

if __name__=='__main__':main(sys.argv[1])
''')
    marker = tmp_path / 'children.json'
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    with (tmp_path / 'output.log').open('w+') as output:
        parent = subprocess.Popen([sys.executable, str(script), str(marker)], env=env,
                                  stdout=output, stderr=output)
        tracked = []
        try:
            deadline = time.monotonic() + 25
            while not marker.exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert marker.exists(), (tmp_path / 'output.log').read_text()
            tracked = json.loads(marker.read_text())
            assert len(tracked) > 1
            parent.send_signal(sig)
            assert parent.wait(timeout=20) == 128 + sig
            assert not set(tracked).intersection(process_table())
        finally:
            if parent.poll() is None:
                parent.kill();parent.wait()
            # Only processes started by this test, if an assertion failed.
            for pid in tracked:
                try:os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:pass

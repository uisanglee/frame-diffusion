import json
from pathlib import Path
import pytest
import framediff.paper_matrix as matrix


@pytest.mark.parametrize('with_tuide',['0','1'])
def test_ui2coden_rerun_aggregates_previous_methods(tmp_path,with_tuide):
    import os, subprocess, sys
    script=tmp_path/'scripts/rerun_pilot_ui2coden.sh'
    script.parent.mkdir()
    script.write_text(Path('scripts/rerun_pilot_ui2coden.sh').read_text())
    activate=tmp_path/'.venv/bin/activate';activate.parent.mkdir(parents=True);activate.write_text('true\n')
    old=tmp_path/'old';out=tmp_path/'new'
    (old/'inputs').mkdir(parents=True)
    (old/'inputs/design2code.jsonl').write_text('{"id":"page"}\n')
    for alias in ('llava-next-7b','glm4v-9b-thinking','websight-8b','design2code-18b'):
        run=old/'design2code'/alias/'seeds/42'
        for name in ('prepare','evaluation','repair'):(run/name).mkdir(parents=True)
        (run/'prepare/prepared.jsonl').write_text('{"id":"page"}\n')
    tools=tmp_path/'bin';tools.mkdir()
    py=tools/'python'
    py.write_text('#!/bin/sh\nif [ "$1" = "-" ]; then exec "$REAL_PYTHON" "$@"; fi\nprintf "%s\\n" "$*" >> "$PY_LOG"\n')
    py.chmod(0o755)
    bash=tools/'bash';bash.write_text('#!/bin/sh\nprintf "%s|%s\\n" "$RUN_TUIDE" "$*" >> "$BASH_LOG"\n');bash.chmod(0o755)
    env=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ['PATH'],REAL_PYTHON=sys.executable,
             RUN_TUIDE=with_tuide,PY_LOG=str(tmp_path/'py.log'),BASH_LOG=str(tmp_path/'bash.log'))
    result=subprocess.run(['/bin/bash',str(script),str(old),str(out)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    calls=(tmp_path/'py.log').read_text()
    assert 'framediff.paper_suite' in calls and '--aliases llava-next-7b,glm4v-9b-thinking' in calls
    assert 'framediff.paper_matrix export' in calls and '--datasets design2code --revision-rounds 2' in calls
    assert ('--skip-tuide' in calls)==(with_tuide=='0')
    assert (out/'design2code/llava-next-7b/seeds/42/evaluation').is_symlink()
    bashcalls=(tmp_path/'bash.log').read_text().splitlines()
    # Three external Initial models plus UI2CodeN's own Initial group.
    assert len(bashcalls)==4
    assert not any('/design2code-18b/' in line for line in bashcalls)
    assert all(line.startswith(with_tuide+'|') for line in bashcalls)
    assert 'run_paper_experiments.sh' in bashcalls[-1]


@pytest.mark.parametrize('pages', ['0', '3'])
def test_full_baseline_wrapper_forwards_fixed_output_and_page_count(tmp_path, pages):
    import os, subprocess
    scripts=tmp_path/'scripts';scripts.mkdir()
    script=scripts/'run_d2c_vlm_baselines.sh'
    script.write_text(Path('scripts/run_d2c_vlm_baselines.sh').read_text())
    (scripts/'run_d2c_pilot10_bf16_2gpu.sh').write_text(
        'printf "%s|%s|%s" "$D2C_PILOT_PAGES" "$D2C_PILOT_OUTPUT" "$CUDA_VISIBLE_DEVICES"\n')
    env=dict(os.environ, D2C_BASELINE_PAGES=pages, D2C_BASELINE_OUTPUT='runs/saved',
             CUDA_VISIBLE_DEVICES='2,4')
    result=subprocess.run(['/bin/bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.endswith(f'{pages}|runs/saved|2,4')


@pytest.mark.parametrize('count', [0, 3, 10])
def test_d2c_only_pilot_freezes_inputs(tmp_path, monkeypatch, count):
    script=Path('scripts/run_d2c_pilot10_bf16_2gpu.sh').read_text()
    selection=script.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
    roots=[tmp_path/'raw-d2c',tmp_path/'raw-hard']
    for root in roots:
        root.mkdir()
        (root/'rick.jpg').write_bytes(b'placeholder')
    rows=[dict(id=f'{i:03d}') for i in reversed(range(12))]
    monkeypatch.setattr(matrix,'check_sources',lambda **kw: rows.copy())
    monkeypatch.setattr('sys.argv',['pilot',*map(str,roots),str(tmp_path/'out'),str(count)])
    exec(selection,{})
    for name in ('design2code','design2code-hard'):
        path=tmp_path/'out/inputs'/f'{name}.jsonl'
        assert [json.loads(line)['id'] for line in path.read_text().splitlines()]==[f'{i:03d}' for i in range(count or 12)]
    assert not (tmp_path/'out/inputs/webui.jsonl').exists()
    exec(selection,{})
    rows[-1]['changed']=True
    with pytest.raises(ValueError,match='Pilot inputs changed'):
        exec(selection,{})


def test_vlm_only_path_evaluates_without_checkpoints(tmp_path):
    import os
    import subprocess
    tools=tmp_path/'bin';tools.mkdir()
    python=tools/'python'
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    python.chmod(0o755)
    manifest=tmp_path/'pages.jsonl';manifest.write_text('{}\n')
    env=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ['PATH'],
        CALL_LOG=str(tmp_path/'calls'),PAGE_MANIFEST=str(manifest),RUN_TUIDE='0',
        SELF_REVISION_PROTOCOL='design2code',REVISION_ROUNDS='2',
        TARGET_IMAGE_SOURCE='reference-render',OFFICIAL_REPO='official',
        GENERATE_INITIAL='1',PAPER_REFERENCE_CACHE=str(tmp_path/'reference-cache'))
    result=subprocess.run(['/bin/bash','scripts/run_visual_web.sh','.',str(tmp_path/'out'),'missing-checkpoints'],
        env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    calls=(tmp_path/'calls').read_text().splitlines()
    assert len(calls)==2
    assert 'web-prepare' in calls[0] and '--rounds 2' in calls[0]
    assert '--target-image-source reference-render' in calls[0]
    assert 'web-evaluate' in calls[1] and 'prepare/prepared.jsonl' in calls[1]
    assert '--official-repo official' in calls[1]


def test_pilot_freezes_ten_unique_pages_and_prefers_desktop(tmp_path, monkeypatch):
    script = Path('scripts/run_paper_pilot10.sh').read_text()
    selection = script.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    rows = [dict(id=f'{i:03d}/{view}', page_id=f'{i:03d}', view=view)
            for i in range(12) for view in ('iPad-Pro', 'default_1280-720')]
    monkeypatch.setattr(matrix, 'check_sources', lambda **kw: rows.copy())
    monkeypatch.setattr('sys.argv', ['pilot', 'webui', 'd2c', 'hard', str(tmp_path)])
    exec(selection, {})
    path = tmp_path/'inputs/webui.jsonl'
    saved = path.read_text()
    selected = [json.loads(line) for line in saved.splitlines()]
    assert len(selected) == len({r['page_id'] for r in selected}) == 10
    assert all(r['view'] == 'default_1280-720' for r in selected)
    for name in ('design2code', 'design2code-hard'):
        assert len((tmp_path/f'inputs/{name}.jsonl').read_text().splitlines()) == 10
    exec(selection, {})
    assert path.read_text() == saved
    rows[0]['page_id'] = 'changed'
    rows[1]['page_id'] = 'changed'
    with pytest.raises(ValueError, match='Pilot inputs changed'):
        exec(selection, {})

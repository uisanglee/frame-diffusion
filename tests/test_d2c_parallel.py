from pathlib import Path
import pytest
import framediff.d2c_parallel_merge as merger


def setup_groups(root):
    for group, models in merger.GROUPS.items():
        (root/group/'inputs').mkdir(parents=True)
        for dataset in merger.DATASETS:
            (root/group/'inputs'/f'{dataset}.jsonl').write_text('{"id":"1"}\n')
            for model in models:
                for suffix in (Path(model), Path('paper/main')/model):
                    (root/group/dataset/suffix).mkdir(parents=True)


def test_merge_preserves_initials_and_checks_cohorts(tmp_path, monkeypatch):
    setup_groups(tmp_path)
    calls=[]
    monkeypatch.setattr(merger, 'export_matrix', lambda *a, **kw: calls.append((a, kw)))
    merger.merge(tmp_path)
    assert len(calls)==3
    for group, models in merger.GROUPS.items():
        for model in models:
            link=tmp_path/'combined/design2code'/model
            assert link.is_symlink()
            assert link.resolve()==tmp_path/group/'design2code'/model
    merger.merge(tmp_path)  # idempotent resume; no copying/regeneration
    (tmp_path/'gpu24/inputs/design2code.jsonl').write_text('{"id":"2"}\n')
    with pytest.raises(ValueError, match='different inputs'):
        merger.merge(tmp_path)


def test_launcher_runs_disjoint_queues_concurrently(tmp_path):
    import os, subprocess
    scripts=tmp_path/'scripts';scripts.mkdir()
    script=scripts/'run_d2c_vlm_parallel.sh'
    script.write_text(Path('scripts/run_d2c_vlm_parallel.sh').read_text())
    activate=tmp_path/'.venv/bin/activate';activate.parent.mkdir(parents=True)
    activate.write_text('true\n')
    (scripts/'run_d2c_vlm_baselines.sh').write_text('''
mkdir -p "$D2C_BASELINE_OUTPUT"
printf '%s|%s' "$CUDA_VISIBLE_DEVICES" "$VLM_SPECS" > "$D2C_BASELINE_OUTPUT/started"
for i in {1..100}; do
  if [[ -f "$D2C_PARALLEL_OUTPUT/gpu01/started" && -f "$D2C_PARALLEL_OUTPUT/gpu24/started" ]]; then exit 0; fi
  sleep 0.05
done
exit 9
''')
    tools=tmp_path/'bin';tools.mkdir()
    python=tools/'python';python.write_text('#!/bin/sh\necho "$*" > "$D2C_PARALLEL_OUTPUT/merge-call"\n')
    python.chmod(0o755)
    out=tmp_path/'out'
    env=dict(os.environ, D2C_PARALLEL_OUTPUT=str(out), PATH=str(tools)+os.pathsep+os.environ['PATH'])
    result=subprocess.run(['/bin/bash',str(script)],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    a=(out/'gpu01/started').read_text(); b=(out/'gpu24/started').read_text()
    assert a.startswith('0,1|') and b.startswith('2,4|')
    assert 'llava-next-7b|' in a and 'glm4v-9b-thinking|' in a and 'websight-8b|' not in a
    assert 'websight-8b|' in b and 'ui2coden|' in b and 'llava-next-7b|' not in b
    assert 'framediff.d2c_parallel_merge' in (out/'merge-call').read_text()

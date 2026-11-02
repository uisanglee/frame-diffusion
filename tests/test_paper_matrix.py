import json
import pytest
from framediff.paper_matrix import DATASETS, MODELS, METRICS, export_matrix, check_sources


def test_matrix_has_all_groups_and_rejects_missing_scores(tmp_path):
    for dataset, _ in DATASETS:
        for alias, _ in MODELS:
            methods = ['Initial VLM', 'TUIDE']
            methods += ['UI2CodeN self-revision'] if alias == 'ui2coden' else ['D2C Self-Revision (RGB)', 'UI2CodeN polishing']
            summary = dict(complete_metrics_required=True, target_image_protocol='local-reference-render-v1',
                initial_seeds=['11','22'], fixed_checkpoint_hashes=['fixed'], methods={
                    m: dict(n_pages=10, metrics={k: dict(mean=.5, sd=.1) for k in METRICS}) for m in methods})
            folder = tmp_path/dataset/'paper/main'/alias
            folder.mkdir(parents=True)
            (folder/'initial-seed-summary.json').write_text(json.dumps(summary))
    output = export_matrix(tmp_path, tmp_path/'table')
    table = output.read_text()
    assert table.count(' & Initial & ') == 6
    assert table.count(' & Self Revision & ') == 6
    assert table.count(' & TUIDE & ') == 6
    assert 'last available implementation' in table
    assert r'\multirow{3}{*}{UI2Code$^{n}$}' in table
    for line in table.splitlines():
        if ' & TUIDE & ' in line:
            assert line.count('&') == 13
    summary['methods']['TUIDE']['metrics']['official_position']['mean'] = None
    (folder/'initial-seed-summary.json').write_text(json.dumps(summary))
    with pytest.raises((ValueError, TypeError)):
        export_matrix(tmp_path, tmp_path/'invalid')


def test_preflight_requires_heldout_html(tmp_path):
    manifest = tmp_path/'pages.jsonl'
    manifest.write_text(json.dumps(dict(id='x', split='train', html='missing', screenshot='missing'))+'\n')
    with pytest.raises(ValueError, match='held-out'):
        check_sources(manifest=manifest)
    manifest.write_text(json.dumps(dict(id='x', split='test', html='missing', screenshot='missing'))+'\n')
    with pytest.raises(ValueError, match='missing html'):
        check_sources(manifest=manifest)


def test_shell_wires_own_draft_shared_inputs_and_webui(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    script = tmp_path/'scripts/run_paper_experiments.sh'
    script.parent.mkdir()
    script.write_text((repo/'scripts/run_paper_experiments.sh').read_text())
    baseline = tmp_path/'baselines/ui2coden/.venv/bin/python'
    baseline.parent.mkdir(parents=True)
    baseline.write_text('#!/bin/sh\nexit 0\n')
    baseline.chmod(0o755)
    tools = tmp_path/'bin'; tools.mkdir()
    py = tools/'python'
    py.write_text('#!/bin/sh\nif [ "$1" = "-" ]; then cat >/dev/null; echo pinned-sha; fi\n')
    py.chmod(0o755)
    bash = tools/'bash'
    bash.write_text("""#!/bin/sh
printf '%s|%s|%s|%s|%s|%s|%s|%s\n' "$1" "$VLM_MODEL" "$WEB_DATASET" "$REVISION_ROUNDS" "$REFERENCE_MANIFEST" "$ROUNDS" "$MODEL_REVISION" "$ABSTRACT_ONLY" >> "$CALL_LOG"
""")
    bash.chmod(0o755)
    env = dict(os.environ, PATH=str(tools)+os.pathsep+os.environ['PATH'], CALL_LOG=str(tmp_path/'calls'),
        OFFICIAL_REPO='official', RUN_UI2CODEN='1', RUN_ABLATION='0', INITIAL_SEEDS='11,22',
        WEB_DATASET='webui', PAGE_MANIFEST='pages.jsonl',
        VLM_SPECS='q|hf|qwen|4bit,ui2coden|hf|zai-org/UI2Code_N|4bit')
    result = subprocess.run(['/bin/bash',str(script),'data','out','train'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    calls = [line.split('|') for line in (tmp_path/'calls').read_text().splitlines()]
    drafts = [r for r in calls if r[0]=='scripts/run_visual_web.sh']
    assert [r[3] for r in drafts] == ['3','3','0','0']
    assert all(r[2]=='webui' and r[7]=='1' for r in drafts)
    assert drafts[0][4] == ''
    assert all(r[4]=='out/q/seeds/11/prepare/prepared.jsonl' for r in drafts[1:])
    polish = [r for r in calls if r[0]=='baselines/ui2coden/run_4090.sh']
    assert len(polish)==4 and all(r[5:7]==['3','pinned-sha'] for r in polish)

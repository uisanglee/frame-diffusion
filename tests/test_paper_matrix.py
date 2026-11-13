import json
import pytest
from framediff.paper_matrix import DATASETS, MODELS, METRICS, NO_SELF_REVISION_ALIASES, LEGACY_MODELS, export_matrix, check_sources


def test_matrix_has_all_groups_and_rejects_missing_scores(tmp_path):
    for dataset, _ in DATASETS:
        for alias, _ in MODELS:
            methods = ['Initial VLM', 'TUIDE']
            methods += ['UI2CodeN self-revision'] if alias == 'ui2coden' else ['D2C Self-Revision (RGB)', 'UI2CodeN polishing']
            if alias in NO_SELF_REVISION_ALIASES: methods.remove('D2C Self-Revision (RGB)')
            summary = dict(complete_metrics_required=True, target_image_protocol='local-reference-render-v1',
                initial_seeds=['11','22'], fixed_checkpoint_hashes=['fixed'], methods={
                    m: dict(n_pages=10, metrics={k: dict(mean=.5, sd=.1) for k in METRICS}) for m in methods})
            summary['cohort_protocol'] = 'initial-success-shared-v1'
            n = 8 if alias == 'llava-next-7b' else 10
            summary['initial_coverage_by_seed'] = {seed: dict(attempted=10, successful=n, failed=10-n) for seed in ('11','22')}
            for item in summary['methods'].values():
                item['n_pages'] = n
                item['n_pages_by_seed'] = {'11': n, '22': n}
            summary['methods']['Initial VLM']['metrics']['initial_success_rate'] = dict(mean=n/10, sd=0.)
            folder = tmp_path/dataset/'paper/main'/alias
            folder.mkdir(parents=True)
            (folder/'initial-seed-summary.json').write_text(json.dumps(summary))
    output = export_matrix(tmp_path, tmp_path/'table')
    table = output.read_text()
    assert r'Position $\uparrow$ & CLIP $\uparrow$ & Box IoU $\uparrow$' in table
    assert 'Success / Improved' in table
    assert '80.000' in table
    assert 'Block $' not in table
    assert table.count(' & Initial & ') == 4
    assert table.count(' & Self Revision & ') == 3
    assert table.count(' & TUIDE & ') == 4
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


@pytest.mark.parametrize('models', [None, LEGACY_MODELS])
def test_export_two_datasets_without_tuide(tmp_path, models):
    datasets=['design2code','design2code-hard']
    for dataset in datasets:
        for alias,_ in (models or MODELS):
            methods=['Initial VLM','UI2CodeN self-revision'] if alias=='ui2coden' else [
                'Initial VLM','D2C Self-Revision (RGB)','UI2CodeN polishing']
            if alias in NO_SELF_REVISION_ALIASES: methods.remove('D2C Self-Revision (RGB)')
            record=dict(complete_metrics_required=True,target_image_protocol='local-reference-render-v1',
                initial_seeds=['42'],fixed_checkpoint_hashes={},cohort_protocol='initial-success-shared-v1',
                initial_coverage_by_seed={'42':dict(attempted=10,successful=8,failed=2)},methods={})
            for method in methods:
                record['methods'][method]=dict(n_pages_by_seed={'42':8},metrics={
                    k:dict(mean=.5,sd=None) for k in (*METRICS,'initial_success_rate')})
            folder=tmp_path/dataset/'paper/main'/alias;folder.mkdir(parents=True)
            (folder/'initial-seed-summary.json').write_text(json.dumps(record))
    table=export_matrix(tmp_path,tmp_path/'table',datasets,False,2, None if models is None else [a for a, _ in models]).read_text()
    assert 'TUIDE' not in table and 'WebUI' not in table
    assert '2 refinement rounds' in table
    assert table.count(' & Initial & ')==len(models or MODELS)
    assert table.count(' & Self Revision & ')==(6 if models else 3)
    assert r'\begin{tabular}{ll*{2}{cccc}}' in table
    assert all(line.count('&')==9 for line in table.splitlines() if ' & Initial & ' in line)


@pytest.mark.parametrize('run_tuide',['0','1'])
@pytest.mark.parametrize('dataset,output,budget', [('webui','webui','16384'),
    ('design2code','design2code','4096'), ('design2code','design2code-hard','4096')])
def test_shell_wires_own_draft_shared_inputs_and_webui(tmp_path,run_tuide,dataset,output,budget):
    import os
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    script = tmp_path/'scripts/run_paper_experiments.sh'
    script.parent.mkdir()
    script.write_text((repo/'scripts/run_paper_experiments.sh').read_text())
    (script.parent/'experiment_log.sh').write_text((repo/'scripts/experiment_log.sh').read_text())
    baseline = tmp_path/'baselines/ui2coden/.venv/bin/python'
    baseline.parent.mkdir(parents=True)
    baseline.write_text('#!/bin/sh\nexit 0\n')
    baseline.chmod(0o755)
    tools = tmp_path/'bin'; tools.mkdir()
    py = tools/'python'
    py.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PY_LOG"\nif [ "$1" = "-" ]; then cat >/dev/null; echo pinned-sha; fi\n')
    py.chmod(0o755)
    bash = tools/'bash'
    bash.write_text("""#!/bin/sh
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' "$1" "$VLM_MODEL" "$WEB_DATASET" "$REVISION_ROUNDS" "$REFERENCE_MANIFEST" "$ROUNDS" "$MODEL_REVISION" "$ABSTRACT_ONLY" "$MAX_NEW_TOKENS" "$MAX_CONTEXT" >> "$CALL_LOG"
""")
    bash.chmod(0o755)
    env = dict(os.environ, PATH=str(tools)+os.pathsep+os.environ['PATH'], CALL_LOG=str(tmp_path/'calls'),
        RUN_TUIDE=run_tuide,PY_LOG=str(tmp_path/'python-calls'),PAPER_DIAGNOSTICS='1',
        OFFICIAL_REPO='official', RUN_UI2CODEN='1', RUN_ABLATION='0', INITIAL_SEEDS='11,22',
        WEB_DATASET=dataset, PAGE_MANIFEST='pages.jsonl', MAX_NEW_TOKENS='16384', MAX_CONTEXT='12288',
        UI2CODEN_MAX_NEW_TOKENS='8192', UI2CODEN_MAX_CONTEXT='0',
        VLM_SPECS='q|hf|qwen|4bit,ui2coden|hf|zai-org/UI2Code_N|4bit')
    result = subprocess.run(['/bin/bash',str(script),'data',output,'train'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    calls = [line.split('|') for line in (tmp_path/'calls').read_text().splitlines()]
    drafts = [r for r in calls if r[0]=='scripts/run_visual_web.sh']
    assert [r[3] for r in drafts] == ['3','3','0','0']
    assert all(r[2]==dataset and r[7]=='1' for r in drafts)
    assert all(r[8] == budget for r in drafts)
    assert drafts[0][4] == ''
    assert all(r[4]==f'{output}/q/seeds/11/prepare/prepared.jsonl' for r in drafts[1:])
    polish = [r for r in calls if r[0]=='baselines/ui2coden/run_4090.sh']
    assert len(polish)==4 and all(r[5:7]==['3','pinned-sha'] for r in polish)
    assert all(r[8:10] == [budget, '0'] for r in polish)
    assert 'overriding inherited token-budget variables' in result.stdout
    py_calls=(tmp_path/'python-calls').read_text()
    assert ('--skip-tuide' in py_calls)==(run_tuide=='0')
    assert ('framediff.repair_diagnostics' in py_calls)==(run_tuide=='1')


def test_official_roster_contains_only_requested_models():
    from pathlib import Path
    expected = ['llava-next-7b', 'glm4v-9b-thinking', 'websight-8b', 'ui2coden']
    assert [alias for alias, _ in MODELS] == expected
    for script in ('run_paper_4090_vlms.sh', 'run_paper_pilot10_bf16_2gpu.sh', 'run_d2c_pilot10_bf16_2gpu.sh'):
        line = next(s for s in (Path('scripts')/script).read_text().splitlines() if s.startswith('export VLM_SPECS='))
        specs = line.split("'", 2)[1].split(',')
        assert [spec.split('|')[0] for spec in specs] == expected


def test_image_only_drafters_select_direct_generation_without_self_revision(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    script = Path('scripts/run_paper_experiments.sh').resolve()
    tools = tmp_path/'bin'; tools.mkdir()
    py = tools/'python'
    py.write_text('#!/bin/sh\nif [ "$1" = "-" ]; then cat >/dev/null; echo pinned-sha; fi\n')
    py.chmod(0o755)
    bash = tools/'bash'
    bash.write_text('#!/bin/sh\nprintf "%s|%s|%s|%s\\n" "$VLM_BACKEND" "$SELF_REVISION_PROTOCOL" "$REVISION_ROUNDS" "$MAX_NEW_TOKENS" >> "$CALL_LOG"\n')
    bash.chmod(0o755)
    env = dict(os.environ, PATH=str(tools)+os.pathsep+os.environ['PATH'],
        CALL_LOG=str(tmp_path/'calls'), OFFICIAL_REPO='official', RUN_UI2CODEN='0', RUN_TUIDE='0',
        RUN_ABLATION='0', INITIAL_SEEDS='42', CUDA_VISIBLE_DEVICES='0,1',
        WEBSIGHT_PYTHON=str(py), DESIGN2CODE_PYTHON=str(py),
        VLM_SPECS='websight-8b|websight|HuggingFaceM4/VLM_WebSight_finetuned|bf16,design2code-18b|design2code-sat|SALT-NLP/Design2Code-18B-v0|bf16')
    result = subprocess.run(['/bin/bash',str(script),'data','out','unused'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path/'calls').read_text().splitlines() == ['websight|none|0|4096', 'design2code-sat|none|0|4096']
    (tmp_path/'calls').unlink()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    result = subprocess.run(['/bin/bash',str(script),'data','out','unused'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode == 2 and 'requires two distinct GPUs' in result.stderr
    assert not (tmp_path/'calls').exists()


def test_paper_run_rejects_unknown_dataset_budget(tmp_path):
    import os
    import subprocess
    env = dict(os.environ, WEB_DATASET='unknown')
    result = subprocess.run(['/bin/bash', 'scripts/run_paper_experiments.sh', 'data', str(tmp_path/'out'), 'train'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'WEB_DATASET must be' in result.stderr
    assert not (tmp_path/'out').exists()

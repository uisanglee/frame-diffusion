"""Path plumbing only: no model loading, data preparation or GPU execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def clean_env():
    env = dict(os.environ)
    for key in ('TRAIN_RUN', 'TRAIN_DATA', 'DETECTOR_CHECKPOINT', 'POLICY_RUN_DIR',
                'POLICY_MODES', 'PARENT_FIRST', 'ABSTRACTION_REPRESENTATION',
                'RAW_CHECKPOINT', 'TUIDE_S_CHECKPOINT', 'TUIDE_M_CHECKPOINT',
                'TUIDE_L_CHECKPOINT', 'OFFICIAL_REPO'):
        env.pop(key, None)
    return env


def launch_training(tmp_path, args, overrides):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copyfile(SCRIPTS / 'train_tuide_scale.sh', scripts / 'train_tuide_scale.sh')
    (scripts / 'train_numeric_policy.sh').write_text(
        'printf "%s\\n" "$POLICY_RUN_DIR" "$@"\n')
    result = subprocess.run(['bash', str(scripts / 'train_tuide_scale.sh'), *args],
        env={**clean_env(), **overrides}, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()[-4:]


@pytest.mark.parametrize('scale', ['s', 'm', 'l'])
def test_training_css_first_defaults(tmp_path, scale):
    run = 'runs/webui-css-mixed-v1/7000'
    assert launch_training(tmp_path, [scale], {}) == [
        f'{run}/tuide-{scale}-abstract', 'data/new-css-v1/7000/css',
        f'{run}/detector/best.pt', f'{run}/shared']


def test_training_environment_paths_and_representation_suffix(tmp_path):
    assert launch_training(tmp_path, ['m'], dict(TRAIN_RUN='custom/run',
        TRAIN_DATA='custom/data', DETECTOR_CHECKPOINT='custom/detector.pt',
        PARENT_FIRST='1', ABSTRACTION_REPRESENTATION='occupancy')) == [
            'custom/run/tuide-m-abstract-parent-first-occupancy',
            'custom/data', 'custom/detector.pt', 'custom/run/shared']


def test_explicit_training_paths_override_environment(tmp_path):
    args = ['s', 'positional/data', 'positional/detector',
            'positional/shared', 'positional/output']
    assert launch_training(tmp_path, args, dict(TRAIN_RUN='env/run',
        TRAIN_DATA='env/data', DETECTOR_CHECKPOINT='env/detector',
        POLICY_RUN_DIR='env/output')) == [
            'positional/output', 'positional/data',
            'positional/detector', 'positional/shared']


def test_policy_output_environment_is_kept_without_positional_output(tmp_path):
    assert launch_training(tmp_path, ['s'], {'POLICY_RUN_DIR':'existing/output'})[0] == 'existing/output'


@pytest.mark.parametrize('custom', [False, True])
def test_ablation_training_paths_follow_current_defaults_and_overrides(tmp_path, custom):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copyfile(SCRIPTS / 'train_webui_color_ablation.sh', scripts / 'train_webui_color_ablation.sh')
    (scripts / 'train_tuide_scale.sh').write_text('printf "%s\\n" "$@"\n')
    run = 'custom/run' if custom else 'runs/webui-css-mixed-v1/7000'
    data = 'custom/data' if custom else 'data/new-css-v1/7000/css'
    env = clean_env()
    if custom:
        env.update(TRAIN_RUN=run, TRAIN_DATA=data)
    result = subprocess.run(['bash', str(scripts / 'train_webui_color_ablation.sh'),
                             'screenshot', '0'], env=env, capture_output=True,
                            text=True, check=True)
    assert result.stdout.splitlines() == ['s', data, f'{run}/detector/best.pt',
                                         f'{run}/shared', f'{run}/tuide-s-screenshot']


@pytest.mark.parametrize('explicit', [False, True])
def test_scale_evaluation_uses_train_run_and_checkpoint_overrides(tmp_path, explicit):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copyfile(SCRIPTS / 'run_tuide_scales.sh', scripts / 'run_tuide_scales.sh')
    run = tmp_path / 'training'
    paths = {'DETECTOR_CHECKPOINT': run / 'detector/best.pt',
             'RAW_CHECKPOINT': run / 'tuide-s-screenshot/replacement/raw-stage1/best.pt'}
    for scale in ('s', 'm', 'l'):
        paths[f'TUIDE_{scale.upper()}_CHECKPOINT'] = run / f'tuide-{scale}-abstract/replacement/abstract-stage2/best.pt'
    if explicit:
        paths = {key: tmp_path / 'overrides' / f'{key}.pt' for key in paths}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    data = tmp_path / 'prepared.jsonl'
    data.write_text('{}\n')
    binary = tmp_path / 'bin'
    binary.mkdir()
    stub = binary / 'python'
    log = tmp_path / 'calls.jsonl'
    stub.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
        'with open(os.environ["TEST_CALLS"],"a") as f:\n'
        '    f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    stub.chmod(0o755)
    env = {**clean_env(), 'TRAIN_RUN': str(run), 'TEST_CALLS': str(log),
           'PATH': str(binary) + os.pathsep + os.environ['PATH']}
    if explicit:
        env.update({key: str(path) for key, path in paths.items()})
    subprocess.run(['bash', str(scripts / 'run_tuide_scales.sh'), str(data),
                    str(tmp_path / 'evaluation')], env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    repairs = [call for call in calls if 'visual-evaluate' in call]
    assert len(repairs) == 3
    for scale, call in zip(('s', 'm', 'l'), repairs):
        assert call[call.index('--abstract-checkpoint') + 1] == str(paths[f'TUIDE_{scale.upper()}_CHECKPOINT'])
        assert call[call.index('--detector-checkpoint') + 1] == str(paths['DETECTOR_CHECKPOINT'])
        if scale == 's':
            assert call[call.index('--raw-checkpoint') + 1] == str(paths['RAW_CHECKPOINT'])
        else:
            assert '--abstract-only' in call

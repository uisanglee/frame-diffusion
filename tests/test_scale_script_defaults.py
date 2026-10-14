"""Exercise shell argument/env plumbing without launching training or a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULTS = dict(STAGE='all', POLICY_SCALE='s', POLICY_MODES='abstract screenshot',
    BATCH_SIZE='2', ACCUMULATION='4', DETECTOR_STEPS='20000', DETECTOR_LR='0.0001',
    POLICY_STAGE1_STEPS='30000', POLICY_STAGE2_STEPS='15000', POLICY_LR='0.0001',
    POLICY_STAGE2_LR='0.00001', VAL_SAMPLES='2000', EVAL_EVERY='500',
    EARLY_STOP_PATIENCE='10', EARLY_STOP_MIN_DELTA='0', ONLINE_WORKERS='2',
    ONLINE_PREFETCH='4', PARENT_FIRST='0', SEED='42', REPAIR_STEPS='20',
    REPEATS='3', ROLLOUT_WORKERS='4', ROLLOUT_BATCH_SIZE='4', ROLLOUT_BATCH_WAIT_MS='10')


def run_script(tmp_path, overrides):
    scripts=tmp_path/'scripts';scripts.mkdir()
    source=Path(__file__).resolve().parents[1]/'scripts/run_webui_scale.sh'
    shutil.copyfile(source,scripts/source.name)
    (scripts/'train_tuide_scale.sh').write_text('python policy-fixture "$@"\n')
    binary=tmp_path/'bin';binary.mkdir()
    stub=binary/'python'
    stub.write_text(f'#!{sys.executable}\n'+'''import json,os,sys
if sys.argv[1]=='-c':
    print(7000)
else:
    with open(os.environ['TEST_LOG'],'a') as f:
        f.write(json.dumps({'args':sys.argv[1:],'env':{k:os.environ.get(k) for k in json.loads(os.environ['TEST_ENV_KEYS'])}})+'\\n')
''')
    stub.chmod(0o755)
    data=tmp_path/'data';(data/'7000').mkdir(parents=True)
    (data/'7000/coverage.json').write_text('{}')
    env={k:v for k,v in os.environ.items() if k not in DEFAULTS}
    env.update(PATH=str(binary)+os.pathsep+env['PATH'],TEST_LOG=str(tmp_path/'calls.jsonl'),
               TEST_ENV_KEYS=json.dumps(list(DEFAULTS)))
    env.update(overrides)
    result=subprocess.run(['bash',str(scripts/source.name),'7000',str(data),str(tmp_path/'runs')],
                          env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return [json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]


def test_shared_defaults_reach_all_stages(tmp_path):
    calls=run_script(tmp_path,{})
    detector=next(c for c in calls if 'visual-train-detector' in c['args'])
    assert all(detector['env'][k]==v for k,v in DEFAULTS.items())
    policies=[c for c in calls if c['args'][0]=='policy-fixture']
    assert [c['env']['POLICY_MODES'] for c in policies]==['abstract','screenshot']
    for call in policies:
        assert all(call['env'][k]==v for k,v in DEFAULTS.items() if k!='POLICY_MODES')
    rollouts=[c for c in calls if 'visual-test-rollout' in c['args']]
    assert len(rollouts)==4
    for call in rollouts:
        args=call['args']
        assert args[args.index('--repeats')+1]=='3'
        assert args[args.index('--workers')+1]=='4'


def test_explicit_overrides_are_preserved(tmp_path):
    overrides=dict(STAGE='evaluate',POLICY_SCALE='m',POLICY_MODES='abstract',
                   VAL_SAMPLES='0',REPEATS='1',ROLLOUT_WORKERS='8',ROLLOUT_BATCH_SIZE='2')
    calls=run_script(tmp_path,overrides)
    assert len(calls)==2
    for call in calls:
        assert all(call['env'][k]==v for k,v in overrides.items())
        args=call['args']
        assert args[args.index('--workers')+1]=='8'
        assert args[args.index('--inference-batch-size')+1]=='2'
        assert args[args.index('--repeats')+1]=='1'
        assert 'tuide-m-abstract' in args[args.index('--abstract-checkpoint')+1]

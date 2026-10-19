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


def test_resume_launcher_fresh_resume_and_preserved_output(tmp_path):
    scripts=tmp_path/'scripts';scripts.mkdir()
    source=Path(__file__).resolve().parents[1]/'scripts/resume_webui_training.sh'
    shutil.copyfile(source,scripts/source.name)
    (scripts/'train_tuide_scale.sh').write_text('echo screenshot-stub\n')
    binary=tmp_path/'bin';binary.mkdir()
    stub=binary/'python';stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');stub.chmod(0o755)
    env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH']}
    for mode,size,output in (
        ('detector-7k',7000,'runs/webui-detector-full-dom-v1/7000/detector'),
        ('detector-56169',56169,'runs/webui-detector-full-dom-v1/56169/detector'),
        ('screenshot-7k',7000,'runs/webui-screenshot-single-v1/7000/tuide-s-screenshot/replacement/raw-stage1')):
        data=tmp_path/f'data/webui-detector-full-dom-v1/{size}/rendered'
        names=['detector-train.jsonl','detector-val.jsonl']
        if mode=='screenshot-7k':
            data=tmp_path/'data/webui-7k-56169-safe-margin-v1/7000/css'
            names=['policy-train.jsonl','policy-val.jsonl','pages-train.jsonl']
        data.mkdir(parents=True,exist_ok=True)
        for name in names:(data/name).write_text('{}\n')
        def run():
            return subprocess.run(['bash',str(scripts/source.name),mode],env=env,check=True,capture_output=True,text=True).stdout
        assert 'Starting fresh' in run()
        out=tmp_path/output;out.mkdir(parents=True)
        (out/'config.json').write_text('old-config')
        assert 'Previous output preserved' in run()
        backups=list(out.parent.glob(out.name+'.backup.*/run/config.json'))
        assert len(backups)==1 and backups[0].read_text()=='old-config'
        out.mkdir();(out/'last.pt').write_text('checkpoint')
        text=run()
        assert 'Resuming:' in text and 'Starting fresh' not in text
        if mode.startswith('detector'):assert '--resume\n' in text
        assert (out/'last.pt').read_text()=='checkpoint'


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


def test_screenshot_all_skips_detector_and_evaluates_single_stage(tmp_path):
    calls=run_script(tmp_path,dict(POLICY_MODES='screenshot'))
    assert len(calls)==3  # one policy run and own/common rollout
    assert calls[0]['args'][0]=='policy-fixture'
    for call in calls[1:]:
        args=call['args']
        assert args[args.index('--raw-checkpoint')+1].endswith('/replacement/raw-stage1/best.pt')
        assert '--detector-checkpoint' not in args


def test_numeric_policy_single_screenshot_and_two_abstract_stages(tmp_path):
    binary=tmp_path/'bin';binary.mkdir()
    stub=binary/'python'
    stub.write_text(f'#!{sys.executable}\n'+'''import json,os,sys
with open(os.environ['TEST_LOG'],'a') as f:
    f.write(json.dumps(sys.argv[1:])+'\\n')
''')
    stub.chmod(0o755)
    shared=tmp_path/'shared'
    for split in ('train','val'):
        folder=shared/f'predicted-{split}';folder.mkdir(parents=True)
        (folder/'data.jsonl').write_text('{}\n')
    detector=tmp_path/'detector.pt';detector.touch()
    output=tmp_path/'policy'
    resume=output/'replacement/raw-stage1/last.pt';resume.parent.mkdir(parents=True);resume.touch()
    env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],
             TEST_LOG=str(tmp_path/'calls.jsonl'),POLICY_MODES='screenshot abstract',
             POLICY_HEADS='replacement',POLICY_RUN_DIR=str(output),PREPARE_DATA='0',
             POLICY_STAGE1_STEPS='100',POLICY_STAGE2_STEPS='50',POLICY_SCREENSHOT_STEPS='120',
             EARLY_STOP_MIN_DELTA='0.003')
    script=Path(__file__).resolve().parents[1]/'scripts/train_numeric_policy.sh'
    result=subprocess.run(['bash',str(script),str(tmp_path/'data'),str(detector),str(shared)],
                          env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    calls=[json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert len(calls)==3
    assert [Path(c[c.index('--out')+1]).name for c in calls]==['raw-stage1','abstract-stage1','abstract-stage2']
    assert [c[c.index('--steps')+1] for c in calls]==['120','100','50']
    assert calls[0][calls[0].index('--resume')+1]==str(resume)
    assert '--predicted-targets' not in calls[0] and '--init-checkpoint' not in calls[0]
    assert '--predicted-targets' in calls[2] and '--init-checkpoint' in calls[2]
    assert all(c[c.index('--early-stop-min-delta')+1]=='0.003' for c in calls)

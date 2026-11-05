"""Isolated, persistent SAT runtime for the official Design2Code-18B checkpoint.

SAT 0.4.10 is deliberately not imported into the main evaluation environment.
The model card defines screenshot-only generation, not a self-revision protocol.
"""
import atexit
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

MODEL_ID = 'SALT-NLP/Design2Code-18B-v0'
OFFICIAL_SOURCE = 'https://huggingface.co/SALT-NLP/Design2Code-18B-v0'
OFFICIAL_SETTINGS = dict(max_length=4096, temperature=0.5, top_p=1.0,
                         top_k=1, repetition_penalty=1.1)
CONTRACT = 'design2code18b-sat-image-only-output-budget-html-stop-v2'
DEFAULT_MAX_NEW_TOKENS = 16384


def output_budget(args):
    budget = int(getattr(args, 'max_new_tokens', DEFAULT_MAX_NEW_TOKENS))
    if budget <= 0:
        raise ValueError('max_new_tokens must be positive')
    return budget


def generation_settings(prompt_tokens, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    if prompt_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError('Input length and output budget must be positive')
    # SAT chat accepts total max_length, not Hugging Face max_new_tokens.
    return {**OFFICIAL_SETTINGS, 'max_length': prompt_tokens + max_new_tokens}


def validate_request(args, images):
    output_budget(args)
    if len(images) != 1 or getattr(args, 'task', '').startswith('revise'):
        raise ValueError('Design2Code-18B official protocol supports one target screenshot only; self-revision is not supported')
    if getattr(args, 'temperature', .5) != .5:
        raise ValueError('Design2Code-18B official temperature is 0.5')
    if getattr(args, 'four_bit', False):
        raise ValueError('Design2Code-18B official SAT runtime uses BF16, not 4-bit')


def runtime_settings(max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    return dict(contract=CONTRACT, official_example_generation=OFFICIAL_SETTINGS.copy(), query='',
                max_new_tokens=max_new_tokens, html_completion_stop=True,
                text_processor_length=2048, weight_precision='bf16', model_parallel_size=2,
                source=OFFICIAL_SOURCE)


def checkpoint_directory(snapshot, revision):
    """The official Hub repository distributes a ZIP, not HF tensor shards."""
    snapshot = Path(snapshot)
    if (snapshot/'model_config.json').is_file():
        return snapshot
    archive = snapshot/'design2code-18b-v0.zip'
    if not archive.is_file():
        raise ValueError(f'Official SAT checkpoint archive missing: {archive}')
    cache = Path(os.environ.get('DESIGN2CODE_CHECKPOINT_CACHE', '.cache/design2code-sat/checkpoints')).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/revision
    import fcntl
    with (cache/(revision+'.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        marker = target/'checkpoint-path.json'
        if marker.exists():
            result = target/json.loads(marker.read_text())['relative_path']
            if (result/'model_config.json').is_file():
                return result
            raise ValueError(f'Incomplete extracted SAT checkpoint: {result}')
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            for member in members:
                path = Path(member.filename)
                if path.is_absolute() or '..' in path.parts or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError('Unsafe path in SAT checkpoint ZIP')
            needed = sum(m.file_size for m in members)
            if shutil.disk_usage(cache).free < needed:
                raise ValueError(f'Not enough disk space to extract SAT checkpoint ({needed} bytes required)')
            stage = Path(tempfile.mkdtemp(prefix=revision+'.partial-', dir=cache))
            try:
                zipped.extractall(stage)
                configs = list(stage.rglob('model_config.json'))
                if len(configs) != 1:
                    raise ValueError('SAT archive must contain exactly one model_config.json')
                relative = str(configs[0].parent.relative_to(stage))
                (stage/'checkpoint-path.json').write_text(json.dumps({'relative_path': relative}))
                stage.rename(target)
            except BaseException:
                shutil.rmtree(stage, ignore_errors=True)
                raise
        return target/relative


class Design2CodeRuntime:
    def __init__(self, args):
        validate_request(args, [None])
        if args.model != MODEL_ID:
            raise ValueError(f'Design2Code SAT backend requires {MODEL_ID}')
        repo = Path(os.environ.get('DESIGN2CODE_REPO', 'external/design2code-official')).resolve()
        if (repo/'CogVLM').is_dir():
            repo = repo/'CogVLM'
        if not (repo/'utils/models').is_dir():
            raise ValueError('Set DESIGN2CODE_REPO to the Design2Code repository (its vendored CogVLM is required)')
        python = os.environ.get('DESIGN2CODE_PYTHON', str(Path('baselines/design2code/.venv/bin/python').resolve()))
        if not python or not shutil.which(python):
            raise ValueError('Set DESIGN2CODE_PYTHON to the Python executable in the separate SAT 0.4.10 environment')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if len(visible) != 2 or any(not value.strip() for value in visible):
            raise ValueError('Design2Code-18B requires exactly two visible GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1')
        from huggingface_hub import snapshot_download
        print(json.dumps({'design2code_stage': 'checkpoint_download', 'status': 'started',
                          'model': MODEL_ID, 'overall_timeout_s': None}), flush=True)
        snapshot = Path(snapshot_download(MODEL_ID, revision=args.revision)).resolve()
        print(json.dumps({'design2code_stage': 'checkpoint_download', 'status': 'completed'}), flush=True)
        self.revision = snapshot.name
        print(json.dumps({'design2code_stage': 'checkpoint_extract', 'status': 'started'}), flush=True)
        checkpoint = checkpoint_directory(snapshot, self.revision)
        print(json.dumps({'design2code_stage': 'checkpoint_extract', 'status': 'completed'}), flush=True)
        self.timeout = float(os.environ.get('DESIGN2CODE_TIMEOUT', '1800'))
        self.directory = Path(tempfile.mkdtemp(prefix='framediff-design2code-'))
        log_dir = Path(os.environ.get('DESIGN2CODE_LOG_DIR', '.cache/design2code-sat')).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = log_dir/f'{uuid.uuid4().hex}.log'
        worker = Path(__file__).with_name('design2code_sat_worker.py')
        command = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   str(worker), '--repo', str(repo), '--checkpoint', str(checkpoint),
                   '--requests', str(self.directory), '--revision', self.revision, '--parent-pid', str(os.getpid())]
        with self.log.open('w') as log:
            self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.close)
        try:
            print(json.dumps({'design2code_stage': 'model_load', 'status': 'started',
                              'timeout_s': self.timeout}), flush=True)
            self._wait(self.directory/'ready.json')
            print(json.dumps({'design2code_stage': 'model_load', 'status': 'completed'}), flush=True)
        except BaseException:
            self.close()
            raise

    def _wait(self, path):
        start = time.monotonic()
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'Design2Code SAT worker exited ({self.process.returncode}); inspect {self.log}')
            if time.monotonic()-start > self.timeout:
                self.close()
                raise TimeoutError(f'Design2Code SAT exceeded {self.timeout}s; worker stopped. Log: {self.log}')
            time.sleep(.1)
        data = json.loads(path.read_text())
        if data.get('error'):
            raise RuntimeError(f"Design2Code SAT: {data['error']}. Log: {self.log}")
        return data

    def generate(self, args, images):
        validate_request(args, images)
        image = self.directory/'input.png'
        images[0].save(image)
        name = uuid.uuid4().hex
        request = dict(id=name, image=str(image), seed=int(args.seed), max_new_tokens=output_budget(args))
        temp = self.directory/'request.tmp'
        temp.write_text(json.dumps(request))
        temp.replace(self.directory/'request.json')
        response = self._wait(self.directory/f'{name}.json')
        (self.directory/f'{name}.json').unlink()
        return response['answer'], response['meta']

    def close(self):
        proc = getattr(self, 'process', None)
        if proc is not None and proc.poll() is None:
            # The whole torchrun process group must exit; no orphan GPU workers.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        directory = getattr(self, 'directory', None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        atexit.unregister(self.close)


def load_design2code_runtime(args):
    return Design2CodeRuntime(args)


def design2code_generate(args, prompt, images, runtime=None):
    validate_request(args, images)
    if prompt.strip():
        raise ValueError('Design2Code official inference requires an empty query')
    # Empty query is intentional: the official model is trained for screenshot-to-code.
    # The caller records this actual protocol, not the generic chat prompt.
    own_runtime = runtime is None
    runtime = runtime or load_design2code_runtime(args)
    try:
        return runtime.generate(args, images)
    finally:
        if own_runtime:
            runtime.close()

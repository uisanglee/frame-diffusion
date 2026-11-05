"""Official image-only WebSight inference, isolated from modern VLM dependencies.

Source: https://huggingface.co/HuggingFaceM4/VLM_WebSight_finetuned
The published protocol has no instruction/chat or self-revision input. In
particular, its 4096 limit counts prompt AND output, unlike max_new_tokens.
"""
import atexit
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

MODEL_ID = 'HuggingFaceM4/VLM_WebSight_finetuned'
CONTRACT = 'websight-official-image-only-960-bilinear-maxlength4096-v1'
SOURCE = 'https://huggingface.co/HuggingFaceM4/VLM_WebSight_finetuned'
MAX_LENGTH = 4096


def convert_to_rgb(image):
    from PIL import Image
    if image.mode == 'RGB':
        return image
    rgba = image.convert('RGBA')
    return Image.alpha_composite(Image.new('RGBA', rgba.size, (255,255,255)), rgba).convert('RGB')


def official_prefix(processor, model):
    length = model.config.perceiver_config.resampler_n_latents
    return processor.tokenizer.bos_token + '<fake_token_around_image>' + '<image>' * length + '<fake_token_around_image>'


def official_inputs(processor, model, image):
    import torch
    from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
    from transformers.image_transforms import resize, to_channel_dimension_format
    def transform(x):
        x = to_numpy_array(convert_to_rgb(x))
        x = resize(x, (960,960), resample=PILImageResampling.BILINEAR)
        x = processor.image_processor.rescale(x, scale=1/255)
        x = processor.image_processor.normalize(x, mean=processor.image_processor.image_mean,
                                               std=processor.image_processor.image_std)
        return torch.tensor(to_channel_dimension_format(x, ChannelDimension.FIRST))
    inputs = processor.tokenizer(official_prefix(processor, model), return_tensors='pt', add_special_tokens=False)
    inputs['pixel_values'] = processor.image_processor([image], transform=transform)
    return inputs


def validate_request(args, images):
    if args.model != MODEL_ID:
        raise ValueError(f'WebSight backend requires {MODEL_ID}')
    if getattr(args, 'four_bit', False):
        raise ValueError('Official WebSight inference uses BF16; four-bit is not enabled')
    if getattr(args, 'task', 'generate-html') != 'generate-html' or getattr(args, 'current', None) or len(images) != 1:
        raise ValueError('WebSight official protocol supports single-image Initial generation only; self-revision is unavailable')


class WebSightRuntime:
    """One long-lived model process; WEBSIGHT_PYTHON may select a legacy venv."""
    def __init__(self, args):
        self.work = tempfile.TemporaryDirectory(prefix='framediff-websight-')
        self.root = Path(self.work.name)
        self.process = None
        self.counter = 0
        self.timeout = float(os.environ.get('WEBSIGHT_TIMEOUT', '1800'))
        if self.timeout <= 0:
            raise ValueError('WEBSIGHT_TIMEOUT must be positive')
        settings = dict(model=args.model, revision=args.revision, seed=args.seed, parent_pid=os.getpid())
        (self.root/'settings.json').write_text(json.dumps(settings))
        python = os.environ.get('WEBSIGHT_PYTHON', str(Path(__file__).resolve().parents[1]/'baselines/websight/.venv/bin/python'))
        if not Path(python).is_file():
            self.work.cleanup()
            raise RuntimeError('WebSight environment is missing; run bash baselines/websight/setup.sh or set WEBSIGHT_PYTHON')
        try:
            self.process = subprocess.Popen([python, str(Path(__file__).resolve()), '--worker', str(self.root)],
                                            start_new_session=True)
            self._wait(self.root/'ready.json')
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def _wait(self, path):
        deadline = time.monotonic() + self.timeout
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'WebSight worker exited ({self.process.returncode}); see worker stderr. '
                                   'Set WEBSIGHT_PYTHON to the dedicated WebSight environment.')
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError(f'WebSight worker exceeded {self.timeout}s')
            time.sleep(.1)
        response = json.loads(path.read_text())
        if response.get('error'):
            raise RuntimeError('WebSight worker: ' + response['error'])
        return response

    def generate(self, args, image):
        self.counter += 1
        image_path = self.root/f'image-{self.counter}.png'
        request = self.root/f'request-{self.counter}.json'
        response = self.root/f'response-{self.counter}.json'
        image.save(image_path)
        _write_json(request, {'image': str(image_path), 'seed': args.seed})
        try:
            result = self._wait(response)
            return result['text'], result['metadata']
        finally:
            image_path.unlink(missing_ok=True)
            request.unlink(missing_ok=True)
            response.unlink(missing_ok=True)

    def close(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
        self.work.cleanup()


def load_websight_runtime(args):
    if args.model != MODEL_ID or getattr(args, 'four_bit', False):
        raise ValueError('Official WebSight runtime requires the WebSight checkpoint in BF16')
    temperature = getattr(args, 'initial_temperature', getattr(args, 'temperature', 0.))
    if temperature not in ('auto', None) and float(temperature) != 0.:
        raise ValueError('Official WebSight generation does not override sampling temperature; use auto/0')
    return WebSightRuntime(args)


def websight_generate(args, prompt, images, runtime=None):
    validate_request(args, images)
    if prompt.strip():
        raise ValueError('Official WebSight input is image-only; do not inject a chat/instruction prompt')
    owned = runtime is None
    runtime = runtime or load_websight_runtime(args)
    try:
        return runtime.generate(args, images[0])
    finally:
        if owned:
            runtime.close()


def _write_json(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value))
    pending.replace(path)


def worker(root):
    # This file is executable directly in an isolated environment. Do not import
    # framediff: its current Transformers requirements need not match VMistral.
    settings = json.loads((root/'settings.json').read_text())
    parent_pid = settings['parent_pid']
    # Kill this worker if its controlling experiment exits, including SIGTERM.
    # atexit alone is insufficient for a background process using a new session.
    if sys.platform == 'linux':
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), 'Cannot install WebSight parent-death signal')
    if os.getppid() != parent_pid:
        return
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError('WebSight requires a CUDA GPU')
    device = torch.device('cuda:0')
    torch.manual_seed(settings['seed'])
    processor = AutoProcessor.from_pretrained(settings['model'], revision=settings['revision'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(settings['model'], revision=settings['revision'],
        trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    if not hasattr(model, 'generate'):
        raise RuntimeError('Custom VMistral generation is incompatible with this Transformers version; '
                           'use WEBSIGHT_PYTHON with a dedicated legacy environment (e.g. transformers==4.37.2)')
    bad_words = processor.tokenizer(['<image>', '<fake_token_around_image>'], add_special_tokens=False).input_ids
    _write_json(root/'ready.json', {'ready': True})
    counter = 1
    while os.getppid() == parent_pid:
        path = root/f'request-{counter}.json'
        if not path.exists():
            time.sleep(.05)
            continue
        try:
            request = json.loads(path.read_text())
            with Image.open(request['image']) as source:
                image = source.copy()
            inputs = official_inputs(processor, model, image)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_tokens = inputs['input_ids'].shape[1]
            torch.manual_seed(request['seed'])
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, bad_words_ids=bad_words, max_length=MAX_LENGTH)
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - start
            text = processor.batch_decode(outputs[:, prompt_tokens:], skip_special_tokens=True)[0]
            generation = model.generation_config.to_dict()
            generation.update(max_length=MAX_LENGTH, bad_words_ids=bad_words)
            metadata = dict(generation_contract=CONTRACT, official_source=SOURCE,
                effective_prompt=official_prefix(processor, model), prompt_protocol='image-only',
                image_preprocessing={'size':[960,960], 'interpolation':'bilinear', 'alpha_background':'white'},
                generation_settings=generation, generation_seconds=duration,
                input_tokens=prompt_tokens, output_tokens=outputs.shape[1]-prompt_tokens,
                weight_precision='bf16', device=str(device), gpu_name=torch.cuda.get_device_name(device),
                resolved_revision=getattr(model.config, '_commit_hash', None) or settings['revision'],
                transformers_version=transformers.__version__, torch_version=torch.__version__,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                placement_contract='websight-single-cuda-no-offload-v1',
                requested_max_new_tokens_ignored=True)
            result = {'text': text, 'metadata': metadata}
            del inputs, outputs
        except Exception as error:
            result = {'error': f'{type(error).__name__}: {error}'}
        _write_json(root/f'response-{counter}.json', result)
        counter += 1


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('Internal WebSight worker; use the paper experiment scripts')
    directory = Path(sys.argv[2])
    try:
        worker(directory)
    except Exception as error:
        _write_json(directory/'ready.json', {'error': f'{type(error).__name__}: {error}'})
        raise

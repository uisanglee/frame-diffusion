"""Isolated UI2CodeN polishing adapter; imports, but never edits, framediff.

Inference has no access to reference HTML/boxes. Evaluation uses web-evaluate.
"""
import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framediff.html_bridge import HtmlBrowser, html_answer

MODEL = 'zai-org/UI2Code_N'
PROMPT = '''The first image is the target UI design. The second image is the current
rendering of the HTML below. Refine the current implementation to match the target.
Return only one complete HTML document with CSS, from <html> to </html> (a doctype
is allowed). Do not return an explanation or a patch. Do not fetch external assets.
'''
LAYOUT = '''This is a layout repair task: preserve the existing text, images, fonts,
colors, DOM elements and their hierarchy. Adjust only CSS layout, position, size
and spacing. Keep existing resource URLs and data URIs unchanged.
'''


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def load_records(path, path_root, limit):
    records = []
    seen = set()
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            raw = json.loads(line)
            record = {k: raw[k] for k in ('id', 'page_id', 'viewport', 'repeat',
                'reference_kind', 'reference_boxes', 'reference_box_error',
                'external_source_model') if k in raw}
            record['id'] = str(record['id'])
            if record['id'] in seen: raise ValueError('Duplicate id: ' + record['id'])
            seen.add(record['id'])
            initial = raw.get('methods', {}).get('initial', {})
            paths = dict(screenshot=raw.get('screenshot') or raw.get('target_image'),
                         initial_html=initial.get('html') or raw.get('initial_html') or raw.get('current_html'),
                         html=raw.get('html') or raw.get('target_html'))
            for key, value in paths.items():
                if not value:
                    if key != 'html': raise ValueError(f'{record["id"]}: missing {key}')
                    continue
                p = Path(value)
                p = (path_root / p).resolve() if not p.is_absolute() else p.resolve()
                if not p.is_file(): raise FileNotFoundError(p)
                record[key] = str(p)
            from PIL import Image
            with Image.open(record['screenshot']) as im: size = list(im.size)
            record.setdefault('viewport', size)
            if record['viewport'] != size: raise ValueError('Target image and viewport sizes differ')
            record['initial_metadata'] = initial
            records.append(record)
            if limit and len(records) >= limit: break
    if not records: raise ValueError('Empty manifest')
    return records


class Runtime:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        if not torch.cuda.is_available(): raise RuntimeError('A CUDA GPU is required')
        if torch.cuda.device_count() != 1: raise RuntimeError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES')
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        free, _ = torch.cuda.mem_get_info()
        if free < 18 * 1024**3:
            raise RuntimeError('Less than 18 GiB free VRAM: use an idle RTX 4090 for this preset')
        self.args = args
        self.processor = AutoProcessor.from_pretrained(MODEL, revision=args.revision)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL,
            revision=args.revision, torch_dtype=torch.bfloat16, device_map={'': 0},
            attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        import transformers
        self.info = dict(model=MODEL, resolved_revision=getattr(self.model.config, '_commit_hash', None),
                         precision='nf4-double-quant/bf16', gpu=torch.cuda.get_device_name(0),
                         torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda)

    def generate(self, target, current, html):
        import torch
        from PIL import Image
        images = []
        for path in (target, current):
            with Image.open(path) as im:
                im = im.convert('RGB')
                ratio = min(1., math.sqrt(self.args.max_pixels / (im.width * im.height)))
                if ratio < 1: im = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))))
                images.append(im)
        prompt = PROMPT + (LAYOUT if self.args.protocol == 'layout' else '') + '\nCurrent HTML:\n' + html
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': images[0]}, {'type': 'image', 'image': images[1]},
            {'type': 'text', 'text': prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt')
        n = inputs['input_ids'].shape[1]
        if n + self.args.max_new_tokens > self.args.max_context:
            raise ValueError(f'Context budget exceeded: input={n}, output_budget={self.args.max_new_tokens}; no truncation applied')
        inputs = inputs.to('cuda:0')
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens,
                do_sample=False, use_cache=True)
        answer = self.processor.decode(ids[0, n:], skip_special_tokens=True)
        stats = dict(input_tokens=n, output_tokens=ids.shape[1]-n,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated())
        del inputs, ids
        return answer, stats


def run_page(record, folder, rounds, runtime, browser):
    folder.mkdir(parents=True, exist_ok=True)
    initial = dict(record['initial_metadata'])
    initial.update(html=record['initial_html'])
    for k, v in dict(failed=False, seconds=0., browser_executions=0, vlm_calls=0).items():
        initial.setdefault(k, v)
    result = {k: v for k, v in record.items() if k not in ('initial_metadata', 'initial_html')}
    result.update(vlm_model=MODEL, vlm_backend='huggingface', vlm_weight_precision='4bit-nf4')
    result['methods'] = {'initial': initial}
    current = Path(record['initial_html'])
    seconds = 0.; calls = tokens_in = tokens_out = renders = captures = failures = 0
    for step in range(1, rounds + 1):
        checkpoint = folder / f'round-{step}.json'
        if checkpoint.exists():
            method = json.loads(checkpoint.read_text())
            if sha(method['html']) != method['html_sha256']: raise ValueError('Cached output changed: ' + method['html'])
        else:
            started = time.perf_counter(); before = browser.executions
            before_captures = browser.screenshots
            error = None; stats = {}; answer = None
            output = current
            try:
                browser.reset_context()
                png = folder / f'input-{step}.png'
                html = current.read_text()
                browser.snapshot(html, record['viewport'], png, 2048)
                calls += 1
                answer, stats = runtime.generate(record['screenshot'], png, html)
                (folder / f'answer-{step}.txt').write_text(answer)
                repaired = html_answer(answer)
                # Validate renderability before adopting. No target scoring/rollback.
                browser.snapshot(repaired, record['viewport'], folder / f'output-{step}.png', 2048)
                output = folder / f'output-{step}.html'
                output.write_text(repaired)
            except Exception as exc:
                # CUDA failures are not safe to continue through in the same process.
                if 'cuda' in str(exc).lower() or 'out of memory' in str(exc).lower(): raise
                error = f'{type(exc).__name__}: {exc}'
                failures += 1
            seconds += time.perf_counter() - started
            renders += browser.executions - before
            captures += browser.screenshots - before_captures
            tokens_in += stats.get('input_tokens', 0); tokens_out += stats.get('output_tokens', 0)
            method = dict(html=str(output), html_sha256=sha(output), failed=bool(error), error=error,
                seconds=initial['seconds'] + seconds, repair_seconds=seconds,
                browser_executions=initial['browser_executions'] + renders,
                vlm_calls=initial['vlm_calls'] + calls, repair_vlm_calls=calls,
                repair_input_tokens=tokens_in, repair_output_tokens=tokens_out,
                candidate_failure_count=failures, feedback_mode='screenshot', target_source='screenshot',
                uses_reference_text=False, revision_protocol='ui2coden-local-polishing-v1',
                visual_timing=dict(seconds=seconds, browser_executions=renders,
                                   browser_screenshots=captures, actions=step-failures),
                generation=stats)
            save(checkpoint, method)
        current = Path(method['html'])
        seconds, calls = method['repair_seconds'], method['repair_vlm_calls']
        tokens_in, tokens_out = method['repair_input_tokens'], method['repair_output_tokens']
        renders = method['browser_executions'] - initial['browser_executions']
        captures = method['visual_timing']['browser_screenshots']
        failures = method['candidate_failure_count']
        result['methods'][f'ui2coden-round-{step}'] = method
        print(json.dumps(dict(id=record['id'], round=step, failed=method['failed'], repair_seconds=seconds)), flush=True)
    return result


def main():
    # Unwind context managers and close Chromium on ordinary termination.
    def terminate(signum, frame):
        raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--path-root', type=Path, default=ROOT)
    p.add_argument('--rounds', type=int, default=3); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--max-context', type=int, default=12288)
    p.add_argument('--revision', default='main'); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--protocol', choices=['layout', 'general'], default='layout')
    p.add_argument('--resume', action='store_true'); p.add_argument('--check-inputs', action='store_true')
    args = p.parse_args()
    if min(args.rounds, args.max_pixels, args.max_new_tokens, args.max_context) <= 0 or args.limit < 0:
        p.error('Budgets must be positive and limit nonnegative')
    records = load_records(args.manifest, args.path_root, args.limit)
    if args.check_inputs:
        print(json.dumps(dict(pages=len(records), reference_html=sum('html' in r for r in records))))
        return
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    # An exclusive lock prevents two processes mutating the same run; freed on exit/crash.
    import fcntl
    with open(out / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = dict(protocol='ui2coden-local-polishing-v1', settings={k: v for k, v in vars(args).items()
            if k not in ('resume', 'check_inputs', 'out', 'path_root')}, prompt=PROMPT, layout_prompt=LAYOUT,
            runner_sha=sha(__file__), records=records,
            files={r[k]: sha(r[k]) for r in records for k in ('screenshot', 'initial_html', 'html') if k in r})
        cfg = out / 'config.json'
        if cfg.exists():
            if not args.resume or json.loads(cfg.read_text()) != config:
                raise ValueError('Run/input/settings changed: use a new output directory')
        else:
            if any(out.iterdir()) and set(x.name for x in out.iterdir()) != {'.lock'}:
                raise ValueError('Refusing nonempty output directory without config.json')
            save(cfg, config)
        # Resolve once, then pin model/processor to the same immutable HF commit on resume.
        from huggingface_hub import HfApi
        pin = out / 'model-revision.json'
        if pin.exists(): args.revision = json.loads(pin.read_text())['sha']
        else:
            args.revision = HfApi().model_info(MODEL, revision=args.revision).sha
            save(pin, {'sha': args.revision})
        runtime = None
        results = []
        with HtmlBrowser() as browser:
            for record in records:
                folder = out / 'pages' / hashlib.sha256(record['id'].encode()).hexdigest()[:24]
                if not all((folder / f'round-{i}.json').exists() for i in range(1, args.rounds+1)):
                    if runtime is None:
                        runtime = Runtime(args); save(out / 'runtime.json', runtime.info)
                results.append(run_page(record, folder, args.rounds, runtime, browser))
        temporary = out / 'results.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
        os.replace(temporary, out / 'results.jsonl')


if __name__ == '__main__': main()

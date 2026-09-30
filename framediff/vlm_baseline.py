"""Precompute VLM revisions so the VLM and denoiser need not share GPU memory."""
import copy
import hashlib
import json
import time
from pathlib import Path

from . import vlm
from .browser import Browser
from .data import differences
from .ir import read_json, read_jsonl, validate, write_json, write_jsonl


def build_baseline(args):
    records = list(read_jsonl(args.data))
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError('Empty input dataset')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate record IDs')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    settings = {key: getattr(args, key) for key in ('backend','model','revision','endpoint','four_bit',
                'max_new_tokens','max_pixels','seed','rounds','input_mode')}
    results = []; statuses = []; runtime = None
    with Browser() as browser:
        for index, record in enumerate(records):
            mode = args.input_mode
            if mode == 'auto':
                mode = 'screenshot' if record.get('screenshot') else 'frames'
            if mode == 'screenshot':
                if not record.get('screenshot') or len(record['observations']) != 1:
                    raise ValueError('Screenshot mode requires a screenshot path and exactly one viewport')
                image_digest = hashlib.sha256(Path(record['screenshot']).read_bytes()).hexdigest()
            else:
                image_digest = None
            fingerprint = hashlib.sha256(json.dumps({'record':record, 'settings':settings,
                'image':image_digest}, sort_keys=True).encode()).hexdigest()
            work = out/'pages'/hashlib.sha256(record['id'].encode()).hexdigest()[:20]
            work.mkdir(parents=True, exist_ok=True)
            cache = work/'baseline.json'
            if args.resume and cache.exists():
                baseline = read_json(cache)
                if baseline['fingerprint'] != fingerprint:
                    raise ValueError('Resume input/config changed; use a new output directory')
            else:
                if args.backend == 'qwen' and runtime is None:
                    runtime = vlm.load_qwen_runtime(args)
                current = copy.deepcopy(record['current']); completed = 0; failure = None
                calls_before = browser.executions
                started = time.perf_counter()
                for step in range(args.rounds):
                    try:
                        current_path = work/f'current-{step}.json'; write_json(current_path, current)
                        rendered = []
                        for view, obs in enumerate(record['observations']):
                            path = work/f'current-{step}-{view}.png'
                            browser.render(current, obs['viewport'], path); rendered.append(str(path))
                        call = copy.copy(args); call.task = 'revise-ir'; call.current = str(current_path)
                        call.frames = None
                        if mode == 'frames':
                            # Only repair inputs, NEVER clean/reference/evaluation observations.
                            frames_path = work/'input-frames.json'
                            write_json(frames_path, record['observations']); call.frames = str(frames_path)
                            call.image = rendered
                            instruction = ('Images show the CURRENT frame layout, one per viewport in the '
                                           'same order as the frame observations. Target coordinates are in the JSON. ')
                        else:
                            call.image = [record['screenshot'], *rendered]
                            instruction = ('Image 1 is the TARGET screenshot. Image 2 is the CURRENT frame '
                                           'rendering, with diagnostic labels. Correct geometry toward Image 1. ')
                        prompt = instruction + vlm.prompt_for(call)
                        write_json(work/f'prompt-{step}.json', {'prompt':prompt,'images':call.image})
                        answer, meta = vlm.generate(call, prompt, vlm.images_for(call), runtime)
                        (work/f'answer-{step}.txt').write_text(answer)
                        write_json(work/f'meta-{step}.json', meta)
                        candidate = validate(vlm.json_answer(answer))
                        differences(current, candidate)  # Same fixed node topology as FrameDiff.
                        current = candidate; completed += 1
                    except Exception as error:
                        failure = f'{type(error).__name__}: {error}'
                        break  # Keep last valid output; never silently discard failed pages.
                baseline = {'fingerprint':fingerprint,'result':current,'seconds':time.perf_counter()-started,
                            'browser_executions':browser.executions-calls_before,'completed_rounds':completed,
                            'requested_rounds':args.rounds,'failure':failure,'input_mode':mode,
                            'status':'ok' if failure is None else 'failed', 'settings':settings}
                write_json(cache, baseline)
            updated = copy.deepcopy(record); updated['vlm_baseline'] = baseline; results.append(updated)
            statuses.append({'id':record['id'], 'status':baseline['status'],'failure':baseline['failure']})
            write_jsonl(out/'test.jsonl', results)
            write_json(out/'status.json', {'total':len(records),'processed':len(results),
                       'failed':sum(s['status']=='failed' for s in statuses),'records':statuses})
            print(f'[{index+1}/{len(records)}] {record["id"]} VLM {baseline["status"]}: '
                  f'{baseline["seconds"]:.3f}s ({mode})', flush=True)

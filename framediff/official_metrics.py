"""Use the upstream Design2Code scoring implementation, not a renamed IoU proxy.

Rendering uses our network-disabled Chromium. Exact pixel operations and bounded
memoization accelerate upstream scoring without changing metric definitions.
The evaluated viewport is explicitly controlled (unlike upstream's default viewport).
"""
import hashlib
import copy
from collections import OrderedDict
import importlib
import math
import shlex
import sys
import tempfile
import types
from contextlib import ExitStack
from functools import wraps
from pathlib import Path

EVALUATION_CONTRACT = 'preserve-html-no-repetition-removal-v1'


def preserve_html(filename):
    """Do not truncate CSS/text or delete repeated DOM elements during scoring."""
    if not Path(filename).is_file():
        raise FileNotFoundError(filename)


class OfficialMetrics:
    def __init__(self, repo):
        repo = Path(repo).resolve()
        path = repo/'Design2Code/metrics/visual_score.py'
        if not path.exists(): raise ValueError('official-repo must contain Design2Code/metrics/visual_score.py')
        source = path.read_text()
        self.source_sha = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((repo/'Design2Code').rglob('*.py')))).hexdigest()
        self.compatibility_patch = None
        self._block_cache = OrderedDict()
        self._init_caches()
        try:
            compile(source,str(path),'exec')
        except IndentationError:
            # Public upstream revision contains a misindented empty-block branch.
            # Repair ONLY that exact region, recording this compatibility change.
            begin = source.index('         if len(predict_blocks) == 0:')
            end = source.index('        if debug:',begin)
            block = source[begin:end].splitlines()
            fixed = []
            for line in block:
                stripped = line.lstrip()
                if stripped.startswith(('if len(predict_blocks)', 'elif len(original_blocks)')):
                    fixed.append('        '+stripped)
                elif stripped:
                    fixed.append('            '+stripped)
                else: fixed.append('')
            source = source[:begin]+'\n'.join(fixed)+'\n'+source[end:]
            compile(source,str(path),'exec')
            self.compatibility_patch = 'normalize upstream empty-block branch indentation (in memory only)'
        sys.path.insert(0,str(repo))
        try:
            self.ocr = importlib.import_module('Design2Code.metrics.ocr_free_utils')
            from . import official_pixel_metrics
            self._pixel_ops = official_pixel_metrics if official_pixel_metrics.supported(self.ocr) else self.ocr
            self.module = types.ModuleType('_framediff_design2code_metrics')
            self.module.__file__ = str(path)
            # Keep downloaded weights inside the workspace cache, not an implicit home path.
            source = source.replace('clip.load("ViT-B/32", device=device)',
                'clip.load("ViT-B/32", device=device, download_root=_framediff_clip_cache)')
            self.module._framediff_clip_cache = str(Path('.cache/clip').resolve())
            exec(compile(source,str(path),'exec'),self.module.__dict__)
            from .official_matching_cache import cached_text_similarity
            if hasattr(self.module, 'calculate_similarity'):
                self.module.calculate_similarity = cached_text_similarity(self.module.calculate_similarity)
        except ImportError as error:
            raise RuntimeError('Install the official metric dependencies in the evaluation environment; see README') from error
        finally:
            sys.path.remove(str(repo))

    def _init_caches(self):
        from .official_eval_cache import ByteLRU
        self._render_cache = ByteLRU()
        self._mask_cache = ByteLRU()
        self._feature_cache = ByteLRU(max_bytes=1024 * 1024, max_entries=32)

    def score(self, browser, predicted_html, reference_html, viewport, out, timing=None):
        from .ir import write_json
        from .evaluation_timing import EvaluationTiming
        from .official_eval_cache import cached_mask, CachedClipModel
        timing = timing or EvaluationTiming(output=str(out))
        # Upstream pre_process truncates repeated strings (including CSS) and
        # deletes repeated elements. Both alter valid rendered implementations.
        # Apply this override to every method, before installing timing hooks.
        self.module.pre_process = preserve_html
        if not hasattr(self, '_render_cache'):
            self._init_caches()
        if getattr(self, '_cache_browser', None) is not browser:
            self._init_caches()
            self._block_cache.clear()
            self._cache_browser = browser
        def measured(name, function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                with timing.stage(name):
                    return function(*args, **kwargs)
            return wrapped
        out = Path(out); out.mkdir(parents=True,exist_ok=True)
        # Upstream preprocessing mutates files. Never pass original dataset/artifacts.
        with tempfile.TemporaryDirectory(prefix='framediff-d2c-') as directory:
            root = Path(directory).resolve()
            pred = root/'pred.html'; ref = root/'ref.html'
            pred.write_text(predicted_html); ref.write_text(reference_html)

            def render(html, png):
                html, png = Path(html), Path(png)
                if not html.resolve().is_relative_to(root) or not png.resolve().is_relative_to(root):
                    raise ValueError('Official evaluator attempted an unexpected file access')
                content = html.read_text()
                key = (id(browser), tuple(viewport), self.source_sha,
                       hashlib.sha256(content.encode()).digest())
                found = self._render_cache.get(key)
                if found is not None:
                    png.write_bytes(found)
                    timing.event('official_render', 'cache_hit')
                    return
                browser.load(content,viewport)
                browser.page.screenshot(path=str(png),animations='disabled')
                browser.screenshots += 1
                if png.is_file():
                    raw = png.read_bytes()
                    self._render_cache.put(key, raw, len(raw))
            render = measured('official_render', render)

            def screenshot_command(command):
                # Interpret only upstream screenshot requests; never execute its shell.
                parts = shlex.split(command)
                if len(parts)!=6 or parts[0]!='python3' or Path(parts[1]).name!='screenshot_single.py' or parts[2]!='--html' or parts[4]!='--png':
                    raise ValueError('Unsupported upstream subprocess request')
                render(parts[3],parts[5]); return 0

            def blocks(image_path):
                html, ph, ph1, pp, pp1 = self.ocr.get_itermediate_names(image_path)
                # Cache extracted blocks, not scores: matching remains upstream.
                # Scoped to this scorer/browser runtime; no cross-machine font reuse.
                key = (id(browser), tuple(viewport), self.source_sha,
                       hashlib.sha256(Path(html).read_bytes()).hexdigest())
                if key in self._block_cache:
                    timing.event('block_extraction', 'cache_hit')
                    self._block_cache.move_to_end(key)
                    return copy.deepcopy(self._block_cache[key])
                with timing.stage('block_html_preprocess'):
                    self.ocr.process_html(html,ph)
                    self.ocr.process_html(html,ph1,offset=50)
                render(ph,pp); render(ph1,pp1)
                with timing.stage('block_pixel_extraction'):
                    pixel_ops = getattr(self, '_pixel_ops', self.ocr)
                    pixels = pixel_ops.find_different_pixels(pp,pp1)
                    if pixels is None:
                        result = []
                    else:
                        text = self.ocr.flatten_tree(self.ocr.extract_text_with_color(ph))
                        result = pixel_ops.get_blocks_from_image_diff_pixels(pp,text,pixels)
                timing.event('block_extraction', 'blocks_found', count=len(result))
                self._block_cache[key] = copy.deepcopy(result)
                while len(self._block_cache) > 16: self._block_cache.popitem(last=False)
                return result
            blocks = measured('block_extraction', blocks)

            self.module.os = types.SimpleNamespace(system=screenshot_command)
            self.module.get_blocks_ocr_free = blocks
            # Wrap actual upstream functions without changing their algorithms.
            # CLIP's .item() synchronizes its final similarity with the CPU.
            # Inpainting is nested under CLIP, so exclusive CLIP time excludes it.
            with ExitStack() as stack:
                mask = getattr(self.module, 'rescale_and_mask', None)
                if mask is not None and hasattr(self.module, 'model'):
                    stack.callback(setattr, self.module, 'rescale_and_mask', mask)
                    self.module.rescale_and_mask = cached_mask(mask, self._mask_cache, timing)
                model = getattr(self.module, 'model', None)
                if model is not None and hasattr(model, 'encode_image'):
                    stack.callback(setattr, self.module, 'model', model)
                    self.module.model = CachedClipModel(model, self._feature_cache, timing)
                for name, stage in (
                    ('pre_process', 'official_html_preprocess'),
                    ('find_possible_merge', 'block_matching'),
                    ('calculate_clip_similarity_with_blocks', 'clip'),
                    ('rescale_and_mask', 'clip_image_preprocess'),
                ):
                    function = getattr(self.module, name, None)
                    if function is not None:
                        stack.callback(setattr, self.module, name, function)
                        setattr(self.module, name, measured(stage, function))
                with timing.stage('official_other'):
                    result = self.module.visual_eval_v3_multi([[str(pred)],str(ref)])
            values = result[0][2]
            if len(values)!=5 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError('Invalid official metric output')
            scores = dict(zip(('official_block','official_text','official_position','official_color','official_clip'),map(float,values)))
            write_json(out/'official.json',{'scores':scores,'source_sha':self.source_sha,
                'evaluation_contract': EVALUATION_CONTRACT,
                'evaluation_timing':timing.report(),
                'optimizations': {
                    'exact_pixel_fast_path': getattr(self, '_pixel_ops', self.ocr) is not self.ocr,
                    'text_similarity_cache': hasattr(getattr(self.module, 'calculate_similarity', None), 'cache_info'),
                    'bounded_render_and_clip_caches': True,
                },
                'compatibility_patch':self.compatibility_patch,
                'protocol':'Upstream metric functions; shared fixed viewport; JS/external network disabled; offline assets; HTML repetition removal disabled'})
            return scores

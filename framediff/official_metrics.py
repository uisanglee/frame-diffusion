"""Use the upstream Design2Code scoring implementation, not a renamed IoU proxy.

Only rendering/process plumbing is replaced with our network-disabled Chromium.
All metric matching, merging, colors, CLIP and OCR-free extraction remain upstream.
The evaluated viewport is explicitly controlled (unlike upstream's default viewport).
"""
import hashlib
import importlib
import math
import shlex
import sys
import tempfile
import types
from pathlib import Path


class OfficialMetrics:
    def __init__(self, repo):
        repo = Path(repo).resolve()
        path = repo/'Design2Code/metrics/visual_score.py'
        if not path.exists(): raise ValueError('official-repo must contain Design2Code/metrics/visual_score.py')
        source = path.read_text()
        self.source_sha = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((repo/'Design2Code').rglob('*.py')))).hexdigest()
        self.compatibility_patch = None
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
            self.module = types.ModuleType('_framediff_design2code_metrics')
            self.module.__file__ = str(path)
            # Keep downloaded weights inside the workspace cache, not an implicit home path.
            source = source.replace('clip.load("ViT-B/32", device=device)',
                'clip.load("ViT-B/32", device=device, download_root=_framediff_clip_cache)')
            self.module._framediff_clip_cache = str(Path('.cache/clip').resolve())
            exec(compile(source,str(path),'exec'),self.module.__dict__)
        except ImportError as error:
            raise RuntimeError('Install the official metric dependencies in the evaluation environment; see README') from error
        finally:
            sys.path.remove(str(repo))

    def score(self, browser, predicted_html, reference_html, viewport, out):
        from .ir import write_json
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
                browser.load(html.read_text(),viewport)
                browser.page.screenshot(path=str(png),animations='disabled')
                browser.screenshots += 1

            def screenshot_command(command):
                # Interpret only upstream screenshot requests; never execute its shell.
                parts = shlex.split(command)
                if len(parts)!=6 or parts[0]!='python3' or Path(parts[1]).name!='screenshot_single.py' or parts[2]!='--html' or parts[4]!='--png':
                    raise ValueError('Unsupported upstream subprocess request')
                render(parts[3],parts[5]); return 0

            def blocks(image_path):
                html, ph, ph1, pp, pp1 = self.ocr.get_itermediate_names(image_path)
                self.ocr.process_html(html,ph)
                self.ocr.process_html(html,ph1,offset=50)
                render(ph,pp); render(ph1,pp1)
                pixels = self.ocr.find_different_pixels(pp,pp1)
                if pixels is None: return []
                text = self.ocr.flatten_tree(self.ocr.extract_text_with_color(ph))
                return self.ocr.get_blocks_from_image_diff_pixels(pp,text,pixels)

            self.module.os = types.SimpleNamespace(system=screenshot_command)
            self.module.get_blocks_ocr_free = blocks
            result = self.module.visual_eval_v3_multi([[str(pred)],str(ref)])
            values = result[0][2]
            if len(values)!=5 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError('Invalid official metric output')
            scores = dict(zip(('official_block','official_text','official_position','official_color','official_clip'),map(float,values)))
            write_json(out/'official.json',{'scores':scores,'source_sha':self.source_sha,
                'compatibility_patch':self.compatibility_patch,
                'protocol':'Upstream metric functions; shared fixed viewport; JS/network disabled; embedded assets; temporary preprocessing copies'})
            return scores

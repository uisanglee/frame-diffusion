from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from framediff.official_metrics import OfficialMetrics


def test_block_cache_reuses_target_and_isolated_from_upstream_mutation(tmp_path):
    scorer = OfficialMetrics.__new__(OfficialMetrics)
    scorer.source_sha = 'fixed'
    scorer.compatibility_patch = None
    scorer._block_cache = OrderedDict()
    calls = []

    def intermediate(image):
        root = Path(image).parent
        return tuple(str(root/name) for name in ('ref.html','a.html','b.html','a.png','b.png'))

    def process(source, target, **kw):
        calls.append(source)
        Path(target).write_text(Path(source).read_text())

    scorer.ocr = SimpleNamespace(get_itermediate_names=intermediate, process_html=process,
        find_different_pixels=lambda *a: [1], flatten_tree=lambda x: x,
        extract_text_with_color=lambda p: [],
        get_blocks_from_image_diff_pixels=lambda *a: [{'text':'original'}])
    scorer.module = SimpleNamespace()

    def evaluate(args):
        ref = Path(args[1]).with_suffix('.png')
        blocks = scorer.module.get_blocks_ocr_free(str(ref))
        assert blocks == [{'text':'original'}]
        blocks[0]['text'] = 'mutated by upstream'
        return [[None, None, [1.,1.,1.,1.,1.]]]

    scorer.module.visual_eval_v3_multi = evaluate
    browser = SimpleNamespace(load=lambda *a: None, screenshots=0,
        page=SimpleNamespace(screenshot=lambda **kw: None))
    first = scorer.score(browser, '<html>p1</html>', '<html>ref</html>', [320,240], tmp_path/'a')
    second = scorer.score(browser, '<html>p2</html>', '<html>ref</html>', [320,240], tmp_path/'b')
    assert first == second and len(calls) == 2
    scorer.score(browser, 'p', '<html>ref</html>', [640,480], tmp_path/'c')
    assert len(calls) == 4
    scorer.score(browser, 'p', '<html>changed</html>', [640,480], tmp_path/'d')
    assert len(calls) == 6


def test_upstream_timing_hooks_restore_after_failure(tmp_path, capsys):
    import pytest
    from framediff.evaluation_timing import EvaluationTiming
    scorer = OfficialMetrics.__new__(OfficialMetrics)
    scorer.source_sha = 'fixed'
    scorer.compatibility_patch = None
    scorer._block_cache = OrderedDict()
    scorer.ocr = SimpleNamespace()
    def clip(*args):
        scorer.module.rescale_and_mask()
        raise ValueError('clip failed')
    preprocess = lambda: None
    scorer.module = SimpleNamespace(calculate_clip_similarity_with_blocks=clip,
        rescale_and_mask=preprocess)
    scorer.module.visual_eval_v3_multi = lambda args: scorer.module.calculate_clip_similarity_with_blocks()
    timing = EvaluationTiming(id='test')
    with pytest.raises(ValueError, match='clip failed'):
        scorer.score(None, 'p', 'r', [320,240], tmp_path, timing=timing)
    assert scorer.module.calculate_clip_similarity_with_blocks is clip
    assert scorer.module.rescale_and_mask is preprocess
    assert timing.calls == {'clip_image_preprocess': 1, 'clip': 1, 'official_other': 1}
    assert '"status": "failed"' in capsys.readouterr().out


def test_render_cache_reuses_bytes_and_invalidates_viewport_and_browser(tmp_path):
    scorer = OfficialMetrics.__new__(OfficialMetrics)
    scorer.source_sha = 'fixed'
    scorer.compatibility_patch = None
    scorer._block_cache = OrderedDict()
    scorer.module = SimpleNamespace()
    scorer.ocr = SimpleNamespace()
    loads = []
    def browser():
        return SimpleNamespace(load=lambda *args: loads.append(args), screenshots=0,
            page=SimpleNamespace(screenshot=lambda **kw: Path(kw['path']).write_bytes(b'PNG')))
    def evaluate(args):
        ref = Path(args[1])
        scorer.module.os.system(f'python3 screenshot_single.py --html {ref} --png {ref.with_suffix(".png")}')
        assert ref.with_suffix('.png').read_bytes() == b'PNG'
        return [[None, None, [1., 1., 1., 1., 1.]]]
    scorer.module.visual_eval_v3_multi = evaluate
    first = browser()
    scorer.score(first, 'p1', 'ref', [320,240], tmp_path/'a')
    scorer.score(first, 'p2', 'ref', [320,240], tmp_path/'b')
    assert len(loads) == 1 and first.screenshots == 1
    scorer.score(first, 'p', 'ref', [640,480], tmp_path/'c')
    scorer.score(first, 'p', 'changed ref', [320,240], tmp_path/'d')
    scorer.score(browser(), 'p', 'ref', [320,240], tmp_path/'e')
    assert len(loads) == 4

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

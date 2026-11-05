import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from framediff import official_pixel_metrics as fast


@pytest.fixture
def upstream():
    pytest.importorskip('cv2')
    path = Path(__file__).parent/'fixtures/design2code_ocr_reference.py'
    spec = importlib.util.spec_from_file_location('ocr_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supported_guard_rejects_modified_implementation(upstream):
    assert fast.supported(upstream)
    changed = SimpleNamespace(**{name: getattr(upstream, name) for name in fast._SUPPORTED})
    changed.similar = lambda a, b: abs(a-b) < 9
    assert not fast.supported(changed)
    assert not fast.supported(SimpleNamespace())


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'L'])
def test_different_pixels_exact_order_wraparound_and_threshold(tmp_path, upstream, mode):
    rng = np.random.default_rng(17)
    a = rng.integers(0, 256, (31, 37, 3), dtype=np.uint8)
    b = ((a.astype(np.int16)+50+rng.integers(-9, 10, a.shape)) % 256).astype(np.uint8)
    p, q = str(tmp_path/'p.png'), str(tmp_path/'q.png')
    Image.fromarray(a).convert(mode).save(p)
    Image.fromarray(b).convert(mode).save(q)
    expected, actual = upstream.find_different_pixels(p, q), fast.find_different_pixels(p, q)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_different_pixels_empty_and_mismatched_size(tmp_path, upstream):
    p, q = str(tmp_path/'p.png'), str(tmp_path/'q.png')
    Image.new('RGB', (5, 4), 'black').save(p)
    Image.new('RGB', (5, 4), 'black').save(q)
    assert fast.find_different_pixels(p, q) is upstream.find_different_pixels(p, q) is None
    Image.new('RGB', (5, 5), 'black').save(q)
    assert fast.find_different_pixels(p, q) is upstream.find_different_pixels(p, q) is None


@pytest.mark.parametrize('seed', range(5))
def test_blocks_exact_with_duplicate_colors_missing_pixels_and_range_overflow(tmp_path, upstream, seed):
    rng = np.random.default_rng(seed)
    # Colors exercise valid thresholds and uint8 overflow near 0/255.
    palette = np.array([[0, 0, 0], [255, 255, 255], [3, 128, 250],
                        [60, 90, 130], [100, 140, 200], [15, 30, 45]], dtype=np.uint8)
    indices = rng.integers(0, len(palette), (31, 37))
    a = palette[indices]
    p = str(tmp_path/'pred_p.png')
    Image.fromarray(a).save(p)
    Image.fromarray(rng.integers(0, 256, a.shape, dtype=np.uint8)).save(tmp_path/'pred.png')
    coords = np.column_stack(np.where(rng.random(indices.shape) > .3))
    coords = np.concatenate([coords[::-1], coords[:15], [[-1, 0], [99, 0]]])
    text = [(f'Text {i}', '#'+''.join(f'{v:02x}' for v in rgb)) for i, rgb in enumerate(palette)]
    text += [('Repeated color with another text', text[-1][1]), ('invalid', '#garbage')]
    assert fast.get_blocks_from_image_diff_pixels(p, text, coords) == upstream.get_blocks_from_image_diff_pixels(p, text, coords)


def test_original_png_is_loaded_once_and_color_cache_preserves_all_texts(tmp_path, upstream, monkeypatch):
    p = str(tmp_path/'pred_p.png')
    Image.new('RGB', (20, 10), (100, 120, 140)).save(p)
    Image.new('RGB', (20, 10), (20, 30, 40)).save(tmp_path/'pred.png')
    coords = np.column_stack(np.where(np.ones((10, 20), dtype=bool)))
    texts = [(str(i), '#64788c') for i in range(15)]
    expected = upstream.get_blocks_from_image_diff_pixels(p, texts, coords)
    opened = []
    original_open = Image.open
    def open_once(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, 'open', open_once)
    actual = fast.get_blocks_from_image_diff_pixels(p, texts, coords)
    assert actual == expected and len(actual) == len(texts)
    assert opened == [str(tmp_path/'pred.png')]


def test_blocks_include_exact_plus_minus_four_bounds(tmp_path, upstream):
    import itertools
    colors = np.array([[100+r, 120+g, 140+b] for r, g, b in
                       itertools.product(range(-5, 6), repeat=3)], dtype=np.uint8).reshape(11, 121, 3)
    p = str(tmp_path/'pred_p.png')
    Image.fromarray(colors).save(p)
    Image.fromarray(colors[::-1]).save(tmp_path/'pred.png')
    coords = np.column_stack(np.where(np.ones(colors.shape[:2], dtype=bool)))
    texts = [('Thresholds', '#64788c')]
    assert fast.get_blocks_from_image_diff_pixels(p, texts, coords) == upstream.get_blocks_from_image_diff_pixels(p, texts, coords)


def test_blocks_no_match_does_not_open_original_image(tmp_path, upstream):
    p = str(tmp_path/'pred_p.png')
    Image.new('RGB', (3, 2), (100, 120, 140)).save(p)
    coords = np.empty((0, 2), dtype=np.int64)
    texts = [('empty', '#64788c')]
    assert fast.get_blocks_from_image_diff_pixels(p, texts, coords) == upstream.get_blocks_from_image_diff_pixels(p, texts, coords) == []

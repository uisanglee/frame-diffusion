"""Exact fast paths for the supported Design2Code OCR-free pixel functions.

The upstream implementation is fingerprinted before enabling these paths. In
particular, its uint8 range overflow and x-major output order are intentional
here: changing either would silently change the official evaluation.
"""
import ast
import hashlib
import inspect
import textwrap

import numpy as np
from PIL import Image


_SUPPORTED = {
    'similar': '9b77b9e0cc0cd49f19e857a6eac4f7eefc991a8d74855c47b47abaf074a8dd2a',
    'find_different_pixels': '85b535f26d834badd96ef046c2a7c9cacd248921c38fdeab12d6441f1909409f',
    'average_color': '88ffd10bd5e41b133dae88e3cf62e9b295893311a6b56d54dddc5deb1ffa2a8b',
    'get_blocks_from_image_diff_pixels': 'b72b12b9f85f72ca4e36a52277b6eea94ba048d9ec085c275b52680ad760c002',
}


def supported(ocr):
    """Return false for unknown upstream versions instead of changing semantics."""
    try:
        for name, expected in _SUPPORTED.items():
            tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(ocr, name))))
            actual = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
            if actual != expected:
                return False
    except (AttributeError, OSError, TypeError, SyntaxError):
        return False
    return True


def find_different_pixels(image1_path, image2_path):
    with Image.open(image1_path) as first, Image.open(image2_path) as second:
        if first.size != second.size:
            print(f'[Warning] Images are not the same size, {image1_path}, {image2_path}')
            return None
        # int16 prevents subtraction from wrapping; upstream compares Python ints.
        a = np.asarray(first.convert('RGB'), dtype=np.int16)
        b = np.asarray(second.convert('RGB'), dtype=np.int16)
    match = np.all(np.abs((a + 50) % 256 - b) <= 8, axis=2)
    x, y = np.nonzero(match.T)  # Upstream visits x first, then y.
    return np.column_stack((y, x)) if len(x) else None


def get_blocks_from_image_diff_pixels(image_path, html_text_color_tree, different_pixels):
    import cv2
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    # Upstream intersects sorted unique coordinate rows for every text block.
    # Compute the common candidate set once, then inspect only those pixels.
    candidates = np.unique(different_pixels, axis=0)
    candidates = candidates[(candidates[:, 0] >= 0) & (candidates[:, 0] < height) &
                            (candidates[:, 1] >= 0) & (candidates[:, 1] < width)]
    sampled = image[candidates[:, 0], candidates[:, 1]].reshape(-1, 1, 3)
    original = None
    color_cache = {}
    blocks = []
    for item in html_text_color_tree:
        try:
            value = item[1].lstrip('#')
            rgb = tuple(int(value[i:i+2], 16) for i in (0, 2, 4))
            color = np.array(rgb[::-1], dtype='uint8')
        except Exception:
            continue
        key = tuple(color)
        if key not in color_cache:
            # Keep the exact upstream uint8 overflow instead of clipping bounds.
            lower, upper = color - 4, color + 4
            if len(candidates):
                mask = cv2.inRange(sampled, lower, upper).ravel()
                coords = candidates[mask > 0]
            else:
                coords = candidates
            if not len(coords):
                color_cache[key] = None
            else:
                if original is None:
                    with Image.open(image_path.replace('_p.png', '.png')) as source:
                        original = np.array(source.convert('RGB'))
                y_min, x_min = np.min(coords, axis=0)
                y_max, x_max = np.max(coords, axis=0)
                colors = original[coords[:, 0], coords[:, 1]]
                average = tuple(np.mean(colors, axis=0).astype(int))
                box = (x_min / width, y_min / height,
                       (x_max - x_min + 1) / width, (y_max - y_min + 1) / height)
                color_cache[key] = (box, average)
        result = color_cache[key]
        if result is not None:
            box, average = result
            blocks.append({'text': item[0].lower(), 'bbox': box, 'color': average})
    return blocks

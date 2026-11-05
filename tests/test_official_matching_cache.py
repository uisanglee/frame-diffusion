from difflib import SequenceMatcher
import random

import pytest

from framediff.official_matching_cache import cached_text_similarity


def calculate_similarity(block1, block2, max_distance=1.42):
    text_similarity = SequenceMatcher(None, block1['text'], block2['text']).ratio()
    return text_similarity


def test_ordered_cache_preserves_asymmetric_difflib_results():
    cached = cached_text_similarity(calculate_similarity)
    assert cached is not calculate_similarity
    left, right = {'text': 'tide'}, {'text': 'diet'}
    assert cached(left, right) == calculate_similarity(left, right) == .25
    assert cached(right, left) == calculate_similarity(right, left) == .5
    assert cached(dict(left, bbox=[1, 2, 3, 4]), right) == .25
    assert cached.cache_info().hits == 1
    assert cached.cache_info().misses == 2


def test_cache_is_bounded_and_uses_lru_eviction():
    cached = cached_text_similarity(calculate_similarity, maxsize=2)
    for text in ('a', 'b', 'a', 'c'):
        cached({'text': text}, {'text': 'reference'})
    assert cached.cache_info().currsize == 2
    assert cached.cache_info().hits == 1
    cached({'text': 'b'}, {'text': 'reference'})  # b was least recently used.
    assert cached.cache_info().misses == 4
    cached.cache_clear()
    assert cached.cache_info().currsize == 0


def test_cache_exactly_preserves_empty_unicode_and_long_texts():
    cached = cached_text_similarity(calculate_similarity)
    rng = random.Random(42)
    texts = ['', '가나다 🏠', 'a' * 250 + 'b', 'a' * 250 + 'c']
    texts += [''.join(rng.choices('abcd 가나', k=50)) for _ in range(12)]
    for left in texts:
        for right in texts:
            a, b = {'text': left}, {'text': right}
            expected = calculate_similarity(a, b)
            assert cached(a, b) == expected
            assert cached(a, b) == expected


def test_unknown_implementation_is_not_optimized():
    def geometry_aware(a, b, max_distance=1.42):
        return a['bbox'][0] + b['bbox'][0]
    assert cached_text_similarity(geometry_aware) is geometry_aware


def test_unhashable_inputs_keep_original_behavior():
    cached = cached_text_similarity(calculate_similarity)
    left, right = {'text': ['a', 'b']}, {'text': ['b']}
    assert cached(left, right) == calculate_similarity(left, right)
    assert cached({'text': 'a'}, {'text': 'b'}, []) == 0.
    assert cached.cache_info().currsize == 0


@pytest.mark.parametrize('maxsize', [0, -1, True, 1.5])
def test_invalid_cache_bounds(maxsize):
    with pytest.raises(ValueError):
        cached_text_similarity(calculate_similarity, maxsize=maxsize)

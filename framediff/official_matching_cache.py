"""Bounded memoization for the upstream text-only matching cost.

The order of the two texts matters: difflib's matching ratio is not symmetric.
Unknown upstream implementations are left untouched rather than assuming that
their result is independent of boxes, colors, or additional block metadata.
"""
from difflib import SequenceMatcher
from functools import lru_cache, wraps
from types import FunctionType


def _upstream_text_similarity(block1, block2, max_distance=1.42):
    text_similarity = SequenceMatcher(None, block1['text'], block2['text']).ratio()
    return text_similarity


def cached_text_similarity(original, maxsize=65536):
    """Wrap the known upstream function, or return an unknown function unchanged.

    Compare compiled structure rather than source-file hashes, so comments and
    line numbers do not disable the optimization. No score, threshold, ordering,
    merging decision, or matching algorithm is changed.
    """
    if not isinstance(maxsize, int) or isinstance(maxsize, bool) or maxsize < 1:
        raise ValueError('Text similarity cache maxsize must be a positive integer')
    if not isinstance(original, FunctionType):
        return original
    expected = _upstream_text_similarity.__code__
    code = original.__code__
    fields = ('co_code', 'co_consts', 'co_names', 'co_varnames',
              'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount')
    if (any(getattr(code, name) != getattr(expected, name) for name in fields)
            or original.__defaults__ != (1.42,) or original.__kwdefaults__
            or original.__closure__
            or original.__globals__.get('SequenceMatcher') is not SequenceMatcher):
        return original

    @lru_cache(maxsize=maxsize)
    def cached(left, right, max_distance):
        # Call the original implementation on misses, including its own defaults
        # and SequenceMatcher autojunk behavior. Preserve text argument order.
        return original({'text': left}, {'text': right}, max_distance)

    @wraps(original)
    def wrapped(block1, block2, max_distance=1.42):
        left, right = block1['text'], block2['text']
        try:
            hash((left, right, max_distance))
        except TypeError:
            return original(block1, block2, max_distance)
        return cached(left, right, max_distance)

    wrapped.cache_info = cached.cache_info
    wrapped.cache_clear = cached.cache_clear
    return wrapped

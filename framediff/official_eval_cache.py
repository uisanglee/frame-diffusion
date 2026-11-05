"""Bounded, process-local caches for deterministic official evaluation work.

No persistent cache keys or score definitions are changed. Cached images and
features are copied on return because upstream code may modify them in place.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path


class ByteLRU:
    def __init__(self, max_bytes=32 * 1024 * 1024, max_entries=16):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.bytes = 0
        self.items = OrderedDict()

    def get(self, key):
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key][0]

    def put(self, key, value, size):
        old = self.items.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        if size > self.max_bytes:
            return
        self.items[key] = (value, size)
        self.bytes += size
        while self.bytes > self.max_bytes or len(self.items) > self.max_entries:
            _, (_, removed) = self.items.popitem(last=False)
            self.bytes -= removed


def cached_mask(function, cache, timing):
    """Key on image bytes AND exact ordered boxes; matching can change the mask."""
    def wrapped(image_path, blocks):
        key = (hashlib.sha256(Path(image_path).read_bytes()).digest(),
               tuple(tuple(box) for box in blocks))
        found = cache.get(key)
        if found is not None:
            timing.event('clip_image_preprocess', 'cache_hit')
            return found.copy()
        result = function(image_path, blocks)
        cache.put(key, result.copy(), len(result.tobytes()))
        return result
    return wrapped


class CachedClipModel:
    """Delegate unchanged model calls; only memoize deterministic encode_image."""
    def __init__(self, model, cache, timing):
        self.model, self.cache, self.timing = model, cache, timing

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode_image(self, image):
        import torch
        # Do not memoize training/dropout or autograd-enabled calls.
        if (self.model.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled(image.device.type)):
            return self.model.encode_image(image)
        with self.timing.stage('clip_feature_cache_lookup'):
            raw = image.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            key = (id(self.model), str(image.device), str(image.dtype), tuple(image.shape),
                   tuple(image.stride()), hashlib.sha256(raw).digest())
            found = self.cache.get(key)
        if found is not None:
            self.timing.event('clip_encode', 'cache_hit')
            return found.clone()
        with self.timing.stage('clip_encode'):
            result = self.model.encode_image(image)
            # Charge GPU execution to this stage, not to a subsequent cache lookup.
            if result.is_cuda:
                torch.cuda.synchronize(result.device)
        self.cache.put(key, result.detach().clone(), result.numel() * result.element_size())
        return result

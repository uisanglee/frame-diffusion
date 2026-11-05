from PIL import Image
import torch

from framediff.evaluation_timing import EvaluationTiming
from framediff.official_eval_cache import ByteLRU, CachedClipModel, cached_mask


def test_cache_is_bounded_by_bytes_and_count():
    cache = ByteLRU(max_bytes=5, max_entries=2)
    cache.put('a', b'aa', 2)
    cache.put('b', b'bb', 2)
    assert cache.get('a') == b'aa'
    cache.put('c', b'cc', 2)
    assert cache.get('b') is None
    assert cache.bytes == 4
    cache.put('a', b'longer', 6)
    assert cache.get('a') is None and cache.bytes == 2


def test_mask_cache_keys_content_and_boxes_and_copies_result(tmp_path):
    a, b = tmp_path/'a.png', tmp_path/'b.png'
    Image.new('RGB', (10,10), 'red').save(a)
    b.write_bytes(a.read_bytes())
    calls = []
    def original(path, boxes):
        calls.append((path, boxes))
        with Image.open(path) as im:
            return im.copy()
    wrapped = cached_mask(original, ByteLRU(), EvaluationTiming())
    result = wrapped(a, [(0.,0.,1.,1.)])
    result.putpixel((0,0), (0,0,0))
    assert wrapped(b, [(0.,0.,1.,1.)]).getpixel((0,0)) == (255,0,0)
    assert len(calls) == 1
    wrapped(b, [(0.,0.,.5,1.)])
    assert len(calls) == 2
    Image.new('RGB', (10,10), 'blue').save(b)
    wrapped(b, [(0.,0.,1.,1.)])
    assert len(calls) == 3


def test_clip_features_are_identical_and_not_mutated_by_normalization():
    class Encoder:
        training = False
        def __init__(self): self.calls = 0
        def encode_image(self, image):
            self.calls += 1
            return image.flatten(1)[:, :4] * 2
    model = Encoder()
    wrapper = CachedClipModel(model, ByteLRU(), EvaluationTiming())
    image = torch.arange(12, dtype=torch.float32).reshape(1,3,2,2)
    with torch.no_grad():
        a = wrapper.encode_image(image)
        expected = a.clone()
        a /= a.norm(dim=-1, keepdim=True)
        b = wrapper.encode_image(image.clone())
        assert torch.equal(b, expected)
        b /= b.norm(dim=-1, keepdim=True)
        assert torch.equal(a, b) and model.calls == 1
        wrapper.encode_image(image + 1)
        assert model.calls == 2
        model.training = True
        wrapper.encode_image(image)
        wrapper.encode_image(image)
        assert model.calls == 4
    model.training = False
    wrapper.encode_image(image)
    assert model.calls == 5  # gradients enabled: no cache
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        wrapper.encode_image(image)
        wrapper.encode_image(image)
    assert model.calls == 7  # autocast contexts never reuse a different precision

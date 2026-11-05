from types import SimpleNamespace
import pytest
from PIL import Image
from framediff.websight_backend import MODEL_ID, MAX_LENGTH, official_prefix, convert_to_rgb, websight_generate, validate_request


def args(**values):
    return SimpleNamespace(**dict(model=MODEL_ID, four_bit=False, task='generate-html', current=None, **values))


def test_official_prefix_has_no_chat_or_instruction():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(bos_token='<s>'))
    model = SimpleNamespace(config=SimpleNamespace(perceiver_config=SimpleNamespace(resampler_n_latents=64)))
    prefix = official_prefix(processor, model)
    assert prefix == '<s><fake_token_around_image>' + '<image>' * 64 + '<fake_token_around_image>'
    assert MAX_LENGTH == 4096


def test_transparent_images_composite_on_white():
    image = Image.new('RGBA', (1,1), (0,0,0,0))
    result = convert_to_rgb(image)
    assert result.mode == 'RGB' and result.getpixel((0,0)) == (255,255,255)


def test_revision_and_prompt_not_silently_ignored():
    cfg = args()
    image = Image.new('RGB', (10,20))
    with pytest.raises(ValueError, match='single-image'):
        validate_request(cfg, [image,image])
    cfg.task = 'revise-html'
    with pytest.raises(ValueError, match='self-revision'):
        validate_request(cfg, [image])
    cfg.task = 'generate-html'
    with pytest.raises(ValueError, match='image-only'):
        websight_generate(cfg, 'Return HTML', [image])


def test_passes_original_image_to_reused_runtime():
    cfg = args()
    image = Image.new('RGBA', (1366,768), (1,2,3,0))
    calls=[]
    runtime=SimpleNamespace(generate=lambda a, im: (calls.append(im) or '<html></html>', {'output_tokens':5}))
    assert websight_generate(cfg, '', [image], runtime)[1]['output_tokens'] == 5
    assert calls == [image]
    assert calls[0].size == (1366,768) and calls[0].mode == 'RGBA'


def test_official_processor_receives_bilinear_960_transform(monkeypatch):
    import sys
    import types
    import numpy as np
    import torch
    from framediff.websight_backend import official_inputs
    calls=[]
    utils = types.ModuleType('transformers.image_utils')
    utils.to_numpy_array = np.asarray
    utils.PILImageResampling = Image.Resampling
    utils.ChannelDimension = SimpleNamespace(FIRST='first')
    transforms = types.ModuleType('transformers.image_transforms')
    def resize(array, size, resample):
        calls.append((size, resample))
        return np.asarray(Image.fromarray(array).resize(size[::-1], resample=resample))
    transforms.resize=resize
    transforms.to_channel_dimension_format=lambda array, dimension: np.moveaxis(array, -1, 0)
    monkeypatch.setitem(sys.modules, 'transformers.image_utils', utils)
    monkeypatch.setitem(sys.modules, 'transformers.image_transforms', transforms)
    class Tokenizer:
        bos_token='<s>'
        def __call__(self, text, **kwargs):
            assert text.count('<image>') == 64
            assert kwargs == {'return_tensors':'pt', 'add_special_tokens':False}
            return {'input_ids':torch.ones((1,67), dtype=torch.long)}
    class Pixels:
        image_mean=[.5,.5,.5]
        image_std=[.5,.5,.5]
        def rescale(self, array, scale):
            assert scale == 1/255
            return array.astype(np.float32)*scale
        def normalize(self, array, mean, std):
            assert mean == self.image_mean and std == self.image_std
            return (array-np.array(mean, dtype=np.float32))/np.array(std, dtype=np.float32)
        def __call__(self, images, transform):
            return torch.stack([transform(im) for im in images])[None]
    processor=SimpleNamespace(tokenizer=Tokenizer(), image_processor=Pixels())
    model=SimpleNamespace(config=SimpleNamespace(perceiver_config=SimpleNamespace(resampler_n_latents=64)))
    inputs=official_inputs(processor, model, Image.new('RGBA', (20,10), (0,0,0,0)))
    assert calls == [((960,960), Image.Resampling.BILINEAR)]
    assert inputs['pixel_values'].shape == (1,1,3,960,960)
    assert torch.all(inputs['pixel_values'] == 1)  # Transparent input became white.

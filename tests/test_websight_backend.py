from types import SimpleNamespace
import pytest
from PIL import Image
from framediff.websight_backend import (MODEL_ID, DEFAULT_MAX_NEW_TOKENS, official_prefix,
    convert_to_rgb, websight_generate, validate_request, generation_options, load_processor)


def args(**values):
    return SimpleNamespace(**dict(model=MODEL_ID, four_bit=False, task='generate-html', current=None, **values))


def test_builtin_processor_does_not_follow_malformed_remote_auto_map(monkeypatch):
    import sys
    seen = {}
    processor = object()
    def load(model, **kwargs):
        # Official auto_map names a built-in class, not a remote module.Class.
        if kwargs.get('trust_remote_code'):
            module, cls = 'IdeficsProcessor'.split('.')
        seen.update(model=model, **kwargs)
        return processor
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=load)))
    assert load_processor(MODEL_ID, 'fixed-sha') is processor
    assert seen == dict(model=MODEL_ID, revision='fixed-sha', trust_remote_code=False)


def test_official_prefix_has_no_chat_or_instruction():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(bos_token='<s>'))
    model = SimpleNamespace(config=SimpleNamespace(perceiver_config=SimpleNamespace(resampler_n_latents=64)))
    prefix = official_prefix(processor, model)
    assert prefix == '<s><fake_token_around_image>' + '<image>' * 64 + '<fake_token_around_image>'
    assert DEFAULT_MAX_NEW_TOKENS == 16384


def test_output_budget_counts_output_only_and_checks_real_context():
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768))
    assert generation_options(model, 67, 16384) == {'max_new_tokens':16384}
    with pytest.raises(ValueError, match='not silently reduced'):
        generation_options(model, 17000, 16384)


def test_invalid_output_budget_rejected():
    with pytest.raises(ValueError, match='positive'):
        validate_request(args(max_new_tokens=0), [object()])


def test_runtime_request_forwards_output_budget(tmp_path):
    import json
    from framediff.websight_backend import WebSightRuntime
    runtime = object.__new__(WebSightRuntime)
    runtime.root, runtime.counter = tmp_path, 0
    captured = {}
    def wait(path):
        captured.update(json.loads((tmp_path/'request-1.json').read_text()))
        return {'text':'<html></html>', 'metadata':{}}
    runtime._wait = wait
    runtime.generate(args(seed=42, max_new_tokens=16384), Image.new('RGB', (2,2)))
    assert captured['max_new_tokens'] == 16384


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
            return torch.stack([transform(im) for im in images])
    processor=SimpleNamespace(tokenizer=Tokenizer(), image_processor=Pixels())
    model=SimpleNamespace(config=SimpleNamespace(perceiver_config=SimpleNamespace(resampler_n_latents=64)))
    inputs=official_inputs(processor, model, Image.new('RGBA', (20,10), (0,0,0,0)))
    assert calls == [((960,960), Image.Resampling.BILINEAR)]
    assert inputs['pixel_values'].shape == (1,3,960,960)
    assert torch.all(inputs['pixel_values'] == 1)  # Transparent input became white.


def test_startup_wait_has_separate_deadline(monkeypatch, tmp_path):
    import framediff.websight_backend as backend
    runtime = object.__new__(backend.WebSightRuntime)
    runtime.timeout = 1800
    runtime.process = SimpleNamespace(poll=lambda: None)
    closed = []
    runtime.close = lambda: closed.append(True)
    ticks = iter([0, 1801, 7201])
    monkeypatch.setattr(backend.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(backend.time, 'sleep', lambda _: None)
    with pytest.raises(TimeoutError, match='startup.*7200'):
        runtime._wait(tmp_path/'ready.json', timeout=7200, phase='startup')
    assert closed == [True]


def test_generation_retains_existing_timeout(monkeypatch, tmp_path):
    import framediff.websight_backend as backend
    runtime = object.__new__(backend.WebSightRuntime)
    runtime.timeout = 1800
    runtime.process = SimpleNamespace(poll=lambda: None)
    runtime.close = lambda: None
    ticks = iter([0, 1801])
    monkeypatch.setattr(backend.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError, match='generation.*1800'):
        runtime._wait(tmp_path/'response.json')


def test_two_gpu_map_keeps_fusion_and_decoder_blocks_intact():
    from framediff.websight_backend import model_device_map
    config = SimpleNamespace(num_hidden_layers=32, use_resampler=True)
    mapping = model_device_map(config, 2)
    assert set(mapping.values()) == {0,1}
    for name in ('embed_tokens','vision_model','modality_projection','perceiver_resampler'):
        assert mapping['model.'+name] == 0
    assert mapping['lm_head'] == 0
    assert mapping['model.norm'] == 1
    assert [mapping[f'model.layers.{i}'] for i in range(32)] == [0]*14+[1]*18
    assert model_device_map(config,1) == {'':0}
    with pytest.raises(ValueError): model_device_map(config,0)
    config.use_resampler = False
    assert 'model.perceiver_resampler' not in model_device_map(config,2)

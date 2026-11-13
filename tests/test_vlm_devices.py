"""CPU tests of GPU planning/dispatch boundaries; no downloaded weights."""
from contextlib import nullcontext
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from framediff import vlm_devices as devices


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv('VLM_DEVICE_MAP', raising=False)
    monkeypatch.delenv('VLM_GPU_MEMORY_GIB', raising=False)


def test_meta_plan_rejects_offload_before_weight_loading(monkeypatch):
    monkeypatch.setenv('VLM_DEVICE_MAP', 'balanced')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda i: (24*1024**3, 24*1024**3))
    seen = {}
    empty = SimpleNamespace(tie_weights=lambda: None, _get_no_split_modules=lambda mode: ['DecoderLayer'])
    def build(config, **kwargs):
        seen['build'] = kwargs
        return empty
    model_class = SimpleNamespace(from_config=build)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda name, **kw: SimpleNamespace())))
    mapping = {'embed': 0, 'layers.0': 0, 'layers.1': 1, 'head': 1}
    def balanced(model, **kwargs):
        seen['budget'] = kwargs
        return kwargs['max_memory']
    def infer(model, **kwargs):
        assert model is empty
        assert kwargs['no_split_module_classes'] == ['DecoderLayer']
        return mapping.copy()
    monkeypatch.setitem(sys.modules, 'accelerate', SimpleNamespace(
        init_empty_weights=lambda **kw: nullcontext(), infer_auto_device_map=infer))
    monkeypatch.setitem(sys.modules, 'accelerate.utils', SimpleNamespace(get_balanced_memory=balanced))
    assert devices.placement_kwargs(model_class, 'model', 'pin') == {'device_map': mapping}
    assert seen['budget']['max_memory'] == {0:20*1024**3, 1:20*1024**3, 'cpu':0}
    assert seen['build']['torch_dtype'] == torch.bfloat16
    mapping['head'] = 'cpu'
    with pytest.raises(RuntimeError, match='Cannot fit'):
        devices.placement_kwargs(model_class, 'model', 'pin')
    mapping['head'] = 'disk'
    with pytest.raises(RuntimeError, match='Cannot fit'):
        devices.placement_kwargs(model_class, 'model', 'pin')


def test_balanced_requires_two_gpus_and_bf16(monkeypatch):
    monkeypatch.setenv('VLM_DEVICE_MAP', 'balanced')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(RuntimeError, match='exactly two'):
        devices.placement_kwargs(None, 'model', 'pin')
    with pytest.raises(ValueError, match='disable four-bit'):
        devices.placement_kwargs(None, 'model', 'pin', True)
    monkeypatch.setenv('VLM_GPU_MEMORY_GIB', 'nan')
    with pytest.raises(ValueError, match='finite'):
        devices.deployment_settings()


def test_actual_devices_and_embedding_input_checked(monkeypatch):
    monkeypatch.setenv('VLM_DEVICE_MAP', 'balanced')
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'test GPU')
    tensors = [SimpleNamespace(device=torch.device('cuda:0')), SimpleNamespace(device=torch.device('cuda:1'))]
    model = SimpleNamespace(parameters=lambda: iter(tensors), buffers=lambda: iter([]),
        get_input_embeddings=lambda: SimpleNamespace(weight=tensors[1]), device=torch.device('cuda:0'),
        hf_device_map={'embed': 1, 'layers': 0})
    devices.verify_cuda_placement(model)
    assert devices.input_device(model) == torch.device('cuda:1')
    assert model._framediff_placement['devices'] == ['cuda:0', 'cuda:1']
    tensors.append(SimpleNamespace(device=torch.device('cpu')))
    with pytest.raises(RuntimeError, match='entirely'):
        devices.verify_cuda_placement(model)


def test_measurement_covers_both_gpus(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda i: calls.append(('sync', i)))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda i: calls.append(('reset', i)))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda i: 100*(i+1))
    model = SimpleNamespace(_framediff_gpu_indices=[0,1])
    devices.start_cuda_measurement(model)
    result = devices.finish_cuda_measurement(model)
    assert calls == [('sync',0), ('reset',0), ('sync',1), ('reset',1), ('sync',0), ('sync',1)]
    assert result == dict(peak_cuda_bytes=200, peak_cuda_bytes_by_device={'0':100,'1':200},
                          sum_device_peak_cuda_bytes=300)


def test_logits_optimization_only_on_supported_forward():
    def supported(input_ids, logits_to_keep=0): pass
    def older(input_ids, num_logits_to_keep=0): pass
    def unsupported(input_ids, **kwargs): pass
    assert devices.efficient_generation_options(SimpleNamespace(forward=supported))['logits_to_keep'] == 1
    assert devices.efficient_generation_options(SimpleNamespace(forward=older))['num_logits_to_keep'] == 1
    options = devices.efficient_generation_options(SimpleNamespace(forward=unsupported))
    assert 'logits_to_keep' not in options and 'num_logits_to_keep' not in options
    assert options['use_cache'] is True and options['output_scores'] is False


@pytest.mark.parametrize('backend', ['hf', 'qwen'])
def test_bf16_loaders_pass_explicit_split_without_quantization(monkeypatch, backend):
    from framediff import vlm
    seen = {}
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash='a'*40))
    model.eval = lambda: model
    def load(name, **kwargs):
        seen.update(kwargs)
        return model
    loader = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForImageTextToText=loader, Qwen3VLForConditionalGeneration=loader,
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
        BitsAndBytesConfig=lambda **kw: pytest.fail('must not quantize BF16')))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(vlm, 'placement_kwargs', lambda *a: {'device_map': {'embed': 0, 'layers': 1}})
    monkeypatch.setattr(vlm, 'verify_cuda_placement', lambda m: None)
    args = SimpleNamespace(model='test/model', revision='a'*40, seed=42, four_bit=False, max_pixels=1024)
    getattr(vlm, f'load_{backend}_runtime')(args)
    assert seen['device_map'] == {'embed':0, 'layers':1}
    assert seen['dtype'] == torch.bfloat16 and seen['attn_implementation'] == 'sdpa'
    assert 'quantization_config' not in seen


def test_two_gpu_pilot_wires_all_bf16_and_separate_output(tmp_path):
    scripts = tmp_path/'scripts'; scripts.mkdir()
    source = Path('scripts/run_paper_pilot10_bf16_2gpu.sh').read_text()
    (scripts/'run_paper_pilot10_bf16_2gpu.sh').write_text(source)
    (scripts/'run_paper_pilot10.sh').write_text('''#!/usr/bin/env bash
printf '%s\\n' "$CUDA_VISIBLE_DEVICES" "$VLM_DEVICE_MAP" "$UI2CODEN_PRECISION" "$PILOT_OUTPUT" "$VLM_SPECS"
''')
    env = {k:v for k,v in os.environ.items() if k not in (
        'VLM_SPECS', 'PILOT_OUTPUT', 'VLM_DEVICE_MAP', 'UI2CODEN_PRECISION')}
    env['CUDA_VISIBLE_DEVICES'] = '3,4'
    result = subprocess.run(['bash', str(scripts/'run_paper_pilot10_bf16_2gpu.sh')],
                            env=env, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert lines[1:4] == ['3,4', 'balanced', 'bf16']
    assert lines[4] == 'runs/paper-pilot10-seed42-n2-bf16-2gpu-v1'
    specs = lines[5].split(',')
    assert len(specs) == 4 and all(spec.endswith('|bf16') for spec in specs)

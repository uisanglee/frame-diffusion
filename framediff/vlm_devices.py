"""Explicit GPU-only VLM placement and inference accounting."""
import inspect
import math
import os


def deployment_settings():
    mode = os.environ.get('VLM_DEVICE_MAP', 'single')
    if mode not in ('single', 'balanced'):
        raise ValueError('VLM_DEVICE_MAP must be single or balanced')
    gib = float(os.environ.get('VLM_GPU_MEMORY_GIB', '20'))
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError('VLM_GPU_MEMORY_GIB must be positive and finite')
    return dict(device_map=mode, gpu_weight_budget_gib=gib,
                contract='gpu-only-balanced-bf16-v1')


def placement_kwargs(model_class, model_id, revision, four_bit=False):
    import torch
    settings = deployment_settings()
    if settings['device_map'] == 'single':
        return {'device_map': {'': torch.cuda.current_device()}}
    if four_bit:
        raise ValueError('Balanced preset requires BF16; disable four-bit loading')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Balanced preset requires exactly two visible GPUs (CUDA_VISIBLE_DEVICES=0,1)')
    # Plan on meta before loading weights: never silently spill onto CPU/disk.
    from accelerate import init_empty_weights, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    from transformers import AutoConfig
    budget = int(settings['gpu_weight_budget_gib'] * 1024**3)
    memory = {}
    for index in range(2):
        free, _ = torch.cuda.mem_get_info(index)
        memory[index] = min(budget, free - 2 * 1024**3)
        if memory[index] <= 0:
            raise RuntimeError(f'GPU {index} has insufficient free memory')
    memory['cpu'] = 0
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    build = getattr(model_class, 'from_config', None) or model_class._from_config
    with init_empty_weights(include_buffers=True):
        empty = build(config, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        empty.tie_weights()
    no_split = empty._get_no_split_modules('balanced')
    memory = get_balanced_memory(empty, max_memory=memory,
                                 no_split_module_classes=no_split, dtype=torch.bfloat16)
    mapping = infer_auto_device_map(empty, max_memory=memory,
                                   no_split_module_classes=no_split, dtype=torch.bfloat16)
    del empty
    devices = set(mapping.values())
    if devices != {0, 1}:
        raise RuntimeError(f'Cannot fit model across both GPUs within weight budgets; planned devices: {devices}')
    return {'device_map': mapping}


def verify_cuda_placement(model):
    import torch
    settings = deployment_settings()
    balanced = settings['device_map'] == 'balanced'
    indices = [0, 1] if balanced else [torch.cuda.current_device()]
    expected = {torch.device('cuda', i) for i in indices}
    devices = {t.device for t in model.parameters()} | {t.device for t in model.buffers()}
    if devices != expected:
        raise RuntimeError(f'Local VLM must reside entirely on {expected}; found {devices}')
    for value in getattr(model, 'hf_device_map', {}).values():
        device = torch.device('cuda', value) if isinstance(value, int) else torch.device(value) if value != 'disk' else None
        if device is None or device.type != 'cuda' or (device.index or 0) not in indices:
            raise RuntimeError(f'CPU/disk/unexpected GPU placement is not allowed: {model.hf_device_map}')
    # Accelerate dispatches downstream layers; feed tokens to their embedding device.
    input_device = model.get_input_embeddings().weight.device if balanced else next(iter(expected))
    model._framediff_input_device = input_device
    model._framediff_gpu_indices = indices
    model._framediff_placement = dict(
        device=str(input_device), gpu_name=torch.cuda.get_device_name(input_device),
        devices=[str(torch.device('cuda', i)) for i in indices],
        gpu_names={str(i): torch.cuda.get_device_name(i) for i in indices},
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device_map={k: str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
        placement_contract=settings['contract'] if balanced else 'single-cuda-no-weight-offload-v2',
        deployment=settings)


def input_device(model):
    return getattr(model, '_framediff_input_device', model.device)


def start_cuda_measurement(model):
    import torch
    indices = getattr(model, '_framediff_gpu_indices', [None])
    for index in indices:
        if index is None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.synchronize(index)
            torch.cuda.reset_peak_memory_stats(index)


def finish_cuda_measurement(model):
    import torch
    peaks = {}
    for index in getattr(model, '_framediff_gpu_indices', [None]):
        if index is None:
            torch.cuda.synchronize()
            peaks['0'] = torch.cuda.max_memory_allocated()
        else:
            torch.cuda.synchronize(index)
            peaks[str(index)] = torch.cuda.max_memory_allocated(index)
    # Sum of device peaks is an upper bound, not a simultaneous aggregate peak.
    return dict(peak_cuda_bytes=max(peaks.values()), peak_cuda_bytes_by_device=peaks,
                sum_device_peak_cuda_bytes=sum(peaks.values()))


def efficient_generation_options(model):
    options = dict(use_cache=True, return_dict_in_generate=False, output_scores=False, output_logits=False,
                   output_attentions=False, output_hidden_states=False)
    # Do not pass unsupported kwargs to older/different model implementations.
    forward = getattr(model, 'forward', None)
    parameters = inspect.signature(forward).parameters if forward is not None else {}
    for name in ('logits_to_keep', 'num_logits_to_keep'):
        if name in parameters:
            options[name] = 1
            break
    return options

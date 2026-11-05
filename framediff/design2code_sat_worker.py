"""torchrun entry point; run only with the isolated SAT environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import signal
import threading


class HTMLStopStrategy:
    """Keep SAT's sampling/EOS behavior and also stop at a completed document.

    SAT 0.4.10 calls forward(), then checks is_done. Every model-parallel rank
    observes the same tokens (broadcast by BaseStrategy) and takes the same exit.
    """
    def __init__(self, strategy, completion):
        self.strategy, self.completion = strategy, completion

    def __getattr__(self, name):
        return getattr(self.strategy, name)

    @property
    def is_done(self):
        return self.strategy.is_done or self.completion.reason is not None

    def forward(self, logits, tokens, mems, *args, **kwargs):
        tokens, mems = self.strategy.forward(logits, tokens, mems, *args, **kwargs)
        self.completion(tokens, None)
        return tokens, mems

    def finalize(self, tokens, mems):
        return self.strategy.finalize(tokens, mems)


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def merge_sat_lora_cpu(model):
    """Merge learned SAT LoRA before MP partitioning, retaining parallel classes.

    SAT 0.4.10's generic splitter cannot partition LoraLinear.matrix_A/B.
    Its own merge helper returns nn.Linear and moves weights to CUDA, so use the
    same FP32 delta formula on CPU and keep the original SAT linear class.
    """
    import torch
    merged = 0
    def visit(module):
        nonlocal merged
        for name, child in list(module.named_children()):
            if type(child).__name__ == 'LoraLinear':
                base = child.original
                if base.weight.device.type != 'cpu':
                    raise RuntimeError('LoRA must be merged on CPU before model partitioning')
                if len(child.matrix_A) != len(child.matrix_B) or not len(child.matrix_A):
                    raise ValueError('Invalid SAT LoRA factor lists')
                offset = 0
                with torch.no_grad():
                    for a, b in zip(child.matrix_A, child.matrix_B):
                        delta = (b.float() @ a.float()) * child.scaling
                        count = delta.shape[0]
                        view = base.weight[offset:offset+count]
                        if view.shape != delta.shape:
                            raise ValueError('SAT LoRA shape does not match original weight')
                        view.copy_((view.float() + delta).to(view.dtype))
                        offset += count
                if offset != base.weight.shape[0]:
                    raise ValueError('SAT LoRA factors do not cover the whole weight')
                setattr(module, name, base)
                merged += 1
            else:
                visit(child)
    visit(model)
    if 'lora' in getattr(model, 'mixins', {}):
        model.del_mixin('lora')
    return merged


def watch_parent(parent_pid):
    while True:
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            os.killpg(os.getpgrp(), signal.SIGTERM)
            return
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    options = parser.parse_args()
    threading.Thread(target=watch_parent, args=(options.parent_pid,), daemon=True).start()
    rank, local_rank, world = (int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    root = Path(options.requests)
    # Import the project constants without importing the main environment's torch.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from framediff.design2code_sat import generation_settings, runtime_settings
    from framediff.html_generation import HTMLCompletionStop
    sys.path.insert(0, options.repo)
    import torch
    import numpy as np
    from importlib.metadata import version as package_version
    if package_version('SwissArmyTransformer') != '0.4.10':
        raise RuntimeError('Design2Code worker requires SwissArmyTransformer==0.4.10')
    from sat.model.mixins import CachedAutoregressiveMixin
    from sat.mpu import get_model_parallel_world_size
    from utils.models import FineTuneTestCogAgentModel
    from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('SAT worker requires exactly two GPUs and two model-parallel ranks')
    torch.cuda.set_device(local_rank)
    config = json.loads((Path(options.checkpoint)/'model_config.json').read_text())
    if config.get('use_lora') and config.get('lora_rank') != 8:
        raise ValueError('Design2Code official checkpoint must use the trained rank-8 LoRA')
    # Upstream allows unexpected state-dict keys with a warning. Do not silently
    # accept an incompatible checkpoint or omit trained parameters in this adapter.
    import sat.model.base_model as sat_base
    original_load = sat_base.load_checkpoint
    def strict_load(model, *args, **kwargs):
        # SAT's HackParameterList suppresses missing-key reporting, so compare
        # keys before its custom loaders run, not only in a post-load hook.
        original_state_load = model.load_state_dict
        expected = set(model.state_dict())
        def verified_state_load(state, *load_args, **load_kwargs):
            missing, unexpected = expected - set(state), set(state) - expected
            if missing or unexpected:
                raise ValueError(f'Checkpoint/model key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
            return original_state_load(state, *load_args, **load_kwargs)
        model.load_state_dict = verified_state_load
        try:
            return original_load(model, *args, **kwargs)
        finally:
            del model.load_state_dict
    sat_base.load_checkpoint = strict_load
    class PartitionableDesign2Code(FineTuneTestCogAgentModel):
        @classmethod
        def from_pretrained_base(cls, *args, build_only=False, overwrite_args=None, **kwargs):
            overrides = dict(overwrite_args or {})
            if build_only and overrides.get('model_parallel_size', 1) > 1:
                # Destination consists of original SAT parallel linear classes.
                overrides.update(use_lora=False, use_qlora=False)
            loaded, settings = FineTuneTestCogAgentModel.from_pretrained_base(
                *args, build_only=build_only, overwrite_args=overrides, **kwargs)
            if not build_only:
                if getattr(settings, 'use_qlora', False):
                    raise ValueError('Official Design2Code checkpoint must not use QLoRA')
                if getattr(settings, 'use_lora', False):
                    merged = merge_sat_lora_cpu(loaded)
                    if not merged:
                        raise ValueError('Checkpoint declares LoRA but no learned LoRA layers were loaded')
                    print(f'Merged {merged} LoRA layers on CPU before SAT partitioning', flush=True)
            return loaded, settings
    model, model_args = PartitionableDesign2Code.from_pretrained(options.checkpoint,
        args=argparse.Namespace(deepspeed=None, local_rank=local_rank, rank=rank, world_size=world,
            model_parallel_size=world, mode='inference', skip_init=True,
            use_gpu_initialization=True, device='cuda', bf16=True, fp16=None, seed=42),
        overwrite_args={'model_parallel_size': world})
    if get_model_parallel_world_size() != world:
        raise RuntimeError('SAT did not initialize two-way model parallelism')
    model = model.eval()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    version = getattr(model_args, 'text_processor_version', 'chat')
    tokenizer = llama2_tokenizer('lmsys/vicuna-7b-v1.5', signal_type=version)
    tokenizer_sha = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
    runtime_versions = {name: package_version(name) for name in ('torch', 'transformers', 'SwissArmyTransformer')}
    image_processor = get_image_processor(model_args.eva_args['image_size'][0])
    cross = get_image_processor(model_args.cross_image_pix) if hasattr(model_args, 'cross_image_pix') else None
    processor = llama2_text_processor_inference(tokenizer, 2048, model.image_length)
    chat_args = argparse.Namespace(bf16=True, fp16=False, stream_chat=False, version='chat')
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(options.repo).rglob('*.py')))).hexdigest()
    # Count the actual generated sequence, including EOS, before upstream decoding.
    token_counts = {}
    completion = None
    original_sampling = chat.__globals__['filling_sequence']
    def measured_sampling(model, seq, *args, **kwargs):
        token_counts['input_tokens'] = int((seq >= 0).sum().item())
        if completion is None:
            raise RuntimeError('HTML completion guard was not initialized')
        kwargs['strategy'] = HTMLStopStrategy(kwargs['strategy'], completion)
        output = original_sampling(model, seq, *args, **kwargs)
        tokens = output[0]
        ids = tokens[0].tolist() if hasattr(tokens[0], 'tolist') else tokens[0]
        token_counts['output_tokens'] = sum(int(token) >= 0 for token in ids)-token_counts['input_tokens']
        return output
    chat.__globals__['filling_sequence'] = measured_sampling
    torch.distributed.barrier()
    if rank == 0:
        atomic_json(root/'ready.json', {'ready': True})
    while True:
        request = [None]
        if rank == 0:
            request_file = root/'request.json'
            while not request_file.exists():
                time.sleep(.1)
            request[0] = json.loads(request_file.read_text())
            request_file.unlink()
        torch.distributed.broadcast_object_list(request, src=0)
        request = request[0]
        seed = request['seed']
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(local_rank)
        token_counts.clear()
        # Exact official image-only prefix, including visual placeholder tokens.
        # The processor's legacy max_target_length is unused for inference; no
        # output truncation or context-size mutation is applied to the checkpoint.
        prefix = processor(processor.history_to_prompt('', []))['input_ids']
        prompt_tokens = int(prefix.shape[1])
        settings = generation_settings(prompt_tokens, request['max_new_tokens'])
        completion = HTMLCompletionStop(tokenizer, prompt_tokens)
        configured_context = config.get('max_sequence_length', config.get('max_position_embeddings'))
        torch.cuda.synchronize(local_rank)
        started = time.perf_counter()
        # All ranks execute identical inputs and sample settings. SAT broadcasts tokens.
        with torch.inference_mode():
            answer, _, cache = chat(request['image'], model, processor, image_processor, '',
                history=None, cross_img_processor=cross, image=None,
                invalid_slices=processor.invalid_slices, args=chat_args, **settings)
        del cache
        torch.cuda.synchronize(local_rank)
        elapsed = time.perf_counter()-started
        memory = torch.tensor(torch.cuda.max_memory_allocated(local_rank), device=local_rank, dtype=torch.int64)
        peaks = [torch.zeros_like(memory) for _ in range(world)]
        torch.distributed.all_gather(peaks, memory)
        if rank == 0:
            output_tokens = token_counts['output_tokens']
            meta = {**runtime_settings(request['max_new_tokens']), **token_counts,
                    'generation_settings': {**settings, 'max_new_tokens': request['max_new_tokens']},
                    'generation_seconds': elapsed,
                    'configured_context_length': configured_context,
                    'exceeds_configured_context': (settings['max_length'] > int(configured_context)
                                                   if configured_context else None),
                    'context_policy': 'dynamic-sat-rotary-no-checkpoint-rescaling',
                    'generation_stop_reason': completion.reason or ('max_new_tokens' if output_tokens >= request['max_new_tokens'] else 'eos'),
                    'hit_output_limit': output_tokens >= request['max_new_tokens'],
                    'resolved_revision': options.revision, 'cogvlm_source_sha': source_hash,
                    'runtime_versions': runtime_versions, 'tokenizer_vocab_sha256': tokenizer_sha,
                    'tokenizer_model': 'lmsys/vicuna-7b-v1.5',
                    'peak_cuda_bytes': sum(int(p.item()) for p in peaks),
                    'peak_cuda_bytes_by_rank': [int(p.item()) for p in peaks],
                    'placement_contract': 'sat-two-gpu-model-parallel-v1',
                    'lora_partition_contract': 'merge-fp32-delta-on-cpu-before-sat-split-v1',
                    'gpu_names': [torch.cuda.get_device_name(i) for i in range(world)],
                    'prompt': '', 'image_preprocessing': 'official EVA + cross image processors from checkpoint config'}
            atomic_json(root/f"{request['id']}.json", {'answer': answer, 'meta': meta})


if __name__ == '__main__':
    main()

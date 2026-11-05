"""Load the real WebSight processor and preprocess an image; no model weights/GPU."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import torch
    import transformers
    from framediff.websight_backend import MODEL_ID, load_processor, official_inputs
    from framediff.websight_attention import load_model_config, allow_optional_flash_imports
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Resolve a commit from the cached config path so all files use one revision.
    config_path = Path(hf_hub_download(MODEL_ID, 'config.json', revision=args.revision))
    revision = config_path.parent.name
    config = json.loads(config_path.read_text())
    processor = load_processor(MODEL_ID, revision)
    model_config, attention_info = load_model_config(MODEL_ID, revision)
    with allow_optional_flash_imports(attention_info):
        model_class = get_class_from_dynamic_module(
            model_config.auto_map['AutoModelForCausalLM'], MODEL_ID, revision=revision)
    model = SimpleNamespace(config=SimpleNamespace(perceiver_config=SimpleNamespace(
        resampler_n_latents=config['perceiver_config']['resampler_n_latents'])))
    for token in ('<image>', '<fake_token_around_image>'):
        if processor.tokenizer.convert_tokens_to_ids(token) == processor.tokenizer.unk_token_id:
            raise RuntimeError(f'Missing WebSight special token: {token}')
    inputs = official_inputs(processor, model, Image.new('RGBA', (12,8), (0,0,0,0)))
    if tuple(inputs['pixel_values'].shape) != (1,3,960,960):
        raise RuntimeError(f'Unexpected image shape: {inputs["pixel_values"].shape}')
    if not torch.isfinite(inputs['pixel_values']).all():
        raise RuntimeError('Non-finite preprocessed pixels')
    print(json.dumps(dict(status='WebSight processor/tokenizer/image transform OK',
        revision=revision, processor=type(processor).__name__,
        tokenizer=type(processor.tokenizer).__name__, transformers=transformers.__version__,
        input_tokens=inputs['input_ids'].shape[1], pixel_shape=list(inputs['pixel_values'].shape),
        model_class=model_class.__name__, **attention_info,
        model_weights_loaded=False), ensure_ascii=False))


if __name__ == '__main__':
    main()

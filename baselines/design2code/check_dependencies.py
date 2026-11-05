"""Audit vendored inference imports and report all dependency errors together.

Does not download model weights or instantiate the 18B model.
"""
import argparse
import ast
import importlib
import importlib.util
from pathlib import Path
import sys

MODULES = {
    'sat': 'SwissArmyTransformer', 'transformers': 'transformers',
    'xformers': 'xformers', 'torch': 'torch', 'torchvision': 'torchvision',
    'deepspeed': 'deepspeed', 'timm': 'timm', 'accelerate': 'accelerate',
    'numpy': 'numpy', 'PIL': 'Pillow', 'spacy': 'spacy', 'seaborn': 'seaborn',
    'matplotlib': 'matplotlib', 'requests': 'requests', 'einops': 'einops',
    'jsonlines': 'jsonlines', 'datasets': 'datasets', 'sentencepiece': 'sentencepiece',
    'google.protobuf': 'protobuf', 'huggingface_hub': 'huggingface-hub', 'tqdm': 'tqdm',
}


def source_imports(repo):
    files = list((repo/'utils/models').rglob('*.py')) + list((repo/'utils/utils').rglob('*.py'))
    files += [repo/'finetune_demo/inference_design2code.py']
    result = {}
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else [])
            for name in names:
                top = name.split('.')[0]
                if top not in sys.stdlib_module_names and top != 'utils':
                    result.setdefault(top, set()).add(str(path.relative_to(repo)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    imports = source_imports(repo)
    errors = []
    for name, paths in sorted(imports.items()):
        print(f'{name}: {", ".join(sorted(paths))}', flush=True)
        if name not in MODULES and name != 'apex':
            errors.append(f'Unmapped upstream import {name}; update the audited requirements')
    if not args.source_only:
        for name, package in MODULES.items():
            try:
                importlib.import_module(name)
            except Exception as error:
                errors.append(f'{package} ({name}): {type(error).__name__}: {error}')
        try:
            import spacy
            spacy.load('en_core_web_sm')
        except Exception as error:
            errors.append(f'en_core_web_sm: {error}')
        sys.path.insert(0, str(repo))
        # Exercise the exact worker imports, including eager dataset/grounding imports.
        for name, members in {
            'utils.models': ['FineTuneTestCogAgentModel'],
            'utils.utils': ['chat', 'llama2_tokenizer', 'llama2_text_processor_inference', 'get_image_processor'],
            'sat.model.mixins': ['CachedAutoregressiveMixin'],
            'sat.mpu': ['get_model_parallel_world_size'],
        }.items():
            try:
                module = importlib.import_module(name)
                for member in members:
                    getattr(module, member)
            except Exception as error:
                errors.append(f'{name}: {type(error).__name__}: {error}')
    print('Optional: NVIDIA Apex (official LayerNorm fallback); bitsandbytes (unused BF16 quantization path).')
    if errors:
        raise SystemExit('Dependency audit failed:\n- ' + '\n- '.join(errors))
    print('Source dependency audit OK' if args.source_only else
          'Official Design2Code model imports OK. Model weights are downloaded at first inference.')


if __name__ == '__main__':
    main()

"""Combine completed disjoint model queues without copying Initials or rerunning inference."""
import argparse
from pathlib import Path

from .paper_matrix import export_matrix

GROUPS = {'gpu01': ('llava-next-7b', 'glm4v-9b-thinking'),
          'gpu24': ('websight-8b', 'ui2coden')}
DATASETS = ('design2code', 'design2code-hard')


def merge(root):
    root = Path(root).resolve()
    links = []
    # Preflight both cohorts and all sources before publishing any links.
    for dataset in DATASETS:
        inputs = [root / group / 'inputs' / f'{dataset}.jsonl' for group in GROUPS]
        if inputs[0].read_bytes() != inputs[1].read_bytes():
            raise ValueError(f'{dataset}: model queues used different inputs')
        for group, models in GROUPS.items():
            for model in models:
                source = root / group / dataset
                for suffix in (Path(model), Path('paper/main') / model):
                    src, dst = source / suffix, root / 'combined' / dataset / suffix
                    if not src.is_dir():
                        raise ValueError(f'Missing completed output: {src}')
                    if (dst.exists() or dst.is_symlink()) and (
                            not dst.is_symlink() or dst.resolve() != src.resolve()):
                        raise ValueError(f'Conflicting combined output: {dst}')
                    links.append((src, dst))
    # Validate every group's completed metrics before publishing the full table.
    for group, models in GROUPS.items():
        export_matrix(root / group, root / group / 'table',
                      datasets=DATASETS, include_tuide=False, revision_rounds=2, models=models)
    for src, dst in links:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_symlink():
            dst.symlink_to(src, target_is_directory=True)
    return export_matrix(root / 'combined', root / 'table',
                         datasets=DATASETS, include_tuide=False, revision_rounds=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    print(merge(parser.parse_args().root))

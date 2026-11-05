"""Share identical own/common evaluation outputs without repeating inference."""
import argparse
import os
import time
from pathlib import Path


def alias_result(source,alias):
    source=Path(source).absolute();alias=Path(alias).absolute()
    if not source.exists():raise ValueError(f'Missing completed evaluation: {source}')
    if source==alias:raise ValueError('Evaluation source and alias must differ')
    if alias.is_symlink() and alias.resolve()==source.resolve():return
    alias.parent.mkdir(parents=True,exist_ok=True)
    # Preserve old independent results, including their provenance, for review.
    if alias.exists() or alias.is_symlink():
        history=alias.parent/'.evaluation-history';history.mkdir(exist_ok=True)
        alias.rename(history/(alias.name+f'-{time.time_ns()}'))
    alias.symlink_to(os.path.relpath(source,alias.parent),target_is_directory=source.is_dir())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--alias',required=True)
    args=parser.parse_args();alias_result(args.source,args.alias)

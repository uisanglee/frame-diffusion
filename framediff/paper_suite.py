"""Build exact main/ablation paper tables from completed VLM experiment runs."""
import argparse
from pathlib import Path

from .ir import read_jsonl
from .paper_report import report


MAIN_METHODS = {
    'initial': 'Initial VLM',
    'self-revision-1': 'D2C Self-Revision (RGB)',
    'abstract-policy': 'TUIDE',
}
ABLATION_METHODS = {
    'abstract-policy': 'TUIDE',
    'screenshot-policy': 'TUIDE w/o abstraction',
    'abstract-vlm-self-revision': 'TUIDE w/o policy (VLM abstract revision)',
}


def selected(rows, mapping):
    result = []
    for row in rows:
        if row.get('method') not in mapping:
            continue
        item = dict(row)
        item['method'] = mapping[item['method']]
        result.append(item)
    return result


def build(root, aliases, representative, out):
    root = Path(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# Paper experiment suite', '', '## Main comparison', '']
    for alias in aliases:
        path = root / alias / 'evaluation' / 'metrics.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Missing completed evaluation for {alias}: {path}')
        rows = selected(read_jsonl(path), MAIN_METHODS)
        missing = set(MAIN_METHODS.values()) - {r['method'] for r in rows}
        if missing:
            raise ValueError(f'{alias} is missing main methods: {sorted(missing)}')
        report(rows, out / 'main' / alias)
        lines.append(f'- [{alias}](main/{alias}/paper-report.md)')
    path = root / representative / 'evaluation' / 'metrics.jsonl'
    rows = selected(read_jsonl(path), ABLATION_METHODS)
    missing = set(ABLATION_METHODS.values()) - {r['method'] for r in rows}
    if missing:
        raise ValueError(f'{representative} is missing ablation methods: {sorted(missing)}')
    report(rows, out / 'ablation')
    lines += ['', '## Ablation', '',
              f'- Representative VLM: `{representative}`',
              '- [Ablation table](ablation/paper-report.md)', '']
    (out / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Separate completed runs into main and ablation paper tables')
    parser.add_argument('--root', required=True)
    parser.add_argument('--aliases', required=True, help='Comma-separated VLM aliases')
    parser.add_argument('--representative', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    aliases = [x.strip() for x in args.aliases.split(',') if x.strip()]
    if not aliases or args.representative not in aliases:
        raise ValueError('representative must be one of aliases')
    build(args.root, aliases, args.representative, args.out)


if __name__ == '__main__':
    main()

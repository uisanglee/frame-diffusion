"""Reevaluate saved pilot outputs without calling any generator or repair model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from .official_metrics import EVALUATION_CONTRACT
from .paper_matrix import MODEL_LABELS


def refresh_cache(out):
    """Archive only old evaluation artifacts; resume this contract on reruns."""
    config_path = out / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if out.exists() and config.get('official_evaluation_contract') != EVALUATION_CONTRACT:
        backup = out.with_name(out.name + '.before-preserve-html-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        out.rename(backup)
        print(f'Archived old evaluation: {backup}', flush=True)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--official-repo', required=True)
    p.add_argument('--models', nargs='+', choices=list(MODEL_LABELS), help='By default infer saved model groups, including legacy runs')
    p.add_argument('--seed', default='42')
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--datasets', nargs='+', default=['design2code', 'design2code-hard'])
    args = p.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        p.error(f'Run does not exist: {root}')
    if not (Path(args.official_repo)/'Design2Code/metrics/visual_score.py').is_file():
        p.error('Official repository is missing visual_score.py')
    def run(*options):
        subprocess.run([sys.executable, '-m', *map(str, options)], check=True)
    if args.models:
        aliases = args.models
    else:
        # Union, not intersection: a group missing from one dataset must be
        # reported rather than silently dropped from the combined comparison.
        aliases = [alias for alias in MODEL_LABELS if any(
            (root/dataset/alias/'seeds'/args.seed).is_dir() for dataset in args.datasets)]
    if not aliases or len(aliases) != len(set(aliases)):
        p.error('No saved model groups found, or duplicate --models')
    missing = []
    jobs = 0
    for dataset in args.datasets:
        for alias in aliases:
            base = root/dataset/alias/'seeds'/args.seed
            for manifest, out in [(base/'prepare/prepared.jsonl', base/'evaluation'),
                                  (base/'ui2coden/results.jsonl', base/'ui2coden/evaluation')]:
                if not manifest.is_file():
                    missing.append(str(manifest))
                    continue
                config = refresh_cache(out)
                options = []
                if 'max_nodes' in config:
                    options = ['--max-nodes', str(config['max_nodes'])]
                elif manifest.parent.name == 'ui2coden':
                    options = ['--max-nodes', '2048']
                run('framediff', 'web-evaluate', '--data', manifest, '--out', out,
                    '--official-repo', args.official_repo, '--resume', *options)
                jobs += 1
    report = {'evaluation_contract': EVALUATION_CONTRACT, 'evaluations': jobs,
              'models': aliases, 'missing_manifests': missing, 'tables_regenerated': False}
    (root/'reevaluation-report.json').write_text(json.dumps(report, indent=2)+'\n')
    if missing:
        raise SystemExit('Evaluation completed for existing manifests. Full tables require missing outputs; no generation was run. See reevaluation-report.json. Existing tables are stale until regeneration succeeds.')
    for dataset in args.datasets:
        run('framediff.paper_suite', '--root', root/dataset,
            '--aliases', ','.join(aliases), '--representative', aliases[0],
            '--out', root/dataset/'paper', '--require-complete-metrics',
            '--own-initial-alias', 'ui2coden', '--revision-rounds', args.rounds,
            '--initial-seeds', args.seed, '--skip-ablation', '--include-ui2coden',
            '--ui2coden-rounds', args.rounds, '--skip-tuide')
    run('framediff.paper_matrix', 'export', '--root', root, '--out', root/'table',
        '--datasets', *args.datasets, '--models', *aliases, '--skip-tuide', '--revision-rounds', args.rounds)
    report['tables_regenerated'] = True
    (root/'reevaluation-report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

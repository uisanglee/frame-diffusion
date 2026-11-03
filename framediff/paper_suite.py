"""Build exact main/ablation paper tables from completed VLM experiment runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from .ir import read_jsonl
from .paper_report import report


MAIN_METHODS = {
    'initial': 'Initial VLM',
    'self-revision-1': 'D2C Self-Revision (RGB)',
    'design2code-self-revision': 'D2C Self-Revision (RGB)',
    'design2code-self-revision-3': 'D2C Self-Revision (RGB)',
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


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_by_id(root):
    path = root / 'prepare' / 'prepared.jsonl'
    records = {}
    for row in read_jsonl(path):
        key = str(row['id'])
        if key in records: raise ValueError(f'Duplicate prepared page: {key}')
        records[key] = row
    if not records:
        raise ValueError(f'Empty prepared manifest: {path}')
    return records


def _check_same_inputs(prepared, results_path, method):
    results = list(read_jsonl(results_path))
    ids = set()
    for row in results:
        page_id = str(row.get('page_id', re.sub(r'/repeat-\d+$', '', row['id'])))
        if page_id not in prepared:
            raise ValueError(f'{method}: unexpected page {page_id}')
        source = prepared[page_id]
        for key, left, right in (
            ('initial HTML', source['methods']['initial']['html'], row['methods']['initial']['html']),
            ('target image', source['screenshot'], row['screenshot']),
        ):
            if _sha(left) != _sha(right):
                raise ValueError(f'{method}: {key} differs on {page_id}')
        if source.get('benchmark_protocol', 'provided-screenshot-v1') != row.get('benchmark_protocol', 'provided-screenshot-v1'):
            raise ValueError(f'{method}: target-image protocol differs on {page_id}')
        ids.add(page_id)
    if ids != set(prepared):
        raise ValueError(f'{method}: page set differs from prepared manifest ({len(ids)} vs {len(prepared)})')
    return results


def _main_table(summary, out):
    keys = [('official_position', 'Position ↑'), ('official_clip', 'CLIP ↑'),
            ('dom_box_iou', 'Box IoU ↑'), ('improved_page_rate', 'Improved pages (%) ↑')]
    lines = ['| Repair method | Pages | ' + ' | '.join(label for _, label in keys) + ' |',
             '|---|---:|' + '---:|' * len(keys)]
    for item in summary['methods']:
        method = item['method']
        cells = []
        for metric, _ in keys:
            if metric == 'improved_page_rate':
                gain = next((row for row in summary['paired'] if row['method'] == method
                             and row['baseline'] == 'Initial VLM' and row['metric'] == 'dom_box_iou'), None)
                cells.append(f"{gain['improved_page_rate'] * 100:.1f}" if gain else '--')
            else:
                value = item['metrics'][metric]
                cells.append(f"{value['mean']:.3f}" if value['mean'] is not None else '--')
        lines.append(f"| {method} | {item['n_pages']} | " + ' | '.join(cells) + ' |')
    note = ('Values are page means from one frozen VLM draft set. '
            'Improved pages compares Box IoU with the same page’s initial rendering. '
            'See paper-report.md for page-bootstrap CIs and metric-specific counts.')
    (out / 'main-table.md').write_text('\n'.join(['# Main comparison', '', *lines, '', note, '']))


def build(root, aliases, representative, out, include_ablation=True, include_ui2coden=False,
          ui2coden_rounds=3, require_three_rounds=False, initial_seed=None,
          own_initial_alias='ui2coden', require_complete_metrics=False, revision_rounds=None):
    root = Path(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# Paper experiment suite', '', '## Main comparison', '']
    target_protocols = set()
    for alias in aliases:
        own_initial = alias == own_initial_alias
        if own_initial and not include_ui2coden:
            raise ValueError('UI2CodeN own-initial group requires UI2CodeN polishing')
        alias_root = root / alias / 'seeds' / str(initial_seed) if initial_seed is not None else root / alias
        path = alias_root / 'evaluation' / 'metrics.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Missing completed evaluation for {alias}: {path}')
        prepared = _prepared_by_id(alias_root) if include_ui2coden or initial_seed is not None or require_complete_metrics else None
        if prepared is not None:
            _check_same_inputs(prepared, alias_root / 'repair' / 'results.jsonl', f'{alias} main')
        required_round = revision_rounds or (3 if require_three_rounds else None)
        revision_method = ('design2code-self-revision' if required_round == 1 else f'design2code-self-revision-{required_round}')
        if required_round and not own_initial and any(revision_method not in row['methods']
                                        for row in read_jsonl(alias_root / 'prepare' / 'prepared.jsonl')):
            raise ValueError(f'{alias}: missing Design2Code self-revision round {required_round}')
        mapping = ({'initial':'Initial VLM','abstract-policy':'TUIDE'} if own_initial else ({'initial':'Initial VLM', revision_method:'D2C Self-Revision (RGB)',
                   'abstract-policy':'TUIDE'} if required_round else MAIN_METHODS))
        rows = selected(read_jsonl(path), mapping)
        if include_ui2coden:
            baseline = alias_root / 'ui2coden'
            _check_same_inputs(prepared, baseline / 'results.jsonl', f'{alias} UI2CodeN')
            ui_method = f'ui2coden-round-{ui2coden_rounds}'
            ui_label = 'UI2CodeN self-revision' if own_initial else 'UI2CodeN polishing'
            ui_rows = [dict(row, method=ui_label) for row in read_jsonl(baseline / 'evaluation' / 'metrics.jsonl')
                       if row['method'] == ui_method]
            if len(ui_rows) != len(prepared):
                raise ValueError(f'{alias}: UI2CodeN round {ui2coden_rounds} has {len(ui_rows)} results for {len(prepared)} pages')
            rows.extend(ui_rows)
        target_protocols.update(r.get('benchmark_protocol','provided-screenshot-v1') for r in rows)
        required = set(mapping.values()) | ({ui_label} if include_ui2coden else set())
        missing = required - {r['method'] for r in rows}
        if missing:
            raise ValueError(f'{alias} is missing main methods: {sorted(missing)}')
        main_summary = report(rows, out / 'main' / alias)
        if require_complete_metrics:
            expected = set(prepared)
            for method in required:
                found = [r for r in rows if r['method'] == method]
                ids = {str(r.get('page_id',re.sub(r'/repeat-\d+$','',r['id']))) for r in found}
                if ids != expected:
                    raise ValueError(f'{alias}/{method}: incomplete evaluation page set')
                for metric in ('official_position','official_clip','dom_box_iou'):
                    from .paper_report import finite
                    missing_ids = [r['id'] for r in found if not finite(r.get(metric))]
                    if missing_ids:
                        raise ValueError(f'{alias}/{method}: missing {metric} for {len(missing_ids)} trials; '
                                         f'first={missing_ids[0]}. See evaluation errors; final table not exported.')
        lines.append(f'- [{alias}](main/{alias}/paper-report.md)')
        _main_table(main_summary, out / 'main' / alias)
        lines.append(f'  - [Paper table](main/{alias}/main-table.md)')
    if len(target_protocols)>1:
        raise ValueError(f'Mixed target-image protocols across VLM runs: {sorted(target_protocols)}')
    lines += ['', f'Target-image protocol: `{next(iter(target_protocols))}`.', '']
    if include_ablation:
        path = (root / representative / 'seeds' / str(initial_seed) / 'evaluation' / 'metrics.jsonl'
                if initial_seed is not None else root / representative / 'evaluation' / 'metrics.jsonl')
        rows = selected(read_jsonl(path), ABLATION_METHODS)
        missing = set(ABLATION_METHODS.values()) - {r['method'] for r in rows}
        if missing:
            raise ValueError(f'{representative} is missing ablation methods: {sorted(missing)}')
        report(rows, out / 'ablation')
        lines += ['', '## Ablation', '',
                  f'- Representative VLM: `{representative}`',
                  '- [Ablation table](ablation/paper-report.md)', '']
    (out / 'README.md').write_text('\n'.join(lines))


def build_initial_seed_suite(root, aliases, representative, out, seeds,
                             include_ablation=True, include_ui2coden=False,
                             own_initial_alias='ui2coden', require_complete_metrics=False, revision_rounds=None):
    """Aggregate fixed-model results across independently sampled initial VLM drafts."""
    root = Path(root); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Initial seeds must be nonempty and distinct')
    for index, seed in enumerate(seeds):
        build(root, aliases, representative, out / 'by-initial-seed' / str(seed),
              include_ablation and index == 0, include_ui2coden, revision_rounds or 3, True, seed,
              own_initial_alias, require_complete_metrics, revision_rounds=revision_rounds)
    lines = ['# Initial-VLM-seed experiment', '',
             'The policy checkpoints and target observations are fixed. Initial VLM drafts vary by generation seed.',
             'Mean ± sample SD is across initial VLM seeds; page-bootstrap CIs are in each seed report.', '']
    common_targets = common_checkpoints = None
    for alias in aliases:
        reports = []
        initial_hashes = []
        target_hashes = []
        policy_hashes = []
        for seed in seeds:
            alias_root = root / alias / 'seeds' / str(seed)
            prepared = _prepared_by_id(alias_root)
            config = json.loads((alias_root / 'prepare' / 'config.json').read_text())
            settings = config['settings']
            if settings['seed'] != int(seed):
                raise ValueError(f'{alias}/{seed}: preparation seed differs from directory')
            if len(seeds) > 1 and settings.get('initial_temperature', 0) <= 0:
                raise ValueError(f'{alias}/{seed}: initial VLM sampling is disabled')
            initial_hashes.append({page: _sha(row['methods']['initial']['html']) for page, row in prepared.items()})
            target_hashes.append({page: _sha(row['screenshot']) for page, row in prepared.items()})
            repair_config = json.loads((alias_root / 'repair' / 'config.json').read_text())
            policy_hashes.append(repair_config['checkpoints'])
            summary_path = out / 'by-initial-seed' / str(seed) / 'main' / alias / 'paper-summary.json'
            reports.append(json.loads(summary_path.read_text()))
        if any(target != target_hashes[0] for target in target_hashes[1:]):
            raise ValueError(f'{alias}: target PNG or page set changed across initial seeds')
        if any(checkpoints != policy_hashes[0] for checkpoints in policy_hashes[1:]):
            raise ValueError(f'{alias}: TUIDE/detector checkpoint changed across initial seeds')
        if common_targets is not None and target_hashes[0] != common_targets:
            raise ValueError(f'{alias}: target PNG/page set differs across initial VLMs')
        if common_checkpoints is not None and policy_hashes[0] != common_checkpoints:
            raise ValueError(f'{alias}: policy checkpoint differs across initial VLMs')
        common_targets, common_checkpoints = target_hashes[0], policy_hashes[0]
        if len(seeds) > 1 and all(initial == initial_hashes[0] for initial in initial_hashes[1:]):
            raise ValueError(f'{alias}: all initial HTML outputs are identical across seeds')
        methods = {item['method'] for item in reports[0]['methods']}
        if any({item['method'] for item in report_['methods']} != methods for report_ in reports[1:]):
            raise ValueError(f'{alias}: method set changed across initial seeds')
        result = {'protocol': 'fixed-model-vary-initial-vlm-seed-v1', 'initial_seeds': seeds,
                  'target_image_protocol': reports[0]['target_image_protocol'],
                  'complete_metrics_required': require_complete_metrics,
                  'fixed_checkpoint_hashes': policy_hashes[0], 'methods': {}}
        for method in sorted(methods):
            items = [next(item for item in report_['methods'] if item['method'] == method) for report_ in reports]
            n_pages = [item['n_pages'] for item in items]
            if len(set(n_pages)) != 1:
                raise ValueError(f'{alias}/{method}: page counts differ across seeds')
            metrics = {}
            for metric in items[0]['metrics']:
                values = [item['metrics'][metric]['mean'] for item in items]
                if all(value is not None for value in values):
                    metrics[metric] = {'mean': statistics.mean(values),
                                       'sd': statistics.stdev(values) if len(values) > 1 else None,
                                       'per_seed': dict(zip(seeds, values))}
            if method != 'Initial VLM':
                rates = [next((gain['improved_page_rate'] for gain in report_['paired']
                               if gain['method'] == method and gain['baseline'] == 'Initial VLM'
                               and gain['metric'] == 'dom_box_iou'), None) for report_ in reports]
                if all(rate is not None for rate in rates):
                    metrics['improved_page_rate'] = {'mean': statistics.mean(rates),
                                                     'sd': statistics.stdev(rates) if len(rates) > 1 else None,
                                                     'per_seed': dict(zip(seeds, rates))}
            result['methods'][method] = {'n_pages': n_pages[0], 'metrics': metrics}
        destination = out / 'main' / alias; destination.mkdir(parents=True, exist_ok=True)
        (destination / 'initial-seed-summary.json').write_text(json.dumps(result, indent=2) + '\n')
        keys = [('official_position', 'Position ↑'), ('official_clip', 'CLIP ↑'),
                ('dom_box_iou', 'Box IoU ↑'), ('improved_page_rate', 'Improved pages (%) ↑')]
        table = ['| Repair method | ' + ' | '.join(label for _, label in keys) + ' |',
                 '|---|' + '---:|' * len(keys)]
        for method, item in result['methods'].items():
            cells = []
            for metric, _ in keys:
                value = item['metrics'].get(metric)
                if value is None: cells.append('--'); continue
                factor = 100 if metric == 'improved_page_rate' else 1
                mean = value['mean'] * factor
                sd = value['sd']
                cells.append(f'{mean:.3f} ± {sd * factor:.3f}' if sd is not None else f'{mean:.3f}')
            table.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
        (destination / 'main-table.md').write_text('\n'.join(['# Fixed-model initial-seed comparison', '', *table, '']))
        lines.append(f'- [{alias}](main/{alias}/main-table.md)')
    (out / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Separate completed runs into main and ablation paper tables')
    parser.add_argument('--root', required=True)
    parser.add_argument('--aliases', required=True, help='Comma-separated VLM aliases')
    parser.add_argument('--representative', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--skip-ablation', action='store_true')
    parser.add_argument('--include-ui2coden', action='store_true')
    parser.add_argument('--ui2coden-rounds', type=int, default=3)
    parser.add_argument('--require-three-rounds', action='store_true')
    parser.add_argument('--revision-rounds', type=int, choices=range(1, 101))
    parser.add_argument('--initial-seeds', help='Comma-separated generation seeds; policy checkpoints remain fixed')
    parser.add_argument('--own-initial-alias', default='ui2coden')
    parser.add_argument('--require-complete-metrics', action='store_true')
    args = parser.parse_args()
    aliases = [x.strip() for x in args.aliases.split(',') if x.strip()]
    if not aliases or args.representative not in aliases:
        raise ValueError('representative must be one of aliases')
    if args.initial_seeds:
        seeds = [part.strip() for part in args.initial_seeds.split(',')]
        if not all(part.isdigit() for part in seeds):
            raise ValueError('Initial seeds must be nonnegative integers')
        build_initial_seed_suite(args.root, aliases, args.representative, args.out, seeds,
                                 not args.skip_ablation, args.include_ui2coden,
                                 args.own_initial_alias, args.require_complete_metrics, args.revision_rounds)
    else:
        build(args.root, aliases, args.representative, args.out, not args.skip_ablation,
              args.include_ui2coden, args.ui2coden_rounds, args.require_three_rounds,
              own_initial_alias=args.own_initial_alias, require_complete_metrics=args.require_complete_metrics,
              revision_rounds=args.revision_rounds)


if __name__ == '__main__':
    main()

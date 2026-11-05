"""Aggregate actual evaluation stage events, never replayed score-cache timings."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

CLIP_STAGES = {'clip', 'clip_image_preprocess', 'clip_encode', 'clip_feature_cache_lookup'}


def summarize_logs(paths):
    paths = [Path(p).resolve() for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError('The same log was supplied more than once')
    stages = defaultdict(lambda: {'seconds': 0., 'calls': 0, 'failed_calls': 0, 'max_seconds': 0.})
    cache_hits = Counter()
    outcomes = Counter()
    pending = Counter()
    malformed = invalid = events = 0
    for path in paths:
        with path.open(errors='replace') as handle:
            for line in handle:
                if '"evaluation_stage"' not in line:
                    continue
                try:
                    event = json.loads(line[line.index('{'):])
                except (ValueError, json.JSONDecodeError):
                    malformed += 1
                    continue
                if not isinstance(event, dict) or not isinstance(event.get('evaluation_stage'), str):
                    invalid += 1
                    continue
                stage, status = event['evaluation_stage'], event.get('status')
                events += 1
                key = (str(path), event.get('id'), event.get('method'), event.get('output'), stage)
                if status == 'cache_hit':
                    cache_hits[stage] += 1
                    continue
                if status == 'started':
                    pending[key] += 1
                    continue
                if status not in ('completed', 'failed'):
                    continue
                if stage == 'method':
                    outcomes[status] += 1
                # Method summaries contain nested totals already represented by
                # individual stage events. Inclusive durations are never summed.
                if 'exclusive_seconds' not in event:
                    continue
                duration = event['exclusive_seconds']
                if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
                    invalid += 1
                    continue
                if pending[key]:
                    pending[key] -= 1
                row = stages[stage]
                row['seconds'] += duration
                row['calls'] += 1
                row['failed_calls'] += status == 'failed'
                row['max_seconds'] = max(row['max_seconds'], duration)
    total = sum(row['seconds'] for row in stages.values())
    ranked = []
    for rank, (stage, row) in enumerate(sorted(stages.items(), key=lambda item: (-item[1]['seconds'], item[0])), 1):
        ranked.append({'rank': rank, 'stage': stage, **row,
                       'percent': row['seconds'] / total * 100 if total else 0.,
                       'mean_seconds': row['seconds'] / row['calls']})
    clip = sum(stages[s]['seconds'] for s in CLIP_STAGES if s in stages)
    # Separate initialization; it includes CLIP loading but is not per-page CLIP work.
    groups = Counter()
    for row in ranked:
        name = row['stage']
        group = ('CLIP total' if name in CLIP_STAGES else
                 'Setup/reference' if name in {'reference_prepare', 'official_initialization_including_clip_load'} else
                 'Rendering' if name in {'render_dom_snapshot', 'official_render'} else
                 'Block extraction' if name in {'block_extraction', 'block_html_preprocess', 'block_pixel_extraction'} else
                 'Block matching' if name == 'block_matching' else
                 'DOM matching' if name == 'dom_box_matching' else 'Other')
        groups[group] += row['seconds']
    return {
        'protocol': 'exclusive-evaluation-log-events-v1', 'logs': list(map(str, paths)),
        'measured_stage_seconds': total, 'clip_total_seconds': clip,
        'clip_percent': clip / total * 100 if total else 0.,
        'stages': ranked,
        'groups': [{'group': name, 'seconds': seconds, 'percent': seconds / total * 100 if total else 0.}
                   for name, seconds in groups.most_common()],
        'cache_reuse': {'page_score_hits': cache_hits.pop('page', 0),
                       'method_score_hits': cache_hits.pop('method', 0),
                       'internal_operation_hits': dict(sorted(cache_hits.items()))},
        'method_outcomes': dict(outcomes), 'stage_event_count': events,
        'unfinished_stage_calls': sum(pending.values()),
        'malformed_event_lines': malformed, 'invalid_event_lines': invalid,
        'notes': ['Only recorded exclusive stage durations are summed, including failed attempts.',
                  'Cached historical scores/timings are not added. Cache-hit event counts are accesses, not unique pages.',
                  'No timings are inferred for old uninstrumented logs. Pending/killed stages have incomplete timing.',
                  'This is measured work time, not total experiment wall time; generation, uninstrumented I/O and log overhead are excluded.',
                  'Multiple logs must cover distinct process executions, not mirrored/copied logs.'],
    }


def markdown(report):
    lines = ['# Evaluation timing', '',
             f"Measured stages: {report['measured_stage_seconds']:.3f} s",
             f"CLIP total (preprocessing + encoding + other CLIP work): {report['clip_total_seconds']:.3f} s ({report['clip_percent']:.2f}%)",
             '', '| Group | Seconds | Share (%) |', '|---|---:|---:|']
    for row in report['groups']:
        lines.append(f"| {row['group']} | {row['seconds']:.3f} | {row['percent']:.2f} |")
    lines += ['', 'Group totals above and stage details below describe the SAME work; do not add them together.', '',
              '| Rank | Stage (exclusive) | Seconds | Share (%) | Calls | Mean (s) | Max (s) | Failed |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in report['stages']:
        lines.append(f"| {r['rank']} | {r['stage']} | {r['seconds']:.3f} | {r['percent']:.2f} | {r['calls']} | {r['mean_seconds']:.3f} | {r['max_seconds']:.3f} | {r['failed_calls']} |")
    reuse = report['cache_reuse']
    lines += ['', '## Cache reuse', '',
              f"- Completed page-score cache accesses: {reuse['page_score_hits']}",
              f"- Identical HTML method-score cache accesses: {reuse['method_score_hits']}"]
    lines += [f'- Internal {name}: {count} hits' for name, count in reuse['internal_operation_hits'].items()]
    lines += ['', f"Unfinished stage calls: {report['unfinished_stage_calls']}",
              f"Malformed/invalid timing lines: {report['malformed_event_lines']}/{report['invalid_event_lines']}",
              '', *[f'- {note}' for note in report['notes']]]
    if not report['stages']:
        lines += ['', '**No completed timed stages found. Historical cache entries cannot reconstruct past stage timings.**']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', nargs='+', required=True, help='Distinct execution log files; safe to read while running')
    parser.add_argument('--out', required=True, help='Report directory (never changes experiment outputs)')
    args = parser.parse_args()
    report = summarize_logs(args.log)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'evaluation-timing.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    text = markdown(report)
    (out/'evaluation-timing.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()

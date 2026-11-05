"""Preflight and export the three-dataset, four-initial-model paper comparison."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import tempfile

from .benchmarks import discover_design2code
from .ir import read_jsonl

DATASETS = [('webui', 'WebUI'), ('design2code', 'Design2Code'), ('design2code-hard', 'Design2Code-Hard')]
MODELS = [('llava-next-7b', 'LLaVA-NeXT-7B'), ('glm4v-9b-thinking', 'GLM-4.1V-9B'),
          ('websight-8b', 'WebSight VLM-8B'),
          ('ui2coden', r'UI2Code$^{n}$')]
# These public checkpoints have an image-only drafting protocol, not a chat
# protocol taking current HTML and two images for self-revision.
NO_SELF_REVISION_ALIASES = frozenset({'websight-8b', 'design2code-18b'})
# Retained only for explicitly requested legacy reports and individual runs.
LEGACY_MODELS = [('qwen3vl8b', 'Qwen3-VL-8B'), ('glm4v-9b-thinking', 'GLM-4.1V-9B'),
                 ('gemma3-12b', 'Gemma-3-12B'), ('llava-next-7b', 'LLaVA-NeXT-7B'),
                 ('pixtral-12b', 'Pixtral-12B'), ('ui2coden', r'UI2Code$^{n}$')]
MODEL_LABELS = dict([*LEGACY_MODELS, ('design2code-18b', 'Design2Code-18B-v0'), *MODELS])

METRICS = ('official_position', 'official_clip', 'dom_box_iou', 'improved_page_rate')


def check_sources(root=None, manifest=None, dataset='design2code', view='default_1280-720', limit=0):
    if bool(root) == bool(manifest):
        raise ValueError('Supply exactly one dataset root or manifest')
    source = Path(manifest or root).resolve()
    if manifest and not source.is_file():
        raise ValueError(f'{dataset}: manifest does not exist: {source}')
    if root and not source.is_dir():
        raise ValueError(f'{dataset}: dataset directory does not exist: {source}')
    if manifest:
        rows = list(read_jsonl(manifest))
    elif dataset == 'webui':
        from .webui_pages import discover_webui
        rows = discover_webui(root, view, limit, recorded_boxes=False)
    else:
        rows = [dict(id=i, screenshot=str(p), html=str(h)) for i,p,h in discover_design2code(root)]
    if limit: rows = rows[:limit]
    if not rows:
        detail = ('manifest has no records' if manifest else
                  'no matching same-name .png/.html pairs found recursively' if dataset != 'webui' else
                  f'no WebUI records for view={view}')
        raise ValueError(f'{dataset}: {detail}: {source}')
    duplicates = [(key, n) for key, n in Counter(str(r['id']) for r in rows).items() if n > 1]
    if duplicates:
        raise ValueError(f'{dataset}: duplicate page/view IDs in {source}; examples={duplicates[:5]}')
    for row in rows:
        if row.get('split','test') != 'test':
            raise ValueError(f'{row["id"]}: paper evaluation requires the held-out test split')
        for field in ('html','screenshot'):
            if not row.get(field) or not Path(row[field]).is_file():
                raise ValueError(f'{row["id"]}: missing {field}; supply reference HTML/PNG pairs, '
                                 'for example css/pages-test.jsonl, not a page-ID-only list')
    return rows


def check_official(repo):
    """Run before any VLM loading: verify dependencies, Chromium and actual scores."""
    if not repo:
        raise ValueError('OFFICIAL_REPO is required for the paper Position/CLIP columns')
    from .official_metrics import OfficialMetrics
    from .html_bridge import HtmlBrowser
    scorer = OfficialMetrics(repo)
    html = '<html><body style="margin:0"><div style="margin:20px;width:180px;height:60px">Metric check</div></body></html>'
    with tempfile.TemporaryDirectory(prefix='paper-metrics-') as tmp, HtmlBrowser() as browser:
        scores = scorer.score(browser, html, html, [320,240], tmp)
    for key in ('official_position','official_clip'):
        if not math.isfinite(scores[key]) or scores[key] < .99:
            raise ValueError(f'Official metric identity check failed: {key}={scores[key]}')
    return scorer.source_sha


def export_matrix(root, out, datasets=None, include_tuide=True, revision_rounds=3, models=None):
    root=Path(root); out=Path(out)
    selected_models = MODELS if models is None else [(key, MODEL_LABELS[key]) for key in models]
    if not selected_models or len({k for k, _ in selected_models}) != len(selected_models):
        raise ValueError('Select distinct, nonempty models')
    selected_datasets=DATASETS if datasets is None else [(key,dict(DATASETS)[key]) for key in datasets]
    if not selected_datasets or len({k for k,_ in selected_datasets})!=len(selected_datasets):
        raise ValueError('Select distinct, nonempty datasets')
    records={}; seeds=checkpoints=None
    for dataset,_ in selected_datasets:
        counts=set()
        for alias,_ in selected_models:
            path=root/dataset/'paper/main'/alias/'initial-seed-summary.json'
            record=json.loads(path.read_text())
            if record.get('cohort_protocol') != 'initial-success-shared-v1':
                raise ValueError(f'{path}: regenerate reports with Initial-success cohort reporting')
            counts.add(tuple(record['initial_coverage_by_seed'][str(seed)]['attempted'] for seed in record['initial_seeds']))
            if not record.get('complete_metrics_required'):
                raise ValueError(f'{path}: regenerate with --require-complete-metrics')
            if record['target_image_protocol'] != 'local-reference-render-v1':
                raise ValueError(f'{path}: expected local reference render protocol')
            if seeds is not None and record['initial_seeds'] != seeds:
                raise ValueError('Initial generation seeds must match across datasets/models')
            if checkpoints is not None and record['fixed_checkpoint_hashes'] != checkpoints:
                raise ValueError('Policy/detector checkpoints must match across datasets/models')
            seeds=record['initial_seeds']; checkpoints=record['fixed_checkpoint_hashes']
            required = ['Initial VLM', 'UI2CodeN self-revision', 'TUIDE'] if alias=='ui2coden' else [
                'Initial VLM','D2C Self-Revision (RGB)','UI2CodeN polishing','TUIDE']
            if alias in NO_SELF_REVISION_ALIASES:required.remove('D2C Self-Revision (RGB)')
            if not include_tuide:required.remove('TUIDE')
            if set(record['methods']) != set(required):
                raise ValueError(f'{path}: incomplete method group')
            for method in required:
                item=record['methods'][method]
                expected_counts = {str(seed): record['initial_coverage_by_seed'][str(seed)]['successful'] for seed in seeds}
                if item['n_pages_by_seed'] != expected_counts:
                    raise ValueError(f'{path}: inconsistent Initial-success cohort for {method}')
                for metric in METRICS:
                    if method=='Initial VLM' and metric=='improved_page_rate': metric='initial_success_rate'
                    if metric != 'initial_success_rate' and any(n == 0 for n in expected_counts.values()): continue
                    value=item['metrics'].get(metric)
                    if not value or not isinstance(value.get('mean'), (int,float)) or not math.isfinite(value['mean']):
                        raise ValueError(f'{path}: missing {method}/{metric}')
                    if len(seeds)>1 and (value['sd'] is None or not math.isfinite(value['sd'])):
                        raise ValueError(f'{path}: missing SD for {method}/{metric}')
            records[(dataset,alias)]=record
        if len(counts)!=1 or any(n <= 0 for n in next(iter(counts))):
            raise ValueError(f'{dataset}: all model/method groups must use the same nonempty test set')
    lines=[r'\begin{table*}[t]',r'\centering',
           r'\caption{Layout repair results using locally rendered reference HTML on '+', '.join(label for _,label in selected_datasets)+'. '
           r'Within each dataset and initial VLM, repair methods share the same initial HTML. '
           r'Self Revision and UI2Code$^{n}$ polishing use '+str(revision_rounds)+' refinement rounds. '
           r'If refinement fails, the last available implementation is retained and remains in the evaluation. '
           r'WebSight and Design2Code use their official image-only drafting protocols; unsupported self-revision is omitted. '
           r'Quality is measured on the shared set of successfully generated Initial pages within each model and seed. '
           r'The last column reports Initial generation success over all attempted inputs, and Box IoU improvement over Initial for repair methods. '
           r'Results are means across initial-generation seeds with fixed model checkpoints; $\pm$ denotes sample standard deviation when multiple seeds are used.}',
           r'\label{tab:vlm-quality}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{ll*{'+str(len(selected_datasets))+r'}{cccc}}',r'\hline',
           r'\multirow{2}{*}{Initial VLM} & \multirow{2}{*}{Repair method} & '
           +' & '.join(r'\multicolumn{4}{c}{'+label+'}' for _,label in selected_datasets)+r' \\',
           ''.join(r'\cline{'+f'{3+4*i}-{6+4*i}'+'}' for i in range(len(selected_datasets))),
           '& & '+' & '.join([r'Position $\uparrow$',r'CLIP $\uparrow$',r'Box IoU $\uparrow$',r'Success / Improved (\%) $\uparrow$']*len(selected_datasets))+r' \\',r'\hline']
    coverage={}
    for alias,label in selected_models:
        methods=[('Initial VLM','Initial')]
        methods += [('UI2CodeN self-revision','Self Revision')] if alias=='ui2coden' else [
            ('D2C Self-Revision (RGB)','Self Revision'),('UI2CodeN polishing',r'UI2Code$^{n}$')]
        if alias in NO_SELF_REVISION_ALIASES:
            methods = [(method, title) for method, title in methods if method != 'D2C Self-Revision (RGB)']
        if include_tuide:methods += [('TUIDE','TUIDE')]
        for index,(method,title) in enumerate(methods):
            cells=[]
            for dataset,_ in selected_datasets:
                item=records[dataset,alias]['methods'][method]
                coverage.setdefault(dataset,{})[alias]=records[dataset,alias]['initial_coverage_by_seed']
                for metric in METRICS:
                    if method=='Initial VLM' and metric=='improved_page_rate': metric='initial_success_rate'
                    value=item['metrics'].get(metric)
                    if value is None: cells.append('--'); continue
                    factor=100 if metric in ('improved_page_rate','initial_success_rate') else 1
                    mean=f'{value["mean"]*factor:.3f}'
                    cells.append('$'+mean+(rf' \pm {value["sd"]*factor:.3f}' if value['sd'] is not None else '')+'$')
            name=rf'\multirow{{{len(methods)}}}{{*}}{{{label}}}' if index==0 else ''
            lines.append(name+' & '+title+' & '+' & '.join(cells)+r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}%',r'}',r'\end{table*}','']
    out.mkdir(parents=True,exist_ok=True)
    (out/'vlm-quality.tex').write_text('\n'.join(lines))
    (out/'coverage.json').write_text(json.dumps(dict(initial_seeds=seeds,pages=coverage),indent=2)+'\n')
    return out/'vlm-quality.tex'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    pre=sub.add_parser('check')
    pre.add_argument('--root'); pre.add_argument('--manifest')
    pre.add_argument('--dataset',choices=['webui','design2code'],default='design2code')
    pre.add_argument('--webui-view',default='default_1280-720')
    pre.add_argument('--limit',type=int,default=0)
    pre.add_argument('--official-repo',required=True)
    export=sub.add_parser('export'); export.add_argument('--root',required=True); export.add_argument('--out',required=True)
    export.add_argument('--datasets',nargs='+',choices=[key for key,_ in DATASETS])
    export.add_argument('--models', nargs='+', choices=list(MODEL_LABELS), help='Defaults to the current official four-model roster; use explicit aliases for legacy reports')
    export.add_argument('--skip-tuide',action='store_true')
    export.add_argument('--revision-rounds',type=int,choices=range(1,101),default=3)
    args=parser.parse_args()
    if args.command=='check':
        rows=check_sources(args.root,args.manifest,args.dataset,args.webui_view,args.limit)
        sha=check_official(args.official_repo)
        print(json.dumps(dict(pages=len(rows),official_source_sha=sha,protocol='local-reference-render-v1')))
    else: print(export_matrix(args.root,args.out,args.datasets,not args.skip_tuide,args.revision_rounds,args.models))


if __name__=='__main__': main()

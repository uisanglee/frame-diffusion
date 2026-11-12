"""Paired held-out CSS magnitude experiment; fixed DOM, sites and directions."""
import argparse
import copy
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

from .ir import read_jsonl, write_json, write_jsonl

LEVELS = {'clean': 0., 'mild': .1, 'medium': .5, 'strong': 1.}


def choose_sites(row, count, seed):
    from .tree_online import mutation_sites
    candidates = []
    for owner, field, value, priority in mutation_sites(row['declaration_state']):
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)px', value):
            continue
        if not any(row['current_boxes'].get(id, [0, 0, 0, 0])[2] > 0 and
                   row['current_boxes'].get(id, [0, 0, 0, 0])[3] > 0
                   for id in row['css_owners'][owner]['matches']):
            continue
        if field in ('width', 'height') and float(value[:-2]) <= 0:
            continue
        candidates.append([owner, field, value, priority])
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return [dict(edit=e, direction=rng.choice([-1, 1])) for e in candidates[:count]]


def scaled_edits(sites, strength, viewport, limits):
    """Identical declarations/signs at every level; never introduce a CSS property."""
    result = []
    if not strength:
        return result
    for site in sites:
        owner, field, value, priority = site['edit']
        old = float(value[:-2]); direction = site['direction']
        if field in ('width', 'height'):
            number = max(limits.get(owner, {}).get(field, 0.), old * 4 ** (direction * strength))
        else:
            axis = viewport[0 if field in ('margin-left', 'margin-right') else 1]
            number = old + direction * strength * .25 * axis
        number = round(number, 4)
        if abs(number - old) > 1e-6:
            result.append([owner, field, f'{number:.4f}'.rstrip('0').rstrip('.') + 'px', priority])
    return result


def build_page(browser, source, work, count, seed):
    import numpy as np
    from . import css_owners
    from .explicit_html import owner_size_limits
    from .fixed_test import boxes, pixel_mae
    from .metrics import box_metrics
    from .tree_online import abstraction
    from .vlm_first_edit_audit import context
    work.mkdir(parents=True, exist_ok=True)
    clean_html = Path(source['target_html']).read_text()
    clean = context(browser, clean_html, source)
    clean_html = browser.page.content()
    # Parse a detached document: fixed=True on the live DOM would strip fields.
    fixed = css_owners.read(browser, clean['current'], html=clean_html, fixed=True)['fixed']
    target = work / 'target.html'; target.write_text(clean_html)
    png, _ = browser.stable_capture(timeout_ms=15000)
    (work / 'target.png').write_bytes(png)
    target_boxes = boxes(browser, clean['current'], source['viewport'])
    target_masks, _, _ = abstraction(browser, copy.deepcopy(clean['current']), source['viewport'], 384)
    sites = choose_sites(clean, count, seed)
    if not sites:
        raise ValueError('No visible owners with editable numeric px declarations')
    view_id = source['id'].rsplit('/', 1)[0]
    rows = []; measurements = []
    for level, strength in LEVELS.items():
        browser.load(clean_html, source['viewport'])
        edits = scaled_edits(sites, strength, source['viewport'], owner_size_limits(clean))
        for edit in edits:
            css_owners.execute(browser, clean['css_owners'], edit)
        html = browser.page.content()
        row = context(browser, html, source)
        if row['css_owners'] != clean['css_owners']:
            raise ValueError('CSS owner identity changed')
        if css_owners.read(browser, row['current'], html=html, fixed=True)['fixed'] != fixed:
            raise ValueError('DOM, content or non-editable CSS changed')
        expected = copy.deepcopy(clean['declaration_state'])
        from .tree_edits import FIELDS
        for owner, field, value, priority in edits:
            expected[owner][FIELDS.index(field)] = [value, priority]
        edited_fields = {(e[0], FIELDS.index(e[1])) for e in edits}
        for owner, fields in enumerate(expected):
            for field_index, desired in enumerate(fields):
                actual = row['declaration_state'][owner][field_index]
                if (owner, field_index) in edited_fields:
                    # Chromium serializes CSS numeric values at finite precision.
                    valid = (actual[1] == desired[1] and actual[0].endswith('px') and
                             math.isclose(float(actual[0][:-2]), float(desired[0][:-2]), rel_tol=1e-5, abs_tol=1e-4))
                else:
                    valid = actual == desired
                if not valid:
                    raise ValueError(f'Unexpected CSS declaration mutation: {owner}/{FIELDS[field_index]}: {actual} != {desired}')
        current_path = work / f'{level}.html'; current_path.write_text(html)
        current_png = work / f'{level}.png'
        png, _ = browser.stable_capture(timeout_ms=15000); current_png.write_bytes(png)
        current_masks, _, _ = abstraction(browser, copy.deepcopy(row['current']), source['viewport'], 384)
        geometry = box_metrics(row['current_boxes'], target_boxes, source['viewport'],
                               (row['current']['nodes'][0]['id'],))
        # Witness that the original values can undo this exact corruption.
        for owner, field, _, _ in reversed(edits):
            value, priority = clean['declaration_state'][owner][FIELDS.index(field)]
            css_owners.execute(browser, clean['css_owners'], [owner, field, value, priority])
        restored, _ = browser.stable_capture(timeout_ms=15000)
        import io
        from PIL import Image
        restore_mae = float(np.abs(np.asarray(Image.open(io.BytesIO(restored)).convert('RGB'), dtype=float) -
                                   np.asarray(Image.open(work/'target.png').convert('RGB'), dtype=float)).mean()/255.)
        if restore_mae > 1e-6:
            raise ValueError(f'Teacher restoration changed pixels: {restore_mae}')
        measurement = dict(id=view_id, level=level, strength=strength, edits=edits,
            sites=sites, changed_declarations=len(edits), structure_preserved=True,
            teacher_restore_pixel_mae=restore_mae, initial_pixel_mae=pixel_mae(work/'target.png', current_png),
            initial_abstraction_mae=float((current_masks-target_masks).abs().mean()),
            initial_box_iou=geometry['box_iou'])
        measurements.append(measurement)
        rows.append({**source, **row, 'id': view_id + '/severity-' + level,
            'target_html': str(target), 'target_image': str(work/'target.png'),
            'current_html': str(current_path), 'current_image': str(current_png),
            'target_declaration_state': clean['declaration_state'],
            'target': clean['current'], 'target_boxes': target_boxes,
            'severity_level': level})
    return dict(rows=rows, measurements=measurements,
                page={'id':view_id, 'evaluation_target_boxes':target_boxes})


def prepare(args):
    from .html_bridge import HtmlBrowser
    from .policy_cause_audit import select_pages
    from .web_experiment import guard_run, digest, signature
    rows = select_pages(list(read_jsonl(args.data)), args.samples, args.seed)
    if not rows or any(r.get('split') != 'test' for r in rows):
        raise ValueError('Use nonempty held-out policy-test.jsonl; do not relabel training rows')
    out = Path(args.out).resolve()
    guard_run(out, {'contract':'same-dom-severity-v1', 'data':digest(args.data),
        'samples':args.samples,'sites':args.sites,'seed':args.seed,'levels':LEVELS,
        'code':digest(__file__),
        'render_code': {name:digest(Path(__file__).with_name(name)) for name in
            ('css_owners.py','html_bridge.py','browser.py','visual.py','explicit_html.py')},
        'sources':{r['id']:digest(r['target_html']) for r in rows}}, True)
    completed=[]; failures=[]
    with HtmlBrowser() as browser:
        for i, row in enumerate(rows, 1):
            work=out/'pages'/signature(row['id'])[:20]; cache=work/'prepared.json'
            try:
                if cache.is_file():
                    result=json.loads(cache.read_text())
                    required=[r[k] for r in result['rows'] for k in ('current_html','current_image','target_html','target_image')]
                    if not all(Path(p).is_file() for p in required):
                        raise ValueError('Cached artifacts missing; remove this page prepared.json to rebuild')
                else:
                    browser.reset_context()
                    page_seed=int(hashlib.sha256(f'{args.seed}:{row["id"]}'.encode()).hexdigest()[:16],16)
                    result=build_page(browser,row,work,args.sites,page_seed)
                    write_json(cache,result)
                completed.append(result)
                print(f'[severity {i}/{len(rows)}] {row["id"]}: ready (4 levels)',flush=True)
            except Exception as exc:
                failures.append(dict(id=row['id'],error=str(exc)))
                print(f'[severity {i}/{len(rows)}] {row["id"]}: {exc}',flush=True)
    write_json(out/'coverage.json',dict(selected=len(rows),paired_pages=len(completed),failures=failures))
    for level in LEVELS:
        write_jsonl(out/f'{level}.jsonl',[r for item in completed for r in item['rows'] if r['severity_level']==level])
    write_jsonl(out/'pages.jsonl',[item['page'] for item in completed])
    write_jsonl(out/'severity.jsonl',[m for item in completed for m in item['measurements']])
    if not completed:raise ValueError('No complete paired pages; see coverage.json')


def report(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    inputs=Path(getattr(args,'inputs',None) or out/'inputs').resolve()
    measurements=list(read_jsonl(inputs/'severity.jsonl'))
    coverage=json.loads((inputs/'coverage.json').read_text())
    lines=['# Same-DOM severity diagnosis', '', f'Coverage: {coverage}', '',
        'Same pages, editable sites and directions; only CSS magnitudes vary. All corruptions have a verified pixel-exact teacher restoration.',
        'Aligned Box IoU uses known same-DOM correspondences; it is not the cross-DOM VLM benchmark IoU.', '',
        '| Level | N | Changed declarations | Initial RGB MAE | Abstraction MAE | Initial aligned IoU |',
        '|---|---:|---:|---:|---:|---:|']
    for level in LEVELS:
        group=[r for r in measurements if r['level']==level]
        means=[statistics.mean(r[k] for r in group) for k in
               ('changed_declarations','initial_pixel_mae','initial_abstraction_mae','initial_box_iou')]
        lines.append(f'| {level} | {len(group)} | '+' | '.join(f'{v:.5f}' for v in means)+' |')
    lines += ['', '| Level | Budget | Target | N | Initial IoU | Final IoU | Delta | Improved % | Failed |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    summary=[]
    expected_ids={r['id'] for r in measurements}
    for path in sorted((out/'evaluations').glob('*/budget-*/results.jsonl')):
        level=path.parent.parent.name; budget=int(path.parent.name.split('-')[-1])
        records=list(read_jsonl(path))
        for method in sorted({r['method'] for r in records}):
            group=[r for r in records if r['method']==method]
            initial=statistics.mean(r['initial_box_iou'] for r in group)
            final=statistics.mean(r['final_box_iou'] for r in group)
            improved=100*statistics.mean(r['final_box_iou']>r['initial_box_iou']+1e-8 for r in group)
            failed=sum(bool(r['failed']) for r in group)
            if {r['page_id'] for r in group} != expected_ids:
                lines.append(f'WARNING: incomplete cohort for {level}/{budget}/{method}; do not compare these means as paired results.')
            summary.append(dict(level=level,budget=budget,method=method,n=len(group),initial=initial,final=final,delta=final-initial,improved=improved,failed=failed))
            lines.append(f'| {level} | {budget} | {method} | {len(group)} | {initial:.4f} | {final:.4f} | {final-initial:+.4f} | {improved:.1f} | {failed} |')
    nonmonotone=[]
    for id in sorted({r['id'] for r in measurements}):
        group={r['level']:r for r in measurements if r['id']==id}
        distances=[group[level]['initial_abstraction_mae'] for level in LEVELS]
        if any(b<a-1e-8 for a,b in zip(distances,distances[1:])):nonmonotone.append(id)
    lines += ['', f'Non-monotone measured abstraction distances: {len(nonmonotone)} pages (see summary.json).',
        'Nominal severity does not guarantee larger visual distance: clipping and layout constraints can suppress changes.',
        'Oracle degradation at high measured severity supports a magnitude/distribution problem. Detector-only degradation supports a perception bottleneck.',
        'Good oracle repair at strong severity does NOT prove HTML structure is the VLM cause: compare actual initial visual distances, viewport coverage and edit feasibility before drawing that conclusion.',
        'Clean controls expose harmful edits on already correct pages. Failed rollouts stay in the denominator; inspect trace files.']
    write_json(out/'summary.json',dict(results=summary,nonmonotone_pages=nonmonotone,coverage=coverage))
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(out/'README.md')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','report']);p.add_argument('--out',required=True)
    p.add_argument('--inputs',help='Report only: reuse an existing severity input directory')
    p.add_argument('--data',default='data/new/7000/css/policy-test.jsonl')
    p.add_argument('--samples',type=int,default=20);p.add_argument('--sites',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.samples<1 or args.sites<1:p.error('samples and sites must be positive')
    (prepare if args.stage=='prepare' else report)(args)

if __name__=='__main__':main()

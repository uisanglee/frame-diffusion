"""Held-out W/H/X/Y composition diagnostic for CSS replacement policies.

Category labels describe the corrupted declaration, not a claim that CSS has
independent geometric effects. Targets are used for diagnostics only.
"""
import itertools
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .ir import read_json, read_jsonl, write_json, write_jsonl
from .tree_edits import FIELDS
from .web_experiment import digest, guard_run

VERSION = 'synthetic-action-composition-v1'
CATEGORIES = {'W': 'width', 'H': 'height', 'X': 'margin-left', 'Y': 'margin-top'}
COMBINATIONS = ['+'.join(c) for n in range(1, 5) for c in itertools.combinations(CATEGORIES, n)]


def category_fields(sample):
    return {**CATEGORIES, 'X': ('margin-left','margin-right')[(sample//2)%2],
            'Y': ('margin-top','margin-bottom')[(sample//4)%2]}


def add_parsers(sub):
    p = sub.add_parser('visual-composition-build', help='Build paired synthetic W/H/X/Y held-out cases')
    p.add_argument('--out', required=True)
    p.add_argument('--samples-per-cell', type=int, default=24)
    p.add_argument('--seed', type=int, default=73129)
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser('visual-composition-evaluate', help='Evaluate fixed-budget CSS composition rollouts')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', help='Current stylesheet replacement policy checkpoint')
    p.add_argument('--detector-checkpoint')
    p.add_argument('--oracle', action='store_true', help='Use DOM target masks, explicitly labeled oracle')
    p.add_argument('--sanity', action='store_true', help='Known reverse edits; tests benchmark, not a learned baseline')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--device', default='auto')
    p.add_argument('--threshold', type=float, default=.4)
    p.add_argument('--geometry-tolerance', type=float, default=1., help='Per-coordinate recovery tolerance in CSS pixels')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')


def page_html(seed, layout, owner_kind, fields):
    rng = random.Random(seed)
    values = []
    cards = []
    rules = []
    for i in range(4):
        props = dict(width=rng.randrange(120, 171), height=rng.randrange(75, 111),
                     **{f'margin-{side}': rng.randrange(8, 25) for side in ('left','right','top','bottom')})
        values.append(props)
        css = ';'.join(f'{k}:{v}px' for k, v in props.items())
        if owner_kind == 'rule': rules.append(f'#card-{i}{{{css}}}')
        inline = f' style="{css}"' if owner_kind == 'inline' else ''
        cards.append(f'<section class="slot"><article id="card-{i}"{inline}>'
                     f'<strong>Panel {i+1}</strong><p>Sample {seed % 997}</p></article></section>')
    if layout == 'isolated':
        horizontal = fields['X'].split('-')[1]
        vertical = fields['Y'].split('-')[1]
        layout_css = ('main{display:grid;grid-template-columns:repeat(2,320px)}'
                      '.slot{width:320px;height:230px;position:relative}'
                      f'article{{position:absolute;{horizontal}:0;{vertical}:0}}')
    else:
        # Flex slots expand with their content: changes propagate to other cards.
        layout_css = 'main{display:flex;flex-wrap:wrap;width:650px;align-items:flex-start}.slot{display:flow-root;flex:none}'
    html = ('<!doctype html><html><head><meta charset="utf-8"><style>'
            '*{box-sizing:border-box}body{margin:0;background:#edf1f6;font:16px sans-serif;color:#253247}'
            'header{padding:20px;background:#243857;color:white}main{margin:24px}'
            'article{background:#b6cfe8;border:2px solid #527499;padding:8px;overflow:hidden}'
            'p{font-size:12px}'+layout_css+''.join(rules)+'</style></head><body>'
            '<header>Composition benchmark</header><main>'+''.join(cards)+'</main></body></html>')
    return html, values


def observe(browser, tree, viewport):
    boxes = browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
    boxes[tree['nodes'][0]['id']] = [0, 0, *viewport]
    return boxes


def build(args):
    from .html_bridge import HtmlBrowser
    from .plans import dom_tree
    from .visual import annotate, elements
    from . import css_owners
    if args.samples_per_cell < 1: raise ValueError('samples-per-cell must be positive')
    out = Path(args.out).resolve()
    guard_run(out, dict(kind=VERSION, seed=args.seed, samples=args.samples_per_cell), args.resume)
    records = []
    viewport = [800, 640]
    with HtmlBrowser() as browser:
        for layout in ('isolated', 'flow'):
            for sample in range(args.samples_per_cell):
                # The same clean scene is paired across combinations and placements.
                base = f'{layout}-{sample:05d}'
                work = out / base
                work.mkdir(exist_ok=True)
                cache = work / 'records.json'
                if args.resume and cache.exists():
                    cached = read_json(cache)
                    for row in cached:
                        for key, sha in row['asset_hashes'].items():
                            if digest(row[key]) != sha: raise ValueError('Cached benchmark asset changed')
                    records.extend(cached)
                    continue
                browser.reset_context()
                owner_kind = 'inline' if sample % 2 == 0 else 'rule'
                fields = category_fields(sample)
                html, values = page_html(args.seed + sample, layout, owner_kind, fields)
                dom = browser.snapshot(html, viewport, work / 'target.png', 127)
                tree, target_boxes, _ = dom_tree(dom)
                annotate(browser, tree)
                clean = browser.page.content()
                target_path = work / 'target.html'
                target_path.write_text(clean)
                owner = css_owners.read(browser, tree)
                card_ids = browser.page.evaluate("() => [0,1,2,3].map(i=>document.getElementById('card-'+i).dataset.fdId)")
                indices = [next(j for j, o in enumerate(owner['owners'])
                                if o['kind'] == owner_kind and
                                (o.get('id') == card_ids[i] if owner_kind == 'inline' else o.get('selector') == f'#card-{i}'))
                           for i in range(4)]
                local = []
                for placement in ('same', 'distributed'):
                    for combination in COMBINATIONS:
                        browser.load(clean, viewport)
                        corruption = []
                        categories = combination.split('+')
                        # Explicit severity assignment; independent of category/placement.
                        magnitude = (8, 24, 48)[sample % 3]
                        sign = -1 if (sample // 3) % 2 else 1
                        for j, category in enumerate(categories):
                            node = sample % 4 if placement == 'same' else (sample + j) % 4
                            field = fields[category]
                            original = values[node][field]
                            changed = max(16, original + sign*magnitude) if category in 'WH' else original + sign*magnitude
                            edit = [indices[node], field, f'{changed}px', '']
                            before = observe(browser, tree, viewport)
                            css_owners.execute(browser, owner['owners'], edit)
                            after = observe(browser, tree, viewport)
                            if before == after: raise ValueError(f'Invisible corruption: {base} {edit}')
                            corruption.append(dict(category=category, edit=edit, node_id=card_ids[node]))
                        current_boxes = observe(browser, tree, viewport)
                        state = css_owners.read(browser, tree)['state']
                        name = f'{placement}-{combination}'
                        current_path = work / f'{name}.html'
                        current_path.write_text(browser.page.content())
                        image_path = work / f'{name}.png'
                        browser.page.screenshot(path=str(image_path), animations='disabled')
                        row = dict(id=f'{base}/{name}', group=f'{VERSION}-{args.seed}-{base}', split='test',
                                   base_scene=base, scene_seed=args.seed+sample,
                                   combination=combination, category_count=len(categories),
                                   placement=placement, layout=layout, owner_kind=owner_kind, category_fields=fields,
                                   magnitude_px=magnitude, affected_nodes=len({c['node_id'] for c in corruption}),
                                   viewport=viewport, current=tree, target_boxes=target_boxes,
                                   initial_boxes=current_boxes, target_elements=elements(tree,target_boxes,viewport),
                                   css_owners=owner['owners'], target_state=owner['state'], initial_state=state,
                                   corruption=corruption, target_html=str(target_path),
                                   target_image=str(work/'target.png'), current_html=str(current_path), current_image=str(image_path))
                        row['asset_hashes'] = {k:digest(row[k]) for k in ('target_html','target_image','current_html','current_image')}
                        local.append(row)
                write_json(cache, local)
                records.extend(local)
                print(f'{base}: {len(local)} verified composition cases', flush=True)
    write_jsonl(out/'manifest.jsonl', records)
    write_json(out/'summary.json', dict(kind=VERSION, cases=len(records), base_scenes=2*args.samples_per_cell,
        combinations=COMBINATIONS, categories={'W':['width'],'H':['height'],
            'X':['margin-left','margin-right'],'Y':['margin-top','margin-bottom']},
        note='Paired variants share base scenes. Flow effects can couple axes.'))


def distance(a, b):
    return sum(x != y for row, target in zip(a[1:], b[1:]) for x, y in zip(row, target))


def geometry_error(boxes, target):
    return statistics.mean(abs(boxes[k][j]-b[j]) for k,b in target.items() for j in range(4))


def diagnostic(row, states, boxes, tolerance):
    """All diagnostics are computed after rollout; never used to choose actions."""
    target = row['target_state']
    d = [distance(s, target) for s in states]
    changed = [(c['edit'][0], FIELDS.index(c['edit'][1])) for c in row['corruption']]
    clean = [(i,j) for i in range(1,len(target)) for j in range(len(FIELDS)) if (i,j) not in changed]
    geometric = [geometry_error(b,row['target_boxes']) for b in boxes]
    recovered = [all(abs(b[k][j]-t[j]) <= tolerance for k,t in row['target_boxes'].items() for j in range(4)) for b in boxes]
    intact = [k for k,t in row['target_boxes'].items()
              if k != row.get('current',{}).get('nodes',[{'id':None}])[0]['id']
              and all(abs(boxes[0][k][j]-t[j])<=tolerance for j in range(4))]
    steps = len(states)-1
    result = dict(initial_symbolic_distance=d[0], final_symbolic_distance=d[-1],
        symbolic_full_recovery=float(d[-1]==0), geometric_full_recovery=float(recovered[-1]),
        ever_geometric_recovery=float(any(recovered)),
        first_recovery_step=next((i for i,v in enumerate(recovered) if v),None),
        category_recovery=sum(states[-1][i][j]==target[i][j] for i,j in changed)/len(changed),
        collateral_declaration_rate=sum(states[-1][i][j]!=target[i][j] for i,j in clean)/len(clean) if clean else 0.,
        collateral_geometry_rate=sum(any(abs(boxes[-1][k][j]-row['target_boxes'][k][j])>tolerance
            for j in range(4)) for k in intact)/len(intact) if intact else None,
        mean_symbolic_distance_reduction=(d[0]-d[-1])/steps if steps else None,
        first_step_improving=float(d[1]<d[0]) if steps else None,
        improving_action_rate=sum(y<x for x,y in zip(d,d[1:]))/steps if steps else None,
        worsening_action_rate=sum(y>x for x,y in zip(d,d[1:]))/steps if steps else None,
        no_change_action_rate=sum(y==x for x,y in zip(d,d[1:]))/steps if steps else None,
        geometry_improving_action_rate=sum(y<x-1e-6 for x,y in zip(geometric,geometric[1:]))/steps if steps else None,
        initial_geometry_error_px=geometric[0], final_geometry_error_px=geometric[-1])
    for category in row['combination'].split('+'):
        c = next(c for c in row['corruption'] if c['category']==category)
        i,field,_,_ = c['edit']; j = FIELDS.index(field)
        result[f'recovered_{category}'] = float(states[-1][i][j]==target[i][j])
    return result


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        keys = ['overall', f"combination/{row['combination']}", f"count/{row['category_count']}",
                f"layout/{row['layout']}", f"placement/{row['placement']}", f"owner/{row['owner_kind']}",
                f"magnitude/{row['magnitude_px']}", f"cell/{row['layout']}/{row['placement']}/{row['combination']}"]
        for key in keys: groups[key].append(row)
    result = {}
    for key, rows in groups.items():
        stats = dict(n=len(rows), base_scenes=len({r['base_scene'] for r in rows}),
                     seed_clusters=len({r['scene_seed'] for r in rows}),
                     failed_rate=sum(r['failed'] for r in rows)/len(rows))
        metrics = set().union(*(r['metrics'] for r in rows))
        for metric in sorted(metrics):
            vals = [r['metrics'].get(metric) for r in rows if r['metrics'].get(metric) is not None]
            stats[metric] = statistics.mean(vals) if vals else None
            stats[metric+'_n'] = len(vals)
        result[key] = stats
    return result


def evaluate(args):
    import torch
    from . import css_owners
    from .html_bridge import HtmlBrowser
    from .metrics import box_metrics
    from .train import select_device
    from .visual import load_policy
    from .visual_experiment import rollout
    from .visual_train import load_detector
    if args.steps<1 or args.limit<0 or args.geometry_tolerance<0 or not 0<=args.threshold<=1:
        raise ValueError('Invalid evaluation budget/tolerance/threshold')
    if not args.sanity and not args.checkpoint: raise ValueError('Supply --checkpoint or --sanity')
    device = select_device(args.device)
    if device=='cpu': torch.set_num_threads(args.cpu_threads)
    policy = parser = None
    checkpoints = []
    if not args.sanity:
        policy, ck = load_policy(args.checkpoint,device); checkpoints.append(ck)
        if policy.cfg.policy_head!='replacement' or not policy.cfg.stylesheets:
            raise ValueError('Use the current stylesheet replacement policy')
        if args.oracle and policy.cfg.mode!='abstract': raise ValueError('Oracle only applies to abstract policies')
        if policy.cfg.mode=='abstract' and not args.oracle:
            if not args.detector_checkpoint: raise ValueError('Predicted abstraction requires detector checkpoint')
            parser, ck = load_detector(args.detector_checkpoint,device); checkpoints.append(ck)
    rows = list(read_jsonl(args.data))
    rows = rows[:args.limit] if args.limit else rows
    if not rows: raise ValueError('Empty benchmark')
    for row in rows:
        for key, sha in row['asset_hashes'].items():
            if digest(row[key])!=sha: raise ValueError(f'Benchmark asset changed: {row[key]}')
        for ck in checkpoints:
            if row['group'] in ck.get('training_groups',[]) or row['asset_hashes']['target_html'] in ck.get('training_hashes',[]):
                raise ValueError('Benchmark overlaps training data')
    out = Path(args.out).resolve()
    config = dict(kind=VERSION, data=digest(args.data), settings={k:v for k,v in vars(args).items() if k not in ('out','resume')},
                  checkpoints={p:digest(p) for p in (args.checkpoint,args.detector_checkpoint) if p})
    guard_run(out,config,args.resume)
    results = []
    with HtmlBrowser() as browser:
        for index,row in enumerate(rows):
            work = out/row['id']; work.mkdir(parents=True,exist_ok=True)
            cache = work/'result.json'
            if args.resume and cache.exists(): results.append(read_json(cache)); continue
            browser.reset_context()
            initial = Path(row['current_html']).read_text()
            html = initial; error = None; stats = {}; states = [row['initial_state']]; boxes = [row['initial_boxes']]
            try:
                if args.sanity:
                    history = []
                    for c in row['corruption'][:args.steps]:
                        i,field,_,_ = c['edit']
                        history.append(dict(action=[i,field,*row['target_state'][i][FIELDS.index(field)]]))
                    stats = dict(history=history, actions=len(history))
                else:
                    html,stats,_ = rollout(browser,initial,row['current'],row['viewport'],row['target_image'],policy,
                        parser,args.threshold,args.steps,oracle_elements=row['target_elements'] if args.oracle else None)
                    if stats.get('failed'):error=stats.get('error') or 'Rollout failed'
                # Replay applied actions after timing. No target state reaches the policy.
                browser.load(initial,row['viewport'])
                actions = [h['action'] for h in stats['history'] if h.get('action') is not None][:stats['actions']]
                for action in actions:
                    css_owners.execute(browser,row['css_owners'],action)
                    states.append(css_owners.read(browser,row['current'])['state'])
                    boxes.append(observe(browser,row['current'],row['viewport']))
                html=browser.page.content()
            except Exception as exc:
                error = str(exc)
                # Consistent retained-initial failure result, included in denominator.
                html = initial; states = [row['initial_state']]; boxes = [row['initial_boxes']]
                browser.reset_context(); browser.load(initial,row['viewport'])
            (work/'final.html').write_text(html)
            browser.page.screenshot(path=str(work/'final.png'),animations='disabled')
            metrics = diagnostic(row,states,boxes,args.geometry_tolerance)
            root = row['current']['nodes'][0]['id']
            for prefix,b in [('initial',boxes[0]),('final',boxes[-1])]:
                for k,v in box_metrics(b,row['target_boxes'],row['viewport'],(root,)).items(): metrics[prefix+'_'+k]=v
            metrics['seconds'] = stats.get('seconds')
            metrics['actions'] = len(states)-1
            record = {k:row[k] for k in ('id','base_scene','scene_seed','combination','category_count','layout','placement','owner_kind','magnitude_px')}
            record.update(failed=error is not None,error=error,metrics=metrics)
            write_json(work/'trace.json',dict(rollout=stats,symbolic_distances=[distance(s,row['target_state']) for s in states],boxes=boxes))
            write_json(cache,record); results.append(record)
            print(f'[{index+1}/{len(rows)}] {row["id"]}: recovered={metrics["geometric_full_recovery"]} error={error}',flush=True)
    write_jsonl(out/'results.jsonl',results)
    summary = summarize(results)
    write_json(out/'summary.json',summary)
    lines = ['# Synthetic Action Composition Benchmark', '',
             'Known-reverse sanity check (not model performance).' if args.sanity else ('Oracle target masks.' if args.oracle else 'Model evaluation.'),
             'Fixed-budget final states with original DOM parents retained. Replay failures retain initial state. Variants share base scenes.', '',
             '| Condition | n | failed | improving ↑ | full geometry recovery ↑ | initial IoU | final IoU ↑ | seconds ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,s in summary.items():
        if key!='overall' and not key.startswith('combination/'): continue
        improving = s['improving_action_rate']
        rate = f'{improving:.4f}' if improving is not None else 'N/A'
        seconds = f'{s["seconds"]:.4f}' if s['seconds'] is not None else 'N/A'
        lines.append(f'| {key} | {s["n"]} | {s["failed_rate"]:.3f} | {rate} | {s["geometric_full_recovery"]:.3f} | {s["initial_box_iou"]:.4f} | {s["final_box_iou"]:.4f} | {seconds} |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')

"""Isolated preparation, image-conditioned training and controlled reverse rollout."""
import copy
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from . import css_owners
from .css_diffusion import CONTRACT, Kernel, numeric_slots, slot_edit
from .html_bridge import HtmlBrowser
from .html_feedback import refresh_geometry
from .ir import read_jsonl, write_json, write_jsonl
from .tree_edits import FIELDS
from .tree_policy import TreeConfig, TreePolicy, POLICY_SCALES, batch_inputs
from .train import seed_all, select_device
from .visual import elements
from .visual_data import validate_splits
from .web_experiment import digest, guard_run


def add_parsers(sub):
    p = sub.add_parser('visual-css-diffusion-prepare', help='Create separate quantized clean CSS data; never rewrite existing corpora')
    p.add_argument('--source', required=True, help='Directory with policy-{train,val,test}.jsonl and optional pages-*.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--bins', type=int, default=129)
    p.add_argument('--limit', type=int, default=0, help='Maximum source pages per split; 0 means all')
    p.add_argument('--max-nodes', type=int, default=512)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--detector-checkpoint', help='Optional frozen detector; cache quantized target abstractions once')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, default=.4)
    p = sub.add_parser('visual-css-diffusion-train')
    for key in ('data', 'out'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--mode', choices=['abstract', 'screenshot'], default='abstract')
    p.add_argument('--policy-scale', choices=['s','m','l'], default='s')
    p.add_argument('--target-source', choices=['oracle', 'detector'], default='oracle')
    p.add_argument('--device', default='auto')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-pretrained', action='store_true')
    for key, default in [('steps',30000),('batch-size',2),('accumulation',4),('eval-every',500),
                         ('log-every',50),('val-samples',128),('diffusion-steps',200),('seed',42),
                         ('cpu-threads',4),('size',384),('max-nodes',512),('token-grid',12)]:
        p.add_argument('--'+key, type=int, default=default)
    for key, default in [('lr',1e-4),('sigma-min',.35),('sigma-max',2.),('move-rate',.2),('auxiliary-weight',.01)]:
        p.add_argument('--'+key, type=float, default=default)
    p = sub.add_parser('visual-css-diffusion-evaluate', help='Controlled paired reverse rollouts, separate from existing policy evaluation')
    for key in ('data','checkpoint','out'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--start-t', type=int, default=200, help='Starting diffusion timestep and maximum reverse updates (1..200)')
    p.add_argument('--abstract-goal-threshold', type=float, default=.01,
                   help='Stop when semantic union-normalized mask error <= threshold; -1 disables')
    p.add_argument('--seed', type=int, default=90210)
    p.add_argument('--cpu-threads', type=int, default=4)


def observe(browser, row, mode):
    tree = row['current']
    boxes = browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
    boxes[tree['nodes'][0]['id']] = [0, 0, *row['viewport']]
    result = {**row, 'current':refresh_geometry(tree, boxes), 'current_boxes':boxes,
              'declaration_state':css_owners.read(browser, tree)['state']}
    if mode == 'screenshot':
        result['current_image'] = browser.page.screenshot(animations='disabled')
    return result


def apply_values(browser, row, values):
    # Apply all slots, including unchanged ones: no geometry-based rejection,
    # no winning-rule filter that would alter the specified transition law.
    if len(values) != len(row['diffusion_slots']):
        raise ValueError('Diffusion slot count mismatch')
    edits = [slot_edit(slot, value, FIELDS, row['bins']) for slot,value in zip(row['diffusion_slots'],values)]
    browser.page.evaluate('''({owners, edits})=>{
      const nodes=new Map([...document.querySelectorAll('[data-fd-id]')].map(e=>[e.dataset.fdId,e]));
      const sheets=[...document.querySelectorAll('style:not([data-framediff-static])')];
      const changed=new Set();
      for(const [index,field,value,priority] of edits){
        const owner=owners[index];let style;
        if(owner.kind==='inline')style=nodes.get(owner.id)?.style;
        else if(owner.kind==='rule'){
          let rules=sheets[owner.block]?.sheet?.cssRules,rule;
          for(const i of owner.path){rule=rules?.[i];rules=rule?.cssRules;}
          if(rule?.selectorText!==owner.selector)throw Error('CSS rule identity changed');
          style=rule.style;changed.add(owner.block);
        }
        if(!style || !CSS.supports(field,value))throw Error('Invalid diffusion CSS declaration');
        style.setProperty(field,value,priority);
      }
      for(const block of changed){const el=sheets[block];el.textContent=[...el.sheet.cssRules].map(r=>r.cssText).join('\\n');}
    }''', {'owners':row['css_owners'],'edits':edits})
    browser.executions += 1


def prepare(args):
    from .visual_train import load_detector, detect, detector_input
    if args.bins < 3 or args.bins % 2 != 1 or args.limit < 0 or not 0 <= args.threshold <= 1:
        raise ValueError('Invalid bins/limit/threshold')
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if source == out or source in out.parents:
        raise ValueError('Use an independent output directory outside the source corpus')
    signatures = {str(p):digest(p) for p in source.glob('*.jsonl') if p.name.startswith(('policy-', 'pages-'))}
    guard_run(out, {'kind':CONTRACT,'source':signatures,'bins':args.bins,'limit':args.limit,
                    'max_nodes':args.max_nodes,'threshold':args.threshold,
                    'detector':digest(args.detector_checkpoint) if args.detector_checkpoint else None}, args.resume)
    detector = load_detector(args.detector_checkpoint, select_device(args.device))[0] if args.detector_checkpoint else None
    all_rows, failures, coverage = [], [], {}
    with HtmlBrowser() as browser:
        for split in ('train','val','test'):
            path = source/f'policy-{split}.jsonl'
            if not path.exists():
                raise FileNotFoundError(path)
            pages_path = source/f'pages-{split}.jsonl'
            pages = {r['id']:r for r in read_jsonl(pages_path)} if pages_path.exists() else {}
            seen, kept = set(), []
            for row in read_jsonl(path):
                page_id = row['id'].rsplit('/',1)[0]
                if page_id in seen:
                    continue
                if args.limit and len(seen) >= args.limit:
                    break
                seen.add(page_id)
                if row['split'] != split:
                    raise ValueError('Source split mismatch')
                key = hashlib.sha256(page_id.encode()).hexdigest()[:24]
                folder = out/split/key; folder.mkdir(parents=True, exist_ok=True)
                saved = folder/'record.json'
                html_path = row.get('target_html') or pages.get(page_id,{}).get('html')
                if args.resume and saved.exists():
                    record = json.loads(saved.read_text())
                    if not html_path or record.get('source_html_sha') != digest(html_path):
                        raise ValueError('Source HTML changed; use a new preparation directory')
                    if any(not Path(record[k]).exists() for k in ('target_html','target_image')):
                        raise ValueError('Incomplete prepared asset cache')
                    if detector is not None and not Path(record['predicted_target_elements']).exists():
                        raise ValueError('Missing prepared detector cache')
                    kept.append(record); continue
                try:
                    browser.reset_context()
                    if not html_path:
                        raise ValueError('No clean target HTML; do not use initial_html')
                    browser.load(Path(html_path).read_text(), row['viewport'])
                    parsed = css_owners.read(browser, row['current'])
                    if len(parsed['owners']) > args.max_nodes:
                        raise ValueError('CSS owner capacity exceeded')
                    slots, x0, error = numeric_slots(parsed['state'], FIELDS, args.bins)
                    if not slots:
                        raise ValueError('No in-range px/% numeric declarations')
                    record = {k:copy.deepcopy(row[k]) for k in ('group','split','source_sha','viewport','current')}
                    record.update(id=page_id, css_owners=parsed['owners'], diffusion_slots=slots,
                                  x0=x0, bins=args.bins, diffusion_contract=CONTRACT, quantization_errors=error,
                                  source_html_sha=digest(html_path))
                    apply_values(browser, record, x0)
                    record = observe(browser, record, 'abstract')
                    record['target_html'] = str(folder/'target.html')
                    record['target_image'] = str(folder/'target.png')
                    Path(record['target_html']).write_text(browser.page.content())
                    browser.page.screenshot(path=record['target_image'], animations='disabled')
                    record['target_elements'] = elements(record['current'], record['current_boxes'], record['viewport'])
                    if detector is not None:
                        record['predicted_target_elements'] = str(folder/'detector.json')
                        write_json(record['predicted_target_elements'], detect(detector, detector_input(record['target_image']), args.threshold))
                    write_json(saved, record)
                    kept.append(record)
                except Exception as exc:
                    failures.append({'id':page_id,'split':split,'error':str(exc)})
                if len(seen) % 50 == 0:
                    print({'split':split,'processed':len(seen),'kept':len(kept)}, flush=True)
            write_jsonl(out/f'{split}.jsonl', kept)
            coverage[split] = {'selected':len(seen),'kept':len(kept)}
            all_rows.extend(kept)
    validate_splits(all_rows)
    write_json(out/'summary.json', {'coverage':coverage,'failures':failures,
               'note':'Quantized controlled targets; original HTML assets are unchanged. px/% only.'})
    if any(coverage[s]['kept'] == 0 for s in coverage):
        raise ValueError('Empty prepared split; inspect summary.json')
    print(coverage, flush=True)


class CSSDenoiser(TreePolicy):
    def __init__(self, cfg, bins, diffusion_steps, pretrained=True):
        super().__init__(cfg, pretrained)
        # Reuse image/CSS/owner encoder, not the replacement string decoder.
        for key in ('token','position','decoder','output'):
            delattr(self, key)
        self.time = nn.Embedding(diffusion_steps+1, cfg.hidden)
        self.bin_head = nn.Sequential(nn.Linear(cfg.hidden,cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden,6*bins))
        self.bins = bins

    def forward(self, rows, timesteps, device, predicted=False):
        batch, target, current, states = batch_inputs(self, rows, device, predicted)
        memory, _ = self.memory(batch, target, current, states)
        owner_memory = memory[:, :batch['mask'].shape[1]] + self.time(torch.tensor(timesteps, device=device))[:,None]
        logits = self.bin_head(owner_memory).reshape(len(rows), -1, 6, self.bins)
        return [logits[i, [s['owner'] for s in r['diffusion_slots']], [s['property'] for s in r['diffusion_slots']]] for i,r in enumerate(rows)]


def materialize(browser, row, xt, mode):
    browser.reset_context()
    browser.load(Path(row['target_html']).read_text(), row['viewport'])
    if css_owners.read(browser, row['current'])['owners'] != row['css_owners']:
        raise ValueError('CSS owners changed after preparation')
    apply_values(browser, row, xt)
    return observe(browser, row, mode)


def check_rows(rows, split, predicted=False):
    if not rows:
        raise ValueError('Empty '+split+' data')
    for row in rows:
        if row['split'] != split or row.get('diffusion_contract') != CONTRACT:
            raise ValueError('Wrong split or diffusion data contract')
        if not row['diffusion_slots'] or len(row['diffusion_slots']) != len(row['x0']):
            raise ValueError('Invalid diffusion slots')
        if predicted and not row.get('predicted_target_elements'):
            raise ValueError('Prepare with --detector-checkpoint for detector conditioning')


def train(args):
    for key in ('steps','batch_size','accumulation','eval_every','log_every','val_samples','cpu_threads'):
        if getattr(args,key) < 1:
            raise ValueError(key+' must be positive')
    if args.lr <= 0 or args.auxiliary_weight < 0:
        raise ValueError('Invalid loss/optimizer settings')
    seed_all(args.seed); torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device); out = Path(args.out).resolve(); data = Path(args.data)
    training = list(read_jsonl(data/'train.jsonl')); validation = list(read_jsonl(data/'val.jsonl'))
    predicted = args.target_source == 'detector'
    check_rows(training,'train',predicted); check_rows(validation,'val',predicted)
    validate_splits(training+validation)
    bins = training[0]['bins']
    if any(r['bins'] != bins for r in training+validation):
        raise ValueError('Mixed quantization grids')
    asset_paths = sorted({r[k] for r in training+validation for k in
                          ('target_html','target_image','predicted_target_elements') if k in r})
    cfg = TreeConfig(mode=args.mode, size=args.size, max_nodes=args.max_nodes, token_grid=args.token_grid,
                     stylesheets=True, existing_values_only=True, action_contract=css_owners.CONTRACT,
                     **POLICY_SCALES[args.policy_scale])
    kernel_config = dict(bins=bins,steps=args.diffusion_steps,sigma_min=args.sigma_min,sigma_max=args.sigma_max,move_rate=args.move_rate)
    signature = {'kind':CONTRACT,'model':asdict(cfg),'kernel':kernel_config,
                 'assets':{p:digest(p) for p in asset_paths},
                 'train':digest(data/'train.jsonl'),'val':digest(data/'val.jsonl'),
                 'settings':{k:v for k,v in vars(args).items() if k not in ('steps','resume','out')}}
    guard_run(out, signature, args.resume)
    kernel = Kernel(**kernel_config).to(device)
    net = CSSDenoiser(cfg,bins,args.diffusion_steps,not args.no_pretrained and not args.resume).to(device)
    optimizer = torch.optim.AdamW(net.parameters(),lr=args.lr)
    rng = random.Random(args.seed); noise = torch.Generator(device=device).manual_seed(args.seed)
    start, best = 0, float('inf')
    if args.resume:
        ck = torch.load(out/'last.pt',map_location=device,weights_only=False)
        if ck['signature'] != signature:
            raise ValueError('Resume signature mismatch')
        net.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        start,best = ck['step'],ck['best'];rng.setstate(ck['rng']);noise.set_state(ck['noise'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    val_rng = random.Random(args.seed+901)
    val_rows = val_rng.sample(validation,min(args.val_samples,len(validation)))
    started = time.perf_counter()

    def sample(browser, selected, generator, randomizer):
        times, clean, noisy, observations = [], [], [], []
        for row in selected:
            t = randomizer.randint(1,args.diffusion_steps)
            x0 = torch.tensor(row['x0'],device=device)
            xt = kernel.sample(x0,t,generator)
            observations.append(materialize(browser,row,xt.tolist(),args.mode))
            times.append(t);clean.append(x0);noisy.append(xt)
        logits = net(observations,times,device,predicted)
        losses, components = zip(*(kernel.loss(l,x,z,t,args.auxiliary_weight) for l,x,z,t in zip(logits,clean,noisy,times)))
        return torch.stack(losses).mean(), {k:sum(c[k] for c in components)/len(components) for k in components[0]}

    def log(record):
        print(record,flush=True)
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')

    with HtmlBrowser() as browser:
        for step in range(start+1,args.steps+1):
            net.train();optimizer.zero_grad(set_to_none=True);total=0.
            for _ in range(args.accumulation):
                loss, components = sample(browser,[rng.choice(training) for _ in range(args.batch_size)],noise,rng)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite diffusion loss')
                (loss/args.accumulation).backward();total+=float(loss.detach())/args.accumulation
            nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            if step % args.log_every == 0:
                log({'step':step,'loss':total,**components,'elapsed_s':time.perf_counter()-started})
            if step % args.eval_every == 0 or step == args.steps:
                net.eval();values=[]
                vg = torch.Generator(device=device).manual_seed(args.seed+902);vr = random.Random(args.seed+903)
                with torch.no_grad():
                    for row in val_rows:
                        loss,_ = sample(browser,[row],vg,vr);values.append(float(loss))
                score = sum(values)/len(values);improved=score<best;best=min(best,score)
                log({'step':step,'validation_loss':score,'validation_pages':len(values),'best':best,
                     'note':'Fixed t/noise validation; loss is not rollout geometry accuracy'})
                ck = {'kind':CONTRACT,'signature':signature,'model':net.state_dict(),'optimizer':optimizer.state_dict(),
                      'step':step,'best':best,'rng':rng.getstate(),'noise':noise.get_state(),'torch_rng':torch.get_rng_state(),
                      'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else []}
                for name in (['last.pt','best.pt'] if improved else ['last.pt']):
                    temporary=out/(name+'.tmp');torch.save(ck,temporary);temporary.replace(out/name)


def abstraction_error(target, current):
    """Soft semantic IoU distance; blank background does not dominate the score."""
    union = torch.maximum(target,current).sum()
    if float(target.sum()) <= 0 or float(union) <= 0:
        return None  # An empty detector result must never report success.
    return float((target-current).abs().sum()/union)


def evaluate(args):
    from .metrics import iou
    from .visual import semantic_masks
    threshold = getattr(args,'abstract_goal_threshold',.01)
    if threshold != -1 and not 0 <= threshold <= 1:
        raise ValueError('abstract-goal-threshold must be -1 or in [0,1]')
    if not 1 <= args.start_t <= 200:
        raise ValueError('start-t must be in 1..200')
    torch.set_num_threads(args.cpu_threads)
    device=select_device(args.device);ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if ck['kind'] != CONTRACT:raise ValueError('Not a CSS diffusion checkpoint')
    sig=ck['signature'];cfg=TreeConfig(**sig['model']);kernel=Kernel(**sig['kernel']).to(device)
    kernel.check_t(args.start_t)
    net=CSSDenoiser(cfg,kernel.bins,kernel.steps,False).to(device);net.load_state_dict(ck['model']);net.eval()
    rows=list(read_jsonl(args.data));check_rows(rows,'test',sig['settings']['target_source']=='detector')
    if args.limit<0:raise ValueError('limit must be nonnegative')
    rows=rows[:args.limit] if args.limit else rows
    out=Path(args.out).resolve();guard_run(out,{'kind':CONTRACT,'checkpoint':digest(args.checkpoint),'data':digest(args.data),
                                             'start_t':args.start_t,'seed':args.seed,'limit':args.limit,
                                             'abstract_goal_threshold':threshold,
                                             'goal_metric':'semantic-union-error-v1'},False)
    generator=torch.Generator(device=device).manual_seed(args.seed);results=[]
    with HtmlBrowser() as browser, torch.no_grad():
        for index,row in enumerate(rows):
            if row['bins'] != kernel.bins:raise ValueError('Quantization grid mismatch')
            x0=torch.tensor(row['x0'],device=device);xt=kernel.sample(x0,args.start_t,generator)
            initial=materialize(browser,row,xt.tolist(),cfg.mode);current=initial
            folder=out/str(index);folder.mkdir()
            browser.page.screenshot(path=str(folder/'initial.png'),animations='disabled')
            trace=[];started=time.perf_counter()
            target_items = (json.loads(Path(row['predicted_target_elements']).read_text())
                            if sig['settings']['target_source']=='detector' else row['target_elements'])
            target_mask = semantic_masks(target_items,row['viewport'],cfg.size)
            def error(observation):
                mask=semantic_masks(elements(observation['current'],observation['current_boxes'],row['viewport']),row['viewport'],cfg.size)
                return abstraction_error(target_mask,mask)
            score=error(current);initial_score=score
            def reached(value):return threshold >= 0 and value is not None and value <= threshold
            stop_reason='step_budget'
            for t in range(args.start_t,0,-1):
                if reached(score):
                    stop_reason='abstract_goal';break
                logits=net([current],[t],device,sig['settings']['target_source']=='detector')[0]
                probabilities=kernel.reverse(logits,xt,t)
                nxt=torch.multinomial(probabilities,1,generator=generator).squeeze(-1)
                apply_values(browser,row,nxt.tolist());current=observe(browser,row,cfg.mode)
                score=error(current)
                changes=[{'owner':s['owner'],'property':FIELDS[s['property']],
                          'before':slot_edit(s,int(old),FIELDS,row['bins'])[2],
                          'after':slot_edit(s,int(new),FIELDS,row['bins'])[2]}
                         for s,old,new in zip(row['diffusion_slots'],xt.tolist(),nxt.tolist()) if old!=new]
                trace.append({'t':t,'next_t':t-1,'changed_slots':len(changes),'changes':changes,
                              'abstract_error':score,'mean_bin_error':float((nxt-x0).abs().float().mean())})
                xt=nxt
            if reached(score):stop_reason='abstract_goal'
            seconds=time.perf_counter()-started
            ids=[n['id'] for n in row['current']['nodes'][1:]]
            def geometry(observation):return sum(iou(observation['current_boxes'][i],row['current_boxes'][i]) for i in ids)/len(ids)
            result={'id':row['id'],'initial_geometry_iou':geometry(initial),'final_geometry_iou':geometry(current),
                    'repair_seconds':seconds,'reverse_steps':len(trace),'max_reverse_steps':args.start_t,
                    'initial_abstract_error':initial_score,'final_abstract_error':score,
                    'stop_reason':stop_reason,'abstract_goal_reached':reached(score),'trace':trace}
            (folder/'repaired.html').write_text(browser.page.content())
            browser.page.screenshot(path=str(folder/'repaired.png'),animations='disabled')
            results.append(result);write_jsonl(out/'results.jsonl',results);print(result['id'],result['final_geometry_iou'],flush=True)
    write_json(out/'summary.json',{'n':len(results),**{k:sum(r[k] for r in results)/len(results) for k in
               ('initial_geometry_iou','final_geometry_iou','repair_seconds','reverse_steps','abstract_goal_reached')},
               'abstract_goal_threshold':threshold,'goal_metric':'semantic-union-error-v1',
               'warning':'Controlled quantized CSS corruption benchmark, not end-to-end VLM HTML repair. Timing excludes initial corruption/render and final evaluation.'})


def run(args):
    return {'visual-css-diffusion-prepare':prepare,'visual-css-diffusion-train':train,
            'visual-css-diffusion-evaluate':evaluate}[args.command](args)

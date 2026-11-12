"""Read-only model/data audit. Writes diagnostics, never training weights or manifests."""
import argparse
import collections
import copy
import hashlib
import io
import json
import random
from pathlib import Path

from .ir import read_jsonl, write_json, write_jsonl
from .tree_edits import FIELDS, EditTokenizer, apply_state, repair_path
from .validation_metrics import differing_sites, declaration_equal


def select_pages(rows, count, seed):
    groups = collections.defaultdict(list)
    for r in rows:
        key = str(r.get('page_id') or r['id'].split('/')[0])
        groups[key].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    return [sorted(groups[k], key=lambda r:r['id'])[0] for k in keys[:count]]


def teacher_check(row):
    current, target = row['declaration_state'], row['target_declaration_state']
    edit = row.get('replacement_edit')
    if edit is None:
        return dict(status='unavailable', reason='No replacement_edit')
    tok = EditTokenizer(max(2,len(current)))
    decoded = tok.decode(tok.encode(edit,len(current)),len(current))
    before = len(differing_sites(current,target,FIELDS))
    after = len(differing_sites(apply_state(current,edit),target,FIELDS))
    state = current
    path = repair_path(current,target)
    for action in path:
        state = apply_state(state,action)
    return dict(status='ok',teacher=edit,token_roundtrip=decoded==list(edit),
                errors_before=before,errors_after=after,teacher_improves=after<before,
                full_symbolic_recovery=not differing_sites(state,target,FIELDS),path=path)


def legality(tok, edit, state, limits, existing=True):
    prefix=[]
    for token in tok.encode(edit,len(state)):
        allowed=tok.allowed(prefix,len(state),state=state if existing else None,size_limits=limits,preserve_priority=True)
        if token not in allowed:
            return dict(allowed=False,prefix=[tok.tokens[t] for t in prefix],blocked=tok.tokens[token],
                        choices=[tok.tokens[t] for t in allowed])
        prefix.append(token)
    return dict(allowed=True)


def html_text(row, target=False):
    if not target and row.get('current_html_text'):
        return row['current_html_text']
    key='target_html' if target else 'current_html'
    return Path(row[key]).read_text()


def capture(browser, row, html):
    browser.load(html,row['viewport'])
    browser.stable_capture()
    snapshot=browser.page.evaluate('''() => Object.fromEntries([...document.querySelectorAll('[data-fd-id]')].map(e=>{
        const r=e.getBoundingClientRect(),s=getComputedStyle(e);
        return [e.dataset.fdId,{box:[r.x,r.y,r.width,r.height],tag:e.tagName,
          css:Object.fromEntries(['width','height','min-width','min-height','padding','border','box-sizing','margin-left','margin-right','margin-top','margin-bottom'].map(k=>[k,s.getPropertyValue(k)])),
          inline:e.getAttribute('style'),image:e.tagName==='IMG'?{complete:e.complete,naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight}:null}];}))''')
    png=browser.page.screenshot(animations='disabled')
    return snapshot,png


def difference(before, after):
    shared=before.keys() & after.keys()
    changes=[]
    for key in sorted(shared):
        a,b=before[key]['box'],after[key]['box']
        error=max(abs(x-y) for x,y in zip(a,b))
        if error>1e-6:changes.append(dict(id=key,before=a,after=b,max_error_px=error))
    return dict(changed=changes,missing=sorted(before.keys()-after.keys()),added=sorted(after.keys()-before.keys()),
                max_box_error_px=max((c['max_error_px'] for c in changes),default=0))


def pixel_difference(a,b):
    import numpy as np
    from PIL import Image
    x=np.asarray(Image.open(io.BytesIO(a)).convert('RGB'),dtype=float)
    y=np.asarray(Image.open(io.BytesIO(b)).convert('RGB'),dtype=float)
    return float(abs(x-y).mean()/255) if x.shape==y.shape else None


def aligned_iou(observed, target):
    """Same-DOM diagnostic only, not the paper's cross-DOM matching metric."""
    values=[]
    for key in target:
        if key not in observed:
            values.append(0.);continue
        x,y,w,h=observed[key]['box'];a,b,c,d=target[key]['box']
        if c<=0 or d<=0:continue
        intersection=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
        union=max(0,w)*max(0,h)+c*d-intersection
        values.append(intersection/union if union else 0.)
    return sum(values)/len(values) if values else None


def browser_teacher(browser,row,folder):
    from . import css_owners
    folder.mkdir(parents=True,exist_ok=True)
    target,target_png=capture(browser,row,html_text(row,True))
    target_parsed=css_owners.read(browser,row['current'])
    before,before_png=capture(browser,row,html_text(row))
    parsed=css_owners.read(browser,row['current'])
    if parsed['owners']!=row['css_owners'] or target_parsed['owners']!=row['css_owners']:
        raise ValueError('Saved/live current/target CSS owner identities differ')
    current_matches=not differing_sites(parsed['state'],row['declaration_state'],FIELDS)
    target_matches=not differing_sites(target_parsed['state'],row['target_declaration_state'],FIELDS)
    for edit in repair_path(parsed['state'],target_parsed['state']):
        css_owners.execute(browser,parsed['owners'],edit)
    repaired=browser.page.content()
    after,after_png=capture(browser,row,repaired)
    actual=css_owners.read(browser,row['current'])['state']
    for name,png in [('current',before_png),('target',target_png),('teacher-repaired',after_png)]:
        (folder/f'{name}.png').write_bytes(png)
    write_json(folder/'teacher-boxes.json',dict(current=before,target=target,repaired=after))
    return dict(saved_current_matches_browser=current_matches,saved_target_matches_browser=target_matches,
                full_browser_recovery=not differing_sites(actual,target_parsed['state'],FIELDS),
                geometry=difference(target,after),pixel_mae=pixel_difference(target_png,after_png))


def decode_audit(net,inputs,row):
    import torch
    batch,target,current,states=inputs
    tok=net.tokenizer;state=states[0];limits=batch.get('explicit_owner_limits',[{}])[0]
    memory,padding=net.memory(batch,target,current,states)
    prefix=[];steps=[]
    for _ in range(tok.max_length):
        allowed=tok.allowed(prefix,len(state),state=state if net.cfg.existing_values_only else None,
                            size_limits=limits,preserve_priority=True)
        if not allowed:raise ValueError('No legal continuation')
        tokens=torch.tensor([[tok.ids['BOS'],*prefix]],device=target.device)
        scores=net.logits(memory,padding,tokens)[0,-1].float()
        chosen=allowed[int(scores[allowed].argmax())]
        def top(ids):
            ordered=sorted(ids,key=lambda t:float(scores[t]),reverse=True)[:5]
            return [dict(token=tok.tokens[t],logit=float(scores[t])) for t in ordered]
        steps.append(dict(prefix=[tok.tokens[t] for t in prefix],chosen=tok.tokens[chosen],
                          unrestricted_top5=top(range(len(tok.tokens))),allowed_top5=top(allowed)))
        prefix.append(chosen)
        if chosen==tok.ids['EOS']:break
    action=tok.decode(prefix,len(state))
    # Cross-check instrumentation against the production decoder.
    production=net.predict(*inputs)
    target_state=row['target_declaration_state']
    before=len(differing_sites(state,target_state,FIELDS))
    after=len(differing_sites(apply_state(state,action),target_state,FIELDS))
    legal=[dict(edit=e,**legality(tok,e,state,limits,net.cfg.existing_values_only))
           for e in repair_path(state,target_state)]
    return dict(action=action,production_action=production,decoder_agrees=action==production,
                errors_before=before,errors_after=after,symbolic_gain=before-after,
                same_declaration=declaration_equal(state[action[0]][FIELDS.index(action[1])],action[2:]),
                size_limits=limits,correct_edits_legality=legal,tokens=steps)


def execute_audit(browser,row,edit,folder):
    from . import css_owners
    target,_=capture(browser,row,html_text(row,True))
    before,png0=capture(browser,row,html_text(row))
    owners=css_owners.read(browser,row['current'])
    if owners['owners']!=row['css_owners']:raise ValueError('Execution owner mismatch')
    css_owners.execute(browser,owners['owners'],edit)
    output=browser.page.content()
    after,png1=capture(browser,row,output)
    state=css_owners.read(browser,row['current'])['state']
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'output.html').write_text(output);(folder/'output.png').write_bytes(png1)
    write_json(folder/'boxes.json',dict(before=before,after=after))
    actual=state[edit[0]][FIELDS.index(edit[1])]
    return dict(actual_declaration=actual,prediction_applied=declaration_equal(actual,edit[2:]),
                geometry=difference(before,after),pixel_mae=pixel_difference(png0,png1),
                diagnostic_aligned_iou_before=aligned_iou(before,target),
                diagnostic_aligned_iou_after=aligned_iou(after,target))


def trace_audit(root):
    rows=[]
    if not root or not Path(root).is_dir():return rows
    for path in sorted(Path(root).rglob('*-r1-trace.json')):
        data=json.loads(path.read_text());history=data.get('history',[])
        repeated=[];unchanged=[];counts=collections.Counter()
        for i,h in enumerate(history):
            if h.get('action'):counts[json.dumps(h['action'][1:3])]+=1
            if i:
                previous=history[i-1]
                if h.get('action') and h.get('action')==previous.get('action'):repeated.append(h['step'])
                if h.get('boxes')==previous.get('boxes'):unchanged.append(h['step'])
        rows.append(dict(path=str(path),actions=data.get('actions'),stop_reason=data.get('stop_reason'),
                         repeated_action_steps=repeated,unchanged_observation_steps=unchanged,
                         value_counts=dict(counts),error=data.get('error')))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/new/7000/css/policy-val.jsonl')
    p.add_argument('--pages',default='data/new/7000/css/pages-val.jsonl')
    p.add_argument('--stage1',default='runs/webui-color-images-v1/7000/tuide-s-abstract/replacement/abstract-stage1/best.pt')
    p.add_argument('--stage2',default='runs/webui-color-images-v1/7000/tuide-s-abstract/replacement/abstract-stage2/best.pt')
    p.add_argument('--detector',default='runs/webui-color-images-v1/7000/detector/best.pt')
    p.add_argument('--trace-root',default='runs/paper-color-pilot10-seed42-n2-target-budget-diagnosis-v1')
    p.add_argument('--out',default='runs/tuide-cause-audit-v1')
    p.add_argument('--samples',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--online-samples',type=int,default=8)
    p.add_argument('--threshold',type=float,default=.4)
    p.add_argument('--data-only',action='store_true')
    args=p.parse_args()
    if args.samples<1 or args.online_samples<0:p.error('Invalid sample counts')
    import torch
    from .html_bridge import HtmlBrowser
    from . import css_owners
    from .visual import load_policy
    from .tree_policy import batch_inputs
    from .visual_train import load_detector,detect,detector_input
    from .web_experiment import digest,guard_run
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    dependencies=[args.data,args.pages] + ([] if args.data_only else [args.stage1,args.stage2,args.detector])
    guard_run(out,dict(contract='policy-cause-audit-v1',settings=vars(args),
                       hashes={str(p):digest(p) for p in dependencies}),True)
    errors=[]
    def save_error(phase,identity,exc):
        errors.append(dict(phase=phase,id=identity,error=str(exc)))
        write_jsonl(out/'errors.jsonl',errors)
        print(errors[-1],flush=True)
    rows=select_pages(list(read_jsonl(args.data)),args.samples,args.seed)
    write_json(out/'selected-ids.json',[r['id'] for r in rows])
    # Small controlled sample from today's online generator, not historical training exposure.
    if args.online_samples:
        from .tree_online import target_pool,OnlineSampler
        try:
            seeds=copy.deepcopy(rows)
            for r in seeds:r['inspection_cache_dir']=str(out/'inspection-cache')
            targets=target_pool(seeds,cache_dir=out/'inspection-cache',workers=1)
            sampler=OnlineSampler(targets,'abstract',args.seed,export_html=True)
            try:
                for i in range(args.online_samples):
                    try:
                        r=sampler.generate(i)
                        folder=out/'online'/str(i);folder.mkdir(parents=True,exist_ok=True)
                        if r.get('current_html_text'):
                            path=folder/'current.html';path.write_text(r.pop('current_html_text'));r['current_html']=str(path)
                        rows.append(r)
                    except Exception as exc:save_error('online_generation',str(i),exc)
            finally:sampler.close()
        except Exception as exc:save_error('online_setup','selected',exc)
    teachers=[];distribution=collections.Counter();preprocessing=[]
    page_rows=list(read_jsonl(args.pages));wanted={str(r.get('page_id') or r['id'].split('/')[0]) for r in rows}
    with HtmlBrowser() as browser:
        for page in page_rows:
            if str(page.get('page_id') or page['id'].split('/')[0]) not in wanted:continue
            original=page.get('original_html');normalized=page.get('html')
            entry=dict(id=page['id'],original_html=original,normalized_html=normalized)
            try:
                if not original or not Path(original).is_file():entry.update(status='unavailable',reason='Original HTML missing')
                else:
                    a,png0=capture(browser,page,Path(original).read_text())
                    b,png1=capture(browser,page,Path(normalized).read_text())
                    entry.update(status='measured',geometry=difference(a,b),pixel_mae=pixel_difference(png0,png1))
                    folder=out/'preprocessing'/hashlib.sha256(page['id'].encode()).hexdigest()[:16];folder.mkdir(parents=True,exist_ok=True)
                    (folder/'before.png').write_bytes(png0);(folder/'after.png').write_bytes(png1)
                    write_json(folder/'boxes.json',dict(before=a,after=b))
            except Exception as exc:entry.update(status='error',error=str(exc));save_error('preprocessing',page['id'],exc)
            preprocessing.append(entry)
        write_jsonl(out/'preprocessing.jsonl',preprocessing)
        for i,row in enumerate(rows):
            print(f'[data {i+1}/{len(rows)}] {row["id"]}',flush=True)
            try:
                if 'target_declaration_state' not in row:
                    browser.load(html_text(row,True),row['viewport'])
                    row['target_declaration_state']=css_owners.read(browser,row['current'])['state']
                for edit in repair_path(row['declaration_state'],row['target_declaration_state']):
                    distribution[(('online' if row.get('online') else 'fixed-validation'),edit[1],edit[2])]+=1
                entry=dict(id=row['id'],**teacher_check(row))
                entry['browser']=browser_teacher(browser,row,out/'samples'/str(i)/'teacher')
                teachers.append(entry)
            except Exception as exc:save_error('teacher',row['id'],exc)
            write_jsonl(out/'teacher_checks.jsonl',teachers)
    write_json(out/'value_distribution.json',[dict(source=k[0],property=k[1],value=k[2],count=v) for k,v in distribution.most_common()])
    predictions=[]
    if not args.data_only:
        detector,detector_ck=load_detector(args.detector,args.device)
        detected={}
        for row in rows:
            try:
                path=row['target_image'];key=digest(path)
                if key not in detected:detected[key]=detect(detector,detector_input(path),args.threshold)
                row['lazy_predicted_elements']=detected[key]
            except Exception as exc:save_error('detector',row['id'],exc)
        detector_contract=detector_ck.get('config',{}).get('visual_geometry_contract')
        del detector,detector_ck
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
        for stage,path in [('stage1',args.stage1),('stage2',args.stage2)]:
            net,ck=load_policy(path,args.device)
            write_json(out/f'{stage}-config.json',dict(config=ck.get('config'),step=ck.get('step'),checkpoint_sha=digest(path)))
            del ck
            if not net.cfg.semantic or not net.cfg.stylesheets:raise ValueError('Audit requires semantic CSS-owner policy')
            if detector_contract!=net.cfg.visual_geometry_contract:raise ValueError('Detector/policy label contract mismatch')
            with HtmlBrowser() as browser,torch.inference_mode():
                for i,row in enumerate(rows):
                    for condition in (['oracle'] if stage=='stage1' else ['oracle','detector']):
                        print(f'[{stage}/{condition} {i+1}/{len(rows)}] {row["id"]}',flush=True)
                        try:
                            inputs=batch_inputs(net,[row],args.device,condition=='detector')
                            result=decode_audit(net,inputs,row)
                            folder=out/'samples'/str(i)/stage/condition
                            result['execution']=execute_audit(browser,row,result['action'],folder)
                            result.update(id=row['id'],stage=stage,condition=condition,source='online' if row.get('online') else 'fixed-validation')
                            predictions.append(result)
                        except Exception as exc:save_error(stage+'/'+condition,row['id'],exc)
                        write_jsonl(out/'predictions.jsonl',predictions)
            del net
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
    traces=trace_audit(args.trace_root);write_jsonl(out/'trace_checks.jsonl',traces)
    groups=collections.defaultdict(list)
    for r in predictions:groups[(r['source'],r['stage'],r['condition'])].append(r)
    summary=[]
    for (source,stage,condition),rs in groups.items():
        summary.append(dict(source=source,stage=stage,condition=condition,n=len(rs),
            improving=sum(r['symbolic_gain']>0 for r in rs),worsening=sum(r['symbolic_gain']<0 for r in rs),
            no_op=sum(r['same_declaration'] for r in rs),height18=sum(r['action'][1:3]==['height','18px'] for r in rs),
            blocked_correct_edits=sum(not e['allowed'] for r in rs for e in r['correct_edits_legality']),
            decode_mismatches=sum(not r['decoder_agrees'] for r in rs),
            execution_mismatches=sum(not r['execution']['prediction_applied'] for r in rs)))
    write_json(out/'checkpoint_comparison.json',summary)
    report=['# TUIDE cause audit','',
      'Counts describe the selected diagnostic samples, not the historical training distribution. Compare matching IDs in predictions.jsonl; failed checks are not successes.',
      f'Selected fixed pages: {len(json.loads((out/"selected-ids.json").read_text()))}; total samples including online: {len(rows)}; errors: {len(errors)}.',
      f'Preprocessing comparisons: {len(preprocessing)}. Missing original files are unavailable, never passed.',
      f'Teacher checks completed: {len(teachers)}. Browser and symbolic checks are separate.',
      '', '| Source | Stage | Target | N | Improve | Worsen | Same declaration | height=18px | Blocked correct edits | Decode mismatch | Execute mismatch |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:report.append('| '+' | '.join(str(r[k]) for k in ['source','stage','condition','n','improving','worsening','no_op','height18','blocked_correct_edits','decode_mismatches','execution_mismatches'])+' |')
    report+=['','Existing VLM trajectories are analyzed without regeneration; their screenshot/IoU results remain in the original budget diagnosis.',
             'Inspect teacher_checks.jsonl for target-label correctness and full teacher replay; predictions.jsonl for token constraints and actual CSS application.',
             'An empty/failed stage is inconclusive. No automatic retraining or repair is performed.']
    (out/'README.md').write_text('\n'.join(report)+'\n')
    with (out/'error.log').open('w') as f:
        for filename in ['README.md','errors.jsonl','checkpoint_comparison.json','value_distribution.json','preprocessing.jsonl','teacher_checks.jsonl','predictions.jsonl','trace_checks.jsonl']:
            file=out/filename
            if file.exists():f.write(f'\n=== {filename} ===\n'+file.read_text())
    print(f'Deliver {out / "error.log"}',flush=True)

if __name__=='__main__':main()

"""Six-field CE preparation: preserve flow, accept only verified px assignments."""
import copy
from pathlib import Path
from .explicit_html import SIZE_SNAPSHOT_JS,SIZE_BOXES_JS,compare_renders
from .box_scroll_guard import capture,compare
from .preparation_workers import write_json

CONTRACT='flow-preserved-safe-size-margin-v1'
# This implementation used the same final viewport+scroll acceptance contract,
# but performed the expensive scroll capture for every intermediate field
# trial.  Successful records produced by it remain valid; failed/incomplete
# records may be retried with the fast-path implementation below.
COMPATIBLE_IMPLEMENTATION_SHAS={
    '1d22f3fd46d95398b87e177878f376323130f0b15d8af55a4831762cf17c30db',
}


def convert(browser,html,viewport,out,mode='sizes',max_pixel_mae=.01,max_box_error=1.,
            action_ids=None,target_tree=None,observation_size=384):
    if mode!='sizes':raise ValueError('Safe flow normalization only supports sizes')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    browser.load(html,viewport)
    before_png,_=browser.stable_capture(action_ids)
    snapshot=browser.page.evaluate(SIZE_SNAPSHOT_JS,action_ids)
    baseline=browser.page.content()
    before=browser.page.evaluate(SIZE_BOXES_JS)
    ids=set(action_ids or [r['fd_id'] for r in before if r.get('fd_id')])
    original_scroll=capture(browser,out/'scroll-before')
    viewport_trials=0;scroll_trials=0
    def assess_viewport(source):
        nonlocal viewport_trials
        viewport_trials+=1;browser.load(source,viewport)
        png,_=browser.stable_capture(action_ids)
        actual=browser.page.evaluate(SIZE_BOXES_JS)
        qa=compare_renders(before,actual,before_png,png,max_box_error,max_pixel_mae,geometry_target_ids=ids)
        return qa,png
    def assess_full(source):
        nonlocal scroll_trials
        qa,png=assess_viewport(source)
        if not qa['accepted']:
            qa['scroll_pixel_preservation']={'accepted':False,'skipped':'viewport_preservation_failed'}
            return qa,png
        scroll_trials+=1
        scroll=compare(original_scroll,capture(browser,out/'scroll-after',original_scroll),
                       ids,max_box_error,max_pixel_mae)
        qa['scroll_pixel_preservation']=scroll;qa['accepted'] &= scroll['accepted']
        return qa,png
    initial,_=assess_full(baseline)
    if not initial['accepted']:
        report={**initial,'error':'Original HTML is not stable across reloads','failure_stage':'original_reload'}
        write_json(out/'report.json',report);return report

    def materialize(ops):
        browser.load(baseline,viewport)
        browser.page.evaluate('''({ops,contract})=>{
          document.documentElement.setAttribute('data-tuide-inline-actions','targeted-v2');
          document.documentElement.setAttribute('data-tuide-safe-sizes',contract);
          for(const a of ops){
            const e=document.querySelector(`[data-tuide-verify-id="${a.id}"]`);
            for(const [p,v] of Object.entries(a.values))e.style.setProperty(p,v,'important');
            const fields=new Set((e.getAttribute('data-tuide-editable-fields')||'').split(' ').filter(Boolean));
            fields.add(a.field);e.setAttribute('data-tuide-editable-fields',[...fields].join(' '));
            if(a.field==='width'||a.field==='height')e.setAttribute('data-tuide-explicit-id',a.id);
          }
        }''',dict(ops=ops,contract=CONTRACT))
        return browser.page.content()
    ops=[]
    for a in snapshot['candidates']:
        for field in a['editable']:
            if field in ('width','height'):
                value=a.get('preserved_values',{}).get(field,f'{a[field]}px')
                values={p:v for p,v in a['constraints'].items() if p in ('min-'+field,'max-'+field)}
            else:
                value=a['margins'].get(field)
                if value is None:continue
                values={}
            ops.append(dict(id=a['id'],field=field,values={**values,field:value}))
    # Intermediate field selection normally needs only the viewport check.
    # The chosen final document must still pass the exact full scroll guard.
    candidate=materialize(ops);quick,_=assess_viewport(candidate)
    accepted=ops;rejected=[]
    if not quick['accepted']:
        accepted=[]
        for op in ops:
            if viewport_trials>=128:
                rejected.append({**op,'reason':'verification_budget'});continue
            trial=materialize([*accepted,op]);result,_=assess_viewport(trial)
            if result['accepted']:accepted.append(op)
            else:rejected.append({**op,'reason':'changes_render_or_scroll'})
    candidate=materialize(accepted);qa,png=assess_full(candidate)
    full_fallback=False
    if not qa['accepted']:
        # A field that is harmless in the initial viewport may still damage a
        # below-fold or nested-scroll state.  Only this uncommon case pays for
        # the old per-field full-scroll verification.
        full_fallback=True;accepted=[];rejected=[]
        for op in ops:
            if viewport_trials>=128:
                rejected.append({**op,'reason':'verification_budget'});continue
            trial=materialize([*accepted,op]);result,_=assess_full(trial)
            if result['accepted']:accepted.append(op)
            else:rejected.append({**op,'reason':'changes_render_or_scroll'})
        candidate=materialize(accepted);qa,png=assess_full(candidate)
    report={**qa,'normalization_strategy':CONTRACT,'normalized_elements':len({a['id'] for a in accepted}),
            'normalized_ids':sorted({a['id'] for a in accepted}),'partial_normalization':bool(rejected),
            'verification_trials':viewport_trials,'scroll_verification_trials':scroll_trials,
            'full_scroll_fallback':full_fallback,'rejected_fields':rejected,
            'preserved_elements':snapshot['skipped'],
            'mode':'sizes','viewport':viewport,'failure_stage':None if qa['accepted'] else 'normalization_preservation'}
    write_json(out/'report.json',report)
    (out/'before.png').write_bytes(before_png);(out/'after.png').write_bytes(png)
    (out/('normalized.html' if qa['accepted'] else 'rejected.html')).write_text(candidate)
    return report

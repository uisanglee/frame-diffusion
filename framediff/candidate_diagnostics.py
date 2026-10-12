"""Read-only evidence for CSS eligibility; never normalize or mutate a page."""
from collections import Counter

from .tree_edits import FIELDS, value_valid


def inspect_saved_target(seed, folder):
    from pathlib import Path
    from .html_bridge import HtmlBrowser
    from .ir import read_json
    folder=Path(folder);report_path=folder/'report.json'
    report=read_json(report_path) if report_path.is_file() else None
    with HtmlBrowser() as browser:
        browser.load(Path(seed['target_html']).read_text(),seed['viewport'])
        return diagnose(browser,seed,report)


DOM_DIAGNOSTICS_JS = r'''fields=>{
 const declarations=s=>Object.fromEntries([...s].map(p=>[p,s.getPropertyValue(p)]));
 const rules=[],inaccessible=[];
 function walk(rs){for(const r of rs){
   if(r.type===CSSRule.STYLE_RULE)rules.push(r);
   else if(r.cssRules && (r.type!==CSSRule.MEDIA_RULE||matchMedia(r.conditionText).matches) &&
     (r.type!==CSSRule.SUPPORTS_RULE||CSS.supports(r.conditionText)))walk(r.cssRules);
 }}
 for(const sheet of document.styleSheets){
   if(sheet.ownerNode?.hasAttribute?.('data-framediff-static'))continue;
   try{walk(sheet.cssRules)}catch(e){inaccessible.push({href:sheet.href,error:String(e)})}
 }
 const all=[...document.querySelectorAll('*')];
 return {inline_only:document.documentElement.getAttribute('data-tuide-inline-actions')==='targeted-v2',
   untagged_elements:all.filter(e=>!e.hasAttribute('data-fd-id')).length,inaccessible_stylesheets:inaccessible,
   elements:all.filter(e=>e.hasAttribute('data-fd-id')).map(e=>{
     const s=getComputedStyle(e),r=e.getBoundingClientRect();
     const transforms=[];
     for(let p=e;p;p=p.parentElement){const ps=getComputedStyle(p);
       if(ps.transform!=='none'||ps.rotate!=='none'||ps.scale!=='none'||ps.translate!=='none'||
          !['1','normal'].includes(ps.zoom))transforms.push({tag:p.tagName,fd_id:p.getAttribute('data-fd-id')});}
     return {fd_id:e.getAttribute('data-fd-id'),verify_id:e.getAttribute('data-tuide-verify-id'),
       tag:e.tagName,namespace:e.namespaceURI,shadow_root:!!e.shadowRoot,
       editable_attribute:e.getAttribute('data-tuide-editable-fields'),inline:declarations(e.style),
       computed:Object.fromEntries([...fields,'display','visibility','writing-mode','direction'].map(p=>[p,s.getPropertyValue(p)])),
       box:[r.x,r.y,r.width,r.height],client_rects:e.getClientRects().length,transformed_ancestors:transforms,
       matching_rules:rules.flatMap(rule=>{try{return e.matches(rule.selectorText)?
         [{selector:rule.selectorText,declarations:declarations(rule.style)}]:[]}catch{return []}})};
   })};
}'''


def diagnose(browser, seed, normalization_report=None):
    """Reconcile stored eligibility with fresh DOM/owner state and old report.

    Historical skips are evidence. Current exclusions are labeled observations,
    not retroactively asserted to be the reason normalization skipped a node.
    """
    from . import css_owners
    from .explicit_html import current_clip_boxes
    from .tree_online import mutation_sites
    report=normalization_report or {}
    dom=browser.page.evaluate(DOM_DIAGNOSTICS_JS,list(FIELDS))
    ids=[n['id'] for n in seed['current']['nodes'][1:]]
    counts=Counter(e['fd_id'] for e in dom['elements'])
    bad_ids=[dict(fd_id=id,count=counts[id]) for id in ids if counts[id]!=1]
    if bad_ids:
        return dict(contract='css-eligibility-diagnostics-v1',identity_errors=bad_ids,
                    elements=dom['elements'],field_reasons={},historical_skip_reasons={},
                    note='DOM identity mismatch prevents reliable owner/field attribution.')
    parsed=css_owners.read(browser,seed['current'])
    clips=current_clip_boxes(browser,ids)
    skipped={str(r['id']):r['reason'] for r in report.get('preserved_elements',[])}
    by_id={}
    for el in dom['elements']:by_id.setdefault(el['fd_id'],[]).append(el)
    field_counts=Counter();skip_counts=Counter();observed_counts=Counter();records=[]
    for node in seed['current']['nodes'][1:]:
        id=node['id'];matches=by_id.get(id,[])
        if len(matches)!=1:
            reason='missing_dom_id' if not matches else 'duplicate_dom_id'
            records.append(dict(fd_id=id,reason=reason));observed_counts[reason]+=1
            continue
        el=matches[0];allowed=(el['editable_attribute'] or '').split()
        historical=skipped.get(el['verify_id'])
        if historical:skip_counts[historical]+=1
        observations=[]
        if node.get('visual_class',0)<=0:observations.append('not_in_current_abstraction')
        if el['namespace']!='http://www.w3.org/1999/xhtml':observations.append('non_html')
        if el['tag'] in ('HTML','BODY','IFRAME','CANVAS') or el['shadow_root']:observations.append('root_or_embedded')
        clip=clips.get(id)
        if clip is not None and (clip[2]<=0 or clip[3]<=0):observations.append('outside_visible_clip')
        if el['computed']['visibility']!='visible' or el['computed']['display']=='none':observations.append('not_visible')
        if el['computed']['display'] in ('inline','contents') or el['computed']['display'].startswith('table') or el['client_rects']!=1:
            observations.append('inline_table_or_fragmented')
        if el['transformed_ancestors']:observations.append('transformed_ancestry')
        fields=[]
        for i,field in enumerate(FIELDS):
            owner_fields=[(j,parsed['state'][j][i][0]) for j,o in enumerate(parsed['owners'])
                          if id in o.get('matches',[])]
            values=[v for _,v in owner_fields if v]
            if dom['inline_only'] and el['editable_attribute'] is None:reason='missing_editable_attribute'
            elif dom['inline_only'] and field not in allowed:reason='field_not_allowed'
            elif not values:reason='empty_parsed_value'
            elif not any(value_valid(v,field) for v in values):reason='unsupported_value'
            else:reason='eligible'
            field_counts[reason]+=1
            evidence=historical if historical else (
                'historical_exclusion_not_recorded' if reason in ('missing_editable_attribute','field_not_allowed') else None)
            fields.append(dict(property=field,reason=reason,exclusion_evidence=evidence,inline_value=el['inline'].get(field,''),
                               parsed_owner_values=owner_fields,computed_value=el['computed'][field]))
        observed_counts.update(observations)
        records.append({**el,'historical_skip_reason':historical,'current_observations':observations,
                        'visual_class':node.get('visual_class'), 'visible_clip':clip,'fields':fields})
    stored=seed.get('target_declaration_state')
    return dict(contract='css-eligibility-diagnostics-v1',inline_only=dom['inline_only'],
        tree_targets=len(ids),tagged_dom_elements=len(dom['elements']),untagged_dom_elements=dom['untagged_elements'],
        stored_editable_sites=len(mutation_sites(stored)) if stored is not None else None,
        fresh_editable_sites=len(mutation_sites(parsed['state'])),
        stored_state_matches_fresh=stored==parsed['state'] if stored is not None else None,
        stored_owners_match_fresh=seed.get('css_owners')==parsed['owners'],
        normalization=dict(normalized_elements=report.get('normalized_elements'),
            skipped_not_action_target=report.get('skipped_not_action_target'),
            strategy=report.get('normalization_strategy'),report_available=normalization_report is not None),
        field_reasons=dict(field_counts),historical_skip_reasons=dict(skip_counts),
        current_observations=dict(observed_counts),elements=records,
        inaccessible_stylesheets=dom['inaccessible_stylesheets'],
        note='Computed values are observations, not automatically editable declarations. '
             'Current observations do not prove historical exclusion causes; consult historical_skip_reason.')

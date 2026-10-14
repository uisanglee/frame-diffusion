"""Normalization is reversible declaration corruption, not a new target.

Keep original declarations and a change journal. No visual teacher search.
Unsupported declarations/constraints are preserved, never silently approximated.
"""
import copy
import hashlib
from pathlib import Path

from . import css_owners
from .ir import read_json
from .preparation_workers import write_json
from .tree_edits import FIELDS, value_valid, repair_path
from .web_experiment import digest

CONTRACT = 'original-declaration-normalization-corruption-v2'


def record_corruption(browser, tree, original_html):
    """Keep only reversible normalization edits, with exact original labels.

Rebuild inline styles from the original, then apply supported normalized fields.
This preserves min/max, padding, logical properties and all other fixed context.
No original declarations are embedded in the model's HTML input.
"""
    ids=[n['id'] for n in tree['nodes'][1:]]
    records=browser.page.evaluate('''({html,ids,fields})=>{
      const original=new DOMParser().parseFromString(html,'text/html');
      return ids.map(id=>{
        const find=doc=>[...doc.querySelectorAll('[data-fd-id]')].find(e=>e.dataset.fdId===id);
        const a=find(original),b=find(document);
        if(!a||!b)throw Error('Missing normalization journal node: '+id);
        const pairs=e=>Object.fromEntries([...e.style].map(p=>[p,[e.style.getPropertyValue(p),e.style.getPropertyPriority(p)]]));
        return {id,style:a.getAttribute('style'),before:fields.map(p=>[a.style.getPropertyValue(p),a.style.getPropertyPriority(p)]),
          after:fields.map(p=>[b.style.getPropertyValue(p),b.style.getPropertyPriority(p)]),
          original_properties:pairs(a),normalized_properties:pairs(b),
          editable:(b.getAttribute('data-tuide-editable-fields')||'').split(' ').filter(Boolean)};
      });
    }''',dict(html=original_html,ids=ids,fields=FIELDS))
    goal=[[['',''] for _ in FIELDS]];journal=[];preserved=[]
    for record in records:
        allowed=[];state=[]
        for j,field in enumerate(FIELDS):
            before,after=record['before'][j],record['after'][j]
            supported=(not before[0] or value_valid(before[0],field)) and (not after[0] or value_valid(after[0],field))
            if field in record['editable'] and supported:
                allowed.append(field);state.append(before)
                if before!=after:journal.append(dict(id=record['id'],field=field,before=before,after=after))
            else:
                state.append(['',''])
                if before!=after:preserved.append(dict(id=record['id'],field=field,reason='unsupported_or_readonly'))
        for field in set(record['original_properties'])|set(record['normalized_properties']):
            if field not in FIELDS and record['original_properties'].get(field)!=record['normalized_properties'].get(field):
                preserved.append(dict(id=record['id'],field=field,reason='outside_action_schema'))
        record['allowed']=allowed;goal.append(state)
    browser.page.evaluate('''({records,fields})=>{
      document.documentElement.setAttribute('data-tuide-inline-actions','targeted-v2');
      for(const r of records){
        const e=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.dataset.fdId===r.id);
        if(r.style===null)e.removeAttribute('style');else e.setAttribute('style',r.style);
        for(const p of r.allowed){const [v,priority]=r.after[fields.indexOf(p)];
          if(v)e.style.setProperty(p,v,priority);else e.style.removeProperty(p);}
        e.setAttribute('data-tuide-editable-fields',r.allowed.join(' '));
        e.removeAttribute('data-tuide-explicit-id');
      }
    }''',dict(records=records,fields=FIELDS))
    parsed=css_owners.read(browser,tree)
    return parsed,goal,dict(strategy='original-declaration-journal-v2',changes=journal,
        preserved_unsupported_changes=preserved,
        reverse_edits=repair_path(parsed['state'],goal),
        note='Direct original values, absence and priorities; no distance-based label selection.')


def prepare_page(task):
    from .html_bridge import HtmlBrowser
    from .explicit_html import convert,feedback_metadata
    from .html_feedback import refresh_geometry
    from .visual import annotate
    from .tree_online import target_pool,OnlineSampler
    args,out,split,index,page_id,row,pages=task
    folder=out/'normalized-pages'/split/hashlib.sha256(page_id.encode()).hexdigest()
    folder.mkdir(parents=True,exist_ok=True)
    source=Path(row['current_html'] if row['id'].endswith('/clean') else row['target_html'])
    signature={'html':digest(source),'target':digest(row['target_image']),
               'tree':row['current'],'boxes':row['current_boxes'],'elements':row['target_elements'],
               'contract':CONTRACT}
    cache=folder/'record.json';sampler=None;stage='recovery_source'
    try:
        if args.resume and cache.exists():
            saved=read_json(cache)
            if saved['signature']!=signature:raise ValueError('Recovery inputs changed; use a new output directory')
            for path,sha in saved['assets'].items():
                if digest(path)!=sha:raise ValueError('Recovery cache asset changed')
            return saved['row'],saved['page'],None
        normalized=folder/'normalized.html';reused=False
        reuse=getattr(args,'reuse_normalized',None)
        if reuse:
            previous=Path(reuse)/'normalized-pages'/split/folder.name
            stamp=previous/'normalization.json'
            if stamp.exists():
                old=read_json(stamp);path=previous/'normalized.html'
                if old['source_sha']==signature['html'] and old['sha']==digest(path):
                    normalized.write_bytes(path.read_bytes());reused=True
        with HtmlBrowser() as browser:
            stage='normalization'
            if not reused:
                qa=convert(browser,source.read_text(),row['viewport'],folder,mode='sizes',
                           max_pixel_mae=args.max_pixel_mae,max_box_error=args.max_box_error,
                           action_ids=[n['id'] for n in row['current']['nodes'][1:] if n.get('visual_class',1)>0],
                           target_tree=row['current'],observation_size=args.observation_size)
                if not qa['accepted']:
                    if qa.get('failure_stage')!='normalization_preservation':
                        raise ValueError(qa.get('error','Unstable original render'))
                    normalized.write_bytes((folder/'rejected.html').read_bytes())
                write_json(folder/'normalization-diagnostics.json',qa)
            browser.load(normalized.read_text(),row['viewport'])
            tree=copy.deepcopy(row['current']);annotate(browser,tree)
            stage='original_declaration_journal'
            parsed,goal,journal=record_corruption(browser,tree,source.read_text())
            if len(parsed['owners'])>args.max_css_owners:raise ValueError('Too many CSS owners')
            normalized.write_text(browser.page.content())
            write_json(folder/'normalization-journal.json',journal)
            boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
            boxes[tree['nodes'][0]['id']]=[0,0,*row['viewport']]
            feedback=feedback_metadata(browser,tree)
        seed={k:row[k] for k in ('group','split','source_sha','viewport','target_image','target_elements')}
        seed.update({k:row[k] for k in ('page_id','view') if k in row})
        seed.update(id=page_id+'/seed',current=refresh_geometry(tree,boxes),current_boxes=boxes,
                    target_html=str(normalized),reference_html=str(source),recovery_contract=CONTRACT,
                    css_owners=parsed['owners'],target_declaration_state=goal,
                    normalization_contract=CONTRACT,recovery_report=journal,
                    normalization_state=parsed['state'],**feedback)
        stage='direct_recovery_labels'
        pool=target_pool([{**seed,'split':'train'}],cache_dir=out/'.online-target-cache',observation_size=args.observation_size)
        stage='recovery_sample'
        sampler=OnlineSampler(pool,'screenshot',seed=args.seed,max_noise=1,allow_clean=True)
        new=sampler.generate(index)
        initial=folder/'initial.html';png=folder/'initial.png'
        initial.write_text(new.pop('current_html_text'));png.write_bytes(new['current_image'])
        new.update(split=split,current_html=str(initial),current_image=str(png),
                   normalization_stats={'reused_normalization':reused,
                       'normalized_elements':len({c['id'] for c in journal['changes']})})
        page={**pages[page_id],'html':str(normalized),'target_html':str(normalized),
              'reference_html':str(source),'screenshot':row['target_image'],'initial_html':str(initial),
              'evaluation_target_boxes':row['current_boxes'],'recovery_contract':CONTRACT}
        assets={str(p):digest(p) for p in (source,Path(row['target_image']),normalized,initial,png)}
        write_json(cache,dict(signature=signature,row=new,page=page,assets=assets))
        return new,page,None
    except Exception as exc:
        failure={'id':page_id,'stage':stage,'error':str(exc)}
        write_json(folder/'failure.json',failure)
        return None,None,failure
    finally:
        if sampler is not None:sampler.close()

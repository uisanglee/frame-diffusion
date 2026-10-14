"""Fixed-viewport, parent-preserving box programs. No CSS restoration teacher."""
import copy
import hashlib
from pathlib import Path

from .ir import read_json
from .preparation_workers import write_json

CONTRACT='parent-relative-xywh-v1'
FIELDS=('x','y','width','height')
CSS_FIELDS=('left','top','width','height')


def read(browser,tree,html=None):
    rows=browser.page.evaluate('''({html,ids})=>{
      const doc=html===null?document:new DOMParser().parseFromString(html,'text/html');
      return ids.map(id=>{
        const e=[...doc.querySelectorAll('[data-fd-id]')].find(e=>e.dataset.fdId===id);
        if(!e)throw Error('Missing box node '+id);
        return ['left','top','width','height'].map(p=>e.hasAttribute('data-tuide-box-target')?
          [e.style.getPropertyValue(p),e.style.getPropertyPriority(p)]:['','']);
      });
    }''',dict(html=html,ids=[n['id'] for n in tree['nodes'][1:]]))
    return dict(owners=[{'kind':'root','matches':[]}]+[
        dict(kind='inline',id=n['id'],matches=[n['id']],box_layout=True) for n in tree['nodes'][1:]],
        state=[[['',''] for _ in FIELDS]]+rows)


def execute(browser,owners,edit):
    from .tree_edits import validate_edit
    i,field,value,priority=validate_edit(edit,len(owners),fields=FIELDS)
    if not value.endswith('px'):raise ValueError('Box actions require explicit px SET values')
    from .explicit_html import check_explicit_edit
    check_explicit_edit(browser,owners[i]['matches'],field,value)
    browser.page.evaluate('''({id,field,value,priority})=>{
      const e=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.dataset.fdId===id);
      if(!e?.hasAttribute('data-tuide-box-target'))throw Error('Not a box action target');
      e.style.setProperty(field,value,priority);
    }''',dict(id=owners[i]['id'],field=CSS_FIELDS[FIELDS.index(field)],value=value,priority=priority))
    browser.executions+=1


def convert(browser,html,viewport,out,tree,max_pixel_mae=.01,max_box_error=1.):
    """Measure every supporting box BEFORE writing styles; never reparent.

    Ancillary boxes are frozen too so removing targets from flow cannot move
    unselected siblings. Only tagged visual targets are exposed to the policy.
    Unsupported geometry is a conversion failure, never training corruption.
    """
    from .explicit_html import SIZE_BOXES_JS,compare_renders
    from .scroll_guard import snapshot,compare
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    browser.load(html,viewport)
    browser.page.evaluate('window.scrollTo({left:0,top:0,behavior:"instant"})')
    ids=[n['id'] for n in tree['nodes'][1:] if n.get('visual_class',0)>0]
    before_png,_=browser.stable_capture(ids)
    # Stable verification IDs are not layout attributes and contain no targets.
    browser.page.evaluate("[...document.querySelectorAll('*')].forEach((e,i)=>e.setAttribute('data-tuide-verify-id',String(i)))")
    before=browser.page.evaluate(SIZE_BOXES_JS)
    from .box_scroll_guard import capture as capture_scroll,compare as compare_scroll_pixels
    try:scroll_pixels_before=capture_scroll(browser,out/'scroll-before')
    except Exception as exc:
        report=dict(accepted=False,contract=CONTRACT,error='Scroll verification incomplete: '+str(exc))
        write_json(out/'report.json',report);return report
    scroll_before=snapshot(browser)
    records=browser.page.evaluate('''()=>[...document.querySelectorAll('*')].map((e,i)=>{
      const r=e.getBoundingClientRect(),s=getComputedStyle(e);
      return {i,parent:e.parentElement?.getAttribute('data-tuide-verify-id'),tag:e.tagName,
        id:e.getAttribute('data-fd-id'),box:[r.x,r.y,r.width,r.height],
        border:[e.clientLeft,e.clientTop],scroll:[e.scrollLeft,e.scrollTop],
        display:s.display,position:s.position,rects:e.getClientRects().length,namespace:e.namespaceURI,
        island:!!e.closest('table')||s.display.startsWith('inline')||!![...function*(p){while(p){yield p;p=p.parentElement;}}(e.parentElement)].find(p=>getComputedStyle(p).display.startsWith('inline')),
        transformed:s.transform!=='none'||s.rotate!=='none'||s.scale!=='none'||s.translate!=='none'||!['1','normal'].includes(s.zoom)};
    })''')
    active=[r for r in records if r['box'][2]>0 and r['box'][3]>0 and r['display']!='none']
    errors=[dict(id=r['id'],tag=r['tag'],reason='unsupported_transform_or_sticky')
            for r in active if r['transformed'] or r['position']=='sticky']
    if errors:
        report=dict(accepted=False,contract=CONTRACT,errors=errors)
        write_json(out/'report.json',report);return report
    (out/'before.png').write_bytes(before_png)
    browser.page.evaluate('''({records,ids})=>{
      const all=[...document.querySelectorAll('*')],byIndex=new Map(records.map(r=>[String(r.i),r]));
      const usable=r=>r&&!r.island&&r.namespace==='http://www.w3.org/1999/xhtml'&&r.rects>0&&r.display!=='none'&&r.display!=='contents';
      // No geometry reads below: all coordinates belong to the original frame.
      for(const r of records){
        if(!usable(r))continue;
        const e=all[r.i],set=(p,v)=>e.style.setProperty(p,String(v),'important');
        set('box-sizing','border-box');set('width',r.box[2]+'px');set('height',r.box[3]+'px');
        for(const axis of ['width','height']){set('min-'+axis,'0px');set('max-'+axis,'none');}
        if(r.tag==='HTML'){set('position','relative');continue;}
        let parent=byIndex.get(r.parent);
        while(parent&&!usable(parent))parent=byIndex.get(parent.parent);
        const origin=r.position==='fixed'?[0,0]:parent?[parent.box[0]+parent.border[0]-parent.scroll[0],parent.box[1]+parent.border[1]-parent.scroll[1]]:[0,0];
        set('position',r.position==='fixed'?'fixed':'absolute');set('inset','auto');set('margin','0px');
        set('left',(r.box[0]-origin[0])+'px');set('top',(r.box[1]-origin[1])+'px');
        set('flex','none');set('align-self','auto');set('justify-self','auto');
        if(r.display==='inline')set('display','block');
        if(ids.includes(r.id)){
          e.setAttribute('data-tuide-box-target','1');
          e.setAttribute('data-tuide-explicit-id',String(r.i));
        }
      }
      document.documentElement.setAttribute('data-tuide-box-layout','parent-relative-xywh-v1');
    }''',dict(records=records,ids=ids))
    candidate=browser.page.content()
    # Reopen serialized HTML: the artifact, not just the live DOM, must preserve it.
    browser.load(candidate,viewport)
    after_png,_=browser.stable_capture(ids)
    after=browser.page.evaluate(SIZE_BOXES_JS)
    qa=compare_renders(before,after,before_png,after_png,max_box_error,max_pixel_mae,geometry_target_ids=set(ids))
    scroll=compare(scroll_before,snapshot(browser));qa['scroll_preservation']=scroll
    scroll_pixels=compare_scroll_pixels(scroll_pixels_before,capture_scroll(browser,out/'scroll-after',scroll_pixels_before),
                                       ids,max_box_error,max_pixel_mae)
    qa['scroll_pixel_preservation']=scroll_pixels
    parents=browser.page.evaluate("[...document.querySelectorAll('*')].map(e=>e.parentElement?.getAttribute('data-tuide-verify-id')||null)")
    same_parents=parents==[r['parent'] for r in records]
    frozen_ids=browser.page.evaluate("[...document.querySelectorAll('[data-tuide-box-target]')].map(e=>e.dataset.fdId)")
    excluded={r['id'] for r in records if r['island'] and r['id'] in ids}
    missing=sorted(set(ids)-set(frozen_ids)-excluded)
    qa.update(contract=CONTRACT,parent_relationships_preserved=same_parents,
              normalized_elements=len(frozen_ids),unconverted_targets=missing,
              preserved_noneditable_targets=sorted(excluded))
    qa['accepted']=qa['accepted'] and scroll['accepted'] and scroll_pixels['accepted'] and same_parents and not missing
    write_json(out/'report.json',qa);write_json(out/'measured-boxes.json',records)
    (out/'after.png').write_bytes(after_png)
    (out/('normalized.html' if qa['accepted'] else 'rejected.html')).write_text(candidate)
    return qa


def prepare_page(task):
    from .html_bridge import HtmlBrowser
    from .explicit_html import feedback_metadata
    from .html_feedback import refresh_geometry
    from .tree_online import target_pool,OnlineSampler
    from .web_experiment import digest
    args,out,split,index,page_id,row,pages=task
    folder=out/'normalized-pages'/split/hashlib.sha256(page_id.encode()).hexdigest();folder.mkdir(parents=True,exist_ok=True)
    source=Path(row['current_html'] if row['id'].endswith('/clean') else row['target_html'])
    signature=dict(html=digest(source),image=digest(row['target_image']),contract=CONTRACT)
    cache=folder/'record.json';sampler=None
    try:
        if args.resume and cache.exists():
            saved=read_json(cache)
            if saved['signature']!=signature:raise ValueError('Box source changed; use a new output')
            for path,sha in saved['assets'].items():
                if digest(path)!=sha:raise ValueError('Box cache asset changed')
            return saved['row'],saved['page'],None
        tree=copy.deepcopy(row['current'])
        with HtmlBrowser() as browser:
            qa=convert(browser,source.read_text(),row['viewport'],folder,tree,args.max_pixel_mae,args.max_box_error)
            if not qa['accepted']:raise ValueError('Box conversion did not preserve original rendering; see report.json')
            parsed=read(browser,tree)
            boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
            boxes[tree['nodes'][0]['id']]=[0,0,*row['viewport']]
            feedback=feedback_metadata(browser,tree)
        if len(parsed['owners'])>args.max_css_owners:raise ValueError('Too many box owners')
        seed={k:row[k] for k in ('group','split','source_sha','viewport','target_image','target_elements')}
        seed.update({k:row[k] for k in ('page_id','view') if k in row})
        seed.update(id=page_id+'/seed',current=refresh_geometry(tree,boxes),current_boxes=boxes,
            target_html=str(folder/'normalized.html'),reference_html=str(source),
            css_owners=parsed['owners'],target_declaration_state=parsed['state'],
            box_layout_contract=CONTRACT,normalization_contract=CONTRACT,**feedback)
        pool=target_pool([{**seed,'split':'train'}],observation_size=args.observation_size)
        sampler=OnlineSampler(pool,'screenshot',seed=args.seed,max_noise=1)
        new=sampler.generate(index)
        initial=folder/'initial.html';png=folder/'initial.png'
        initial.write_text(new.pop('current_html_text'));png.write_bytes(new['current_image'])
        new.update(split=split,current_html=str(initial),current_image=str(png),normalization_stats=qa)
        page={**pages[page_id],'html':seed['target_html'],'target_html':seed['target_html'],
            'reference_html':str(source),'screenshot':row['target_image'],'initial_html':str(initial),
            'evaluation_target_boxes':row['current_boxes'],'box_layout_contract':CONTRACT}
        assets={str(p):digest(p) for p in (source,Path(row['target_image']),folder/'normalized.html',initial,png)}
        write_json(cache,dict(signature=signature,row=new,page=page,assets=assets))
        return new,page,None
    except Exception as exc:
        failure=dict(id=page_id,stage='box_conversion',error=str(exc));write_json(folder/'failure.json',failure)
        return None,None,failure
    finally:
        if sampler:sampler.close()


def generate(browser,target,rng,index,max_noise,mode,parent_first=False):
    from .tree_edits import repair_path
    from .tree_online import abstraction
    from .html_feedback import refresh_geometry
    from .explicit_html import feedback_metadata
    tree=target['current'];viewport=target['viewport'];owners=target['css_owners']
    goal=target['target_declaration_state'];state=read(browser,tree)['state'];noise=[]
    sites=[(i,j) for i in range(1,len(state)) for j in range(4) if state[i][j][0]]
    if not sites:raise ValueError('No box targets')
    for _ in range(rng.randint(1,max_noise)):
        i,j=rng.choice(sites);old=float(state[i][j][0][:-2]);field=FIELDS[j]
        value=old+rng.choice([-32,-16,-8,8,16,32])
        if j>=2:
            from .explicit_html import explicit_size_limits
            floor=explicit_size_limits(browser,owners[i]['matches'])[owners[i]['id']][field]
            value=max(floor+1,value)
        if value==old:value=old+8
        edit=[i,field,f'{value:g}px','important'];execute(browser,owners,edit)
        noise.append(edit);state=read(browser,tree)['state']
    path=repair_path(state,goal,rng.getrandbits(64),fields=FIELDS)
    if not path:raise ValueError('Box corruption returned to clean state')
    depth=rng.randrange(len(path))
    for edit in path[:depth]:execute(browser,owners,edit)
    state=read(browser,tree)['state'];remaining=path[depth:];teacher=remaining[0]
    if parent_first:
        from .parent_first import owner_order
        order={i:d for d,i in enumerate(owner_order(tree,owners))}
        teacher=min(remaining,key=lambda e:order[e[0]])
    _,boxes,_=abstraction(browser,tree,viewport,target.get('corruption_observation_size',384))
    result={**target,'id':target['id']+f'/online-{index}',
        'current':refresh_geometry(tree,boxes),'current_boxes':boxes,
        'declaration_state':state,'replacement_edit':teacher,'teacher_strategy':CONTRACT,
        'symbolic_distance':len(remaining),'corruption_contract':CONTRACT,
        'current_html_text':browser.page.content(),**feedback_metadata(browser,tree)}
    if mode=='screenshot':result['current_image']=browser.page.screenshot(animations='disabled')
    result['online']=dict(index=index,retries=0,seconds=0.,errors=[],corruptions=noise,
        path_length=len(path),reverse_prefix=depth,teacher_owner='inline',probe_rejections=0)
    return result

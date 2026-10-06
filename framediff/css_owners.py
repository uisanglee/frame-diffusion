"""Address inline declarations and embedded stylesheet rules by stable paths.

The browser owns CSS parsing and cascade. Rules are not identified by selector
text alone (duplicate selectors and nested @media rules are common).
"""
import copy
import json

from .tree_edits import FIELDS, validate_edit

CONTRACT = 'css-owner-tree-v2'

# Only embedded stylesheets are editable. External sheets keep participating in
# browser layout. Group rules preserve their index paths when values change.
READ_JS = r'''({html, ids, fields, fixed=false}) => {
 const doc = html === null ? document : new DOMParser().parseFromString(html,'text/html');
 const tagged = [...doc.querySelectorAll('[data-fd-id]')];
 if(new Set(tagged.map(e=>e.dataset.fdId)).size!==tagged.length)throw Error('Duplicate DOM IDs');
 const aliases=['all','inline-size','block-size','margin-inline','margin-block',
   'margin-inline-start','margin-inline-end','margin-block-start','margin-block-end'];
 function read(style) {
   if(aliases.some(p=>style.getPropertyValue(p)))throw Error('Unsupported overlapping CSS declarations');
   return fields.map(p=>[style.getPropertyValue(p),style.getPropertyPriority(p)]);
 }
 const owners=[{kind:'root',matches:[]}],state=[fields.map(()=>['',''])];
 for(const id of ids) {
   const el=tagged.find(e=>e.dataset.fdId===id);if(!el)throw Error('Missing DOM ID: '+id);
   owners.push({kind:'inline',id,matches:[id]});state.push(read(el.style));
   if(fixed)for(const p of fields)el.style.removeProperty(p);
 }
 const styles=[...doc.querySelectorAll('style:not([data-framediff-static])')];
 styles.forEach((el,block)=>{
   let sheet=el.sheet;
   if(html!==null){sheet=new CSSStyleSheet();sheet.replaceSync(el.textContent);}
   if(!sheet)throw Error('Embedded stylesheet unavailable');
   // Constructable sheets drop @import, which would shift live rule paths.
   if(/@import\b/i.test(el.textContent))throw Error('Embedded @import is not supported; bundle CSS first');
   function walk(rules,path,conditions) {
     [...rules].forEach((rule,index)=>{
       const next=[...path,index];
       if(rule.type===CSSRule.STYLE_RULE) {
         if(rule.cssRules?.length)throw Error('Nested CSS style rules are not supported');
         // Pseudo-elements cannot be addressed as DOM boxes. Leave untouched.
         let matched=[];try{matched=[...doc.querySelectorAll(rule.selectorText)]
           .map(e=>e.getAttribute('data-fd-id')).filter(id=>ids.includes(id));}catch{}
         if(!matched.length)return;
         owners.push({kind:'rule',block,path:next,selector:rule.selectorText,conditions,matches:matched});
         state.push(read(rule.style));
         if(fixed)for(const p of fields)rule.style.removeProperty(p);
       } else if(rule.cssRules && (rule.type===CSSRule.MEDIA_RULE || rule.type===CSSRule.SUPPORTS_RULE ||
                   rule.constructor.name==='CSSLayerBlockRule')) {
         walk(rule.cssRules,next,[...conditions,rule.cssText.split('{')[0].trim()]);
       }
     });
   }
   walk(sheet.cssRules,[],[]);
   if(fixed)el.textContent=[...sheet.cssRules].map(r=>r.cssText).join('\n');
 });
 let signature=null;
 if(fixed){
   doc.querySelectorAll('style[data-framediff-static]').forEach(e=>e.remove());
   function serial(node){
     if(node.nodeType!==1)return [node.nodeType,node.textContent];
     const attrs=[...node.attributes].filter(a=>a.name!=='style').map(a=>[a.name,a.value]).sort();
     const style=[...node.style].map(p=>[p,node.style.getPropertyValue(p),node.style.getPropertyPriority(p)]).sort();
     return [node.tagName,attrs,style,[...node.childNodes].map(serial)];
   }
   signature=JSON.stringify(serial(doc.documentElement));
 }
 return {owners,state,fixed:signature};
}'''


def read(browser, tree, html=None, fixed=False):
    return browser.page.evaluate(READ_JS, dict(html=html,
        ids=[n['id'] for n in tree['nodes'][1:]], fields=FIELDS, fixed=fixed))


def execute(browser, owners, edit):
    index, field, value, priority = validate_edit(edit, len(owners))
    browser.page.evaluate(r'''({owner,field,value,priority})=>{
      let style,sheet,el;
      if(owner.kind==='inline') {
        el=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.dataset.fdId===owner.id);
        if(!el)throw Error('Missing inline owner');style=el.style;
      } else if(owner.kind==='rule') {
        el=document.querySelectorAll('style:not([data-framediff-static])')[owner.block];
        if(!el?.sheet)throw Error('Missing stylesheet owner');sheet=el.sheet;
        let rule, rules=sheet.cssRules;
        for(const i of owner.path){rule=rules[i];if(!rule)throw Error('Missing rule path');rules=rule.cssRules;}
        if(rule.selectorText!==owner.selector)throw Error('Rule identity changed');style=rule.style;
      } else throw Error('Invalid owner kind');
      if(value==='')style.removeProperty(field);
      else {if(!CSS.supports(field,value))throw Error('Browser rejected CSS value');style.setProperty(field,value,priority);}
      // CSSOM mutations alone are absent from page.content(). Persist all group
      // rules to the owning style element before exporting or reloading HTML.
      if(sheet)el.textContent=[...sheet.cssRules].map(r=>r.cssText).join('\n');
    }''', dict(owner=owners[index], field=field, value=value, priority=priority))
    browser.executions += 1


def context_tree(tree, boxes, owners):
    """Owner geometry is the union of currently matched DOM boxes, never target boxes."""
    result=copy.deepcopy(tree);geometry=copy.deepcopy(boxes)
    if [o.get('id') for o in owners[1:len(tree['nodes'])]] != [n['id'] for n in tree['nodes'][1:]]:
        raise ValueError('Inline owner ordering differs from DOM')
    for index,owner in enumerate(owners[len(tree['nodes']):],len(tree['nodes'])):
        node=copy.deepcopy(tree['nodes'][0]);node.update(id=f'__css_rule_{index}',parent=tree['nodes'][0]['id'],
            name=json.dumps({k:v for k,v in owner.items() if k!='matches'},sort_keys=True),visual_class=0)
        result['nodes'].append(node)
        members=[boxes[i] for i in owner['matches'] if i in boxes]
        if members:
            x=min(b[0] for b in members);y=min(b[1] for b in members)
            geometry[node['id']]=[x,y,max(b[0]+b[2] for b in members)-x,max(b[1]+b[3] for b in members)-y]
        else: geometry[node['id']]=[0,0,0,0]
    return result,geometry


def computed(browser, owners, edit):
    """Observe whether the sampled declaration affects its matched elements.

    This execution probe delegates specificity/!important/layers/media to the
    browser. It is not a hand-written or complete CSS winning-rule resolver.
    """
    owner,field,_,_=edit
    return browser.page.evaluate('''({ids,field})=>ids.map(id=>{
      const e=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.dataset.fdId===id);
      return e ? getComputedStyle(e).getPropertyValue(field) : null;
    })''',dict(ids=owners[owner]['matches'],field=field))

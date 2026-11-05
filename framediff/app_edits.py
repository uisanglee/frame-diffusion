"""Transfer verified geometric edits to an intact JavaScript application's DOM.

The original application source is retained byte-for-byte. A separate runtime
overlay reapplies CSS declarations to guarded rendered-node bindings. This does
not rewrite JSX or claim semantic identity for arbitrary application states.
"""
import json
import re

from bs4 import BeautifulSoup, Comment, NavigableString


BINDING = 'data-tuide-app-node'
POLICY_FIELDS = {'width', 'height', 'margin-top', 'margin-right', 'margin-bottom', 'margin-left'}
NORMALIZATION_FIELDS = {'min-width', 'max-width', 'min-height', 'max-height'}
FIELDS = POLICY_FIELDS | NORMALIZATION_FIELDS
OVERLAY_ID = 'tuide-application-layout-edits'


def _declarations(value):
    """Split a browser-serialized declaration list without breaking URL strings."""
    parts=[];token=[];quote=None;depth=0;i=0
    while i<len(value):
        char=value[i]
        if char=='\\':
            token.append(value[i:i+2]);i+=2;continue
        if quote:
            token.append(char)
            if char==quote:quote=None
        elif char in ('"', "'"):
            quote=char;token.append(char)
        elif value[i:i+2]=='/*':
            end=value.find('*/',i+2)
            if end<0:raise ValueError('Unclosed inline CSS comment')
            token.append(' ');i=end+2;continue
        elif char=='(':depth+=1;token.append(char)
        elif char==')':depth-=1;token.append(char)
        elif char==';' and depth==0:parts.append(''.join(token));token=[]
        else:token.append(char)
        if depth<0:raise ValueError('Unbalanced inline CSS')
        i+=1
    if quote or depth:raise ValueError('Unclosed inline CSS value')
    parts.append(''.join(token))
    return parts


def _styles(value):
    declarations = {}
    for item in _declarations(value or ''):
        if not item.strip():continue
        key,separator,raw=item.partition(':')
        if not separator or not re.fullmatch(r'(?:--)?[-a-zA-Z0-9_]+',key.strip()):
            raise ValueError('Unsupported inline CSS in application transfer')
        key=key.strip() if key.strip().startswith('--') else key.strip().lower()
        important=re.search(r'!\s*important\s*$',raw,re.I)
        content=raw[:important.start()].strip() if important else raw.strip()
        properties={key:content}
        if key=='margin':
            # CSSOM collapses four normalized margin longhands into shorthand.
            # Expand it before comparing so serialization is not mistaken for a
            # non-layout edit. Whitespace inside var()/calc() stays in one value.
            values=[];start=0;depth=0;quote=None;escaped=False
            for index,char in enumerate(content):
                if escaped:escaped=False;continue
                if char=='\\':escaped=True;continue
                if quote:
                    if char==quote:quote=None
                elif char in ('"',"'"):quote=char
                elif char=='(':depth+=1
                elif char==')':depth-=1
                elif char.isspace() and depth==0:
                    if content[start:index].strip():values.append(content[start:index].strip())
                    start=index+1
            if content[start:].strip():values.append(content[start:].strip())
            if not 1<=len(values)<=4:raise ValueError('Unsupported inline margin shorthand')
            top=values[0];right=values[1] if len(values)>1 else top
            bottom=values[2] if len(values)>2 else top
            left=values[3] if len(values)>3 else right
            properties=dict(zip(('margin-top','margin-right','margin-bottom','margin-left'),
                                (top,right,bottom,left)))
        for name,content in properties.items():
            current = declarations.get(name)
            # Match CSS inline cascade when a declaration is repeated.
            if current and current[1] == 'important' and not important:
                continue
            declarations[name] = (content, 'important' if important else '')
    return declarations


def _fingerprint(node):
    return ' '.join(' '.join(str(n) for n in node.children
                            if isinstance(n, str) and not isinstance(n, Comment)).split())


def _document(html):
    soup = BeautifulSoup(html, 'html.parser')
    for item in soup.select('style[data-framediff-static]'):
        item.decompose()
    return soup


def _fixed_signature(soup):
    """Compare source structure/content after removing permitted edit metadata."""
    # AppBrowser's doctype + clone serialization has a newline which Chromium's
    # page.content() drops. Ignore only whitespace outside the document element;
    # body whitespace can affect layout and must remain part of the signature.
    for node in list(soup.children):
        if isinstance(node,NavigableString) and not str(node).strip():node.extract()
    for node in soup.find_all(True):
        styles = _styles(node.get('style', ''))
        styles = {k: v for k, v in styles.items() if k not in FIELDS}
        node.attrs.pop('style', None)
        if styles:
            node['style'] = json.dumps(styles, sort_keys=True)
        for key in list(node.attrs):
            if key.startswith(('data-tuide-', 'data-fd-')):
                del node[key]
        node.attrs = dict(sorted(node.attrs.items()))
    return str(soup)


def _bindings(soup):
    result = {}
    for node in soup.select(f'[{BINDING}]'):
        selector = node[BINDING]
        if selector in result:
            raise ValueError('Duplicate application DOM binding: ' + selector)
        # Only allow structural selectors generated by AppBrowser, never arbitrary
        # model-generated selectors or executable code in the overlay.
        if not re.fullmatch(r'body(?: > [a-zA-Z][a-zA-Z0-9-]*:nth-of-type\([1-9][0-9]*\))*', selector):
            raise ValueError('Invalid application DOM binding: ' + selector)
        matches=soup.select(selector)
        if len(matches)!=1 or matches[0] is not node:
            raise ValueError('Application DOM binding does not identify its element: '+selector)
        result[selector] = node
    return result


def apply_application_edits(source_html, before_html, after_html):
    """Return original application + guarded CSS overlay, and edit metadata.

    Both snapshots must retain AppBrowser bindings. Only inline size/margin
    declarations and normalization min/max size declarations may change.
    Stylesheet changes, structural/content changes and missing bindings fail.
    """
    before = _document(before_html)
    after = _document(after_html)
    old, new = _bindings(before), _bindings(after)
    if not old or set(old) != set(new):
        raise ValueError('Application DOM bindings changed or are absent')
    if _fixed_signature(_document(before_html)) != _fixed_signature(_document(after_html)):
        raise ValueError('Unsupported application DOM/content/stylesheet change')
    edits = []
    for selector, node in old.items():
        initial, final = _styles(node.get('style', '')), _styles(new[selector].get('style', ''))
        changed = {k for k in initial.keys() | final.keys() if initial.get(k) != final.get(k)}
        if changed - FIELDS:
            raise ValueError('Unsupported application CSS edits: ' + ', '.join(sorted(changed - FIELDS)))
        if not changed:
            continue
        declarations = []
        for name in sorted(changed):
            value, priority = final.get(name, ('', ''))
            declarations.append(dict(property=name, value=value, priority=priority,
                kind='normalization' if name in NORMALIZATION_FIELDS else 'layout'))
        edits.append(dict(selector=selector, tag=node.name, id=node.get('id', ''),
                          text=_fingerprint(node), declarations=declarations))
    if not edits:
        return source_html, []
    if OVERLAY_ID in source_html:
        raise ValueError('Application already contains a layout overlay; use the original source')
    closing = list(re.finditer(r'</body\s*>', source_html, re.I))
    if not closing:
        raise ValueError('Application source must have a closing body tag')
    # Escaping '<' prevents script termination via JSON strings from the page.
    payload = json.dumps(edits, ensure_ascii=True).replace('<', '\\u003c')
    script = '''<script id="''' + OVERLAY_ID + '''">
(() => {
  const edits = ''' + payload + ''';
  const report = {contract:'guarded-application-layout-overlay-v1',applied:[],errors:[]};
  window.__tuideApplicationEdits = report;
  const text = e => [...e.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE)
    .map(n=>n.textContent).join(' ').replace(/\\s+/g,' ').trim();
  let queued=false;
  function apply(){
    queued=false;report.applied=[];report.errors=[];
    for(const edit of edits){
      const nodes=document.querySelectorAll(edit.selector);
      if(nodes.length!==1){report.errors.push({selector:edit.selector,reason:'missing_or_ambiguous'});continue;}
      const e=nodes[0];
      if(e.tagName.toLowerCase()!==edit.tag||(e.id||'')!==edit.id||text(e)!==edit.text){
        report.errors.push({selector:edit.selector,reason:'identity_mismatch'});continue;
      }
      for(const d of edit.declarations){
        if(d.value==='' ){
          if(e.style.getPropertyValue(d.property))e.style.removeProperty(d.property);
        }else if(!CSS.supports(d.property,d.value)){
          report.errors.push({selector:edit.selector,reason:'unsupported_css',property:d.property});
        }else{
          // Compare canonical CSSOM values, e.g. 180.0px -> 180px, so our own
          // attribute mutation cannot schedule an endless reapplication loop.
          const style=document.createElement('div').style;
          style.setProperty(d.property,d.value,d.priority);
          const value=style.getPropertyValue(d.property),priority=style.getPropertyPriority(d.property);
          if(e.style.getPropertyValue(d.property)!==value||e.style.getPropertyPriority(d.property)!==priority)
            e.style.setProperty(d.property,value,priority);
        }
      }
      report.applied.push(edit.selector);
    }
  }
  function schedule(){if(!queued){queued=true;queueMicrotask(apply);}}
  new MutationObserver(schedule).observe(document.documentElement,
    {subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['style','id','class']});
  apply();
})();
</script>'''
    offset = closing[-1].start()
    return source_html[:offset] + script + source_html[offset:], edits

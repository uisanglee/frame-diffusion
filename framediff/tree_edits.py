"""Typed CSS subtree replacements, independent of rendered-image distance.

Reference: revalo/tree-diffusion@e8b29f2, td/samplers/mutator.py.
Our restricted tree has fixed DOM nodes and six editable declaration children.
No claim of minimum image distance, full HTML synthesis, or generic CSS parsing.
"""
import copy
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .ir import read_json, read_jsonl, write_json, write_jsonl
from .visual import NUMERIC_FIELDS, elements
from .web_experiment import digest, guard_run

CONTRACT = 'css-declaration-tree-v1'
FIELDS = tuple(NUMERIC_FIELDS)
UNITS = ('px', '%', 'em', 'rem', 'vw', 'vh', 'vmin', 'vmax', 'ch', 'ex', 'cm', 'mm', 'in', 'pt', 'pc', 'q')
KEYWORDS = ('auto', 'inherit', 'initial', 'unset', 'revert', 'revert-layer')
MAX_VALUE = 32


def keywords(field):
    return KEYWORDS + (('min-content', 'max-content', 'fit-content') if field in ('width', 'height') else ())


def value_valid(value, field):
    if not isinstance(value, str) or len(value) > MAX_VALUE: return False
    if value in keywords(field): return True
    m = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))([a-z%]*)', value)
    if not m: return False
    number, unit = m.groups()
    if field in ('width', 'height') and float(number) < 0: return False
    return unit in UNITS or (not unit and float(number) == 0)


def value_prefix(value, field):
    if len(value) > MAX_VALUE: return False
    if any(k.startswith(value) for k in keywords(field)): return True
    sign = r'[+]?' if field in ('width', 'height') else r'[+-]?'
    if re.fullmatch(sign + r'\d*(?:\.\d*)?', value):
        return len(value) < MAX_VALUE or value_valid(value, field)
    m = re.fullmatch(sign + r'(?:\d+(?:\.\d*)?|\.\d+)([a-z%]+)', value)
    return bool(m and any(u.startswith(m[1]) and len(value) + len(u) - len(m[1]) <= MAX_VALUE for u in UNITS))


def validate_edit(edit, node_count):
    node, field, value, priority = edit
    if type(node) is not int or not 0 < node < node_count or field not in FIELDS:
        raise ValueError('Invalid CSS declaration position')
    if priority not in ('', 'important') or (value == '' and priority):
        raise ValueError('Invalid CSS declaration priority')
    if value != '' and not value_valid(value, field):
        raise ValueError(f'Outside restricted CSS grammar: {field}={value!r}')
    return list(edit)


def apply_state(state, edit):
    node, field, value, priority = validate_edit(edit, len(state))
    result = copy.deepcopy(state)
    result[node][FIELDS.index(field)] = [value, priority]
    return result


def repair_path(current, target, seed=0):
    """Diff actual declarations, not corruption history or negated deltas.

    For this fixed-topology, single-declaration grammar every differing child
    can be replaced directly. The path ends at exact target declaration state.
    Shuffle independent substitutions, as the reference randomizes path order.
    """
    if len(current) != len(target): raise ValueError('DOM topology mismatch')
    path = []
    for node in range(1, len(current)):
        for j, field in enumerate(FIELDS):
            if current[node][j] != target[node][j]:
                path.append(validate_edit([node, field, *target[node][j]], len(current)))
    random.Random(seed).shuffle(path)
    state = current
    for action in path: state = apply_state(state, action)
    if state != target: raise ValueError('Path does not restore target declarations')
    return path


# DOMParser creates inert documents: no scripts, navigation, screenshot or layout.
# CSSOM expands margin shorthands and resolves duplicate inline declarations.
EXTRACT_JS = r'''({html, ids, fields}) => {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const nodes = [...doc.querySelectorAll('[data-fd-id]')];
  if (new Set(nodes.map(e=>e.getAttribute('data-fd-id'))).size !== nodes.length)
    throw new Error('Duplicate DOM IDs');
  const state = [fields.map(()=>['',''])];
  for (const id of ids) {
    const el = nodes.find(e=>e.getAttribute('data-fd-id')===id);
    if (!el) throw new Error('Missing DOM ID: '+id);
    const aliases=['all','inline-size','block-size','margin-inline','margin-block',
      'margin-inline-start','margin-inline-end','margin-block-start','margin-block-end'];
    if(aliases.some(p=>el.style.getPropertyValue(p)))
      throw new Error('Unsupported overlapping inline CSS declarations: '+id);
    state.push(fields.map(p=>[el.style.getPropertyValue(p),el.style.getPropertyPriority(p)]));
    for (const p of fields) el.style.removeProperty(p);
  }
  // Compare the noneditable DOM/CSS too. A categorical ancestor change is not
  // repairable by this grammar and must not silently become training noise.
  doc.querySelectorAll('style[data-framediff-static]').forEach(e=>e.remove());
  function serial(node) {
    if (node.nodeType!==1) return [node.nodeType,node.textContent];
    const attrs = [...node.attributes].filter(a=>a.name!=='style').map(a=>[a.name,a.value]).sort();
    const style = [...node.style].map(p=>[p,node.style.getPropertyValue(p),node.style.getPropertyPriority(p)]).sort();
    return [node.tagName,attrs,style,[...node.childNodes].map(serial)];
  }
  return {state, fixed:JSON.stringify(serial(doc.documentElement))};
}'''


def extract(browser, html, tree):
    return browser.page.evaluate(EXTRACT_JS, {'html': html, 'ids': [n['id'] for n in tree['nodes'][1:]], 'fields': FIELDS})


def current_state(browser, tree):
    """Read current declarations from the live DOM; no target or HTML reparsing."""
    state=browser.page.evaluate('''({ids,fields})=>ids.map(id=>{
      const e=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.getAttribute('data-fd-id')===id);
      if(!e)throw new Error('Missing DOM ID');
      return fields.map(p=>[e.style.getPropertyValue(p),e.style.getPropertyPriority(p)]);
    })''',{'ids':[n['id'] for n in tree['nodes'][1:]],'fields':FIELDS})
    return [[['',''] for _ in FIELDS],*state]


def execute(browser, tree, edit):
    """Mutate one declaration of the already loaded document, preserve cascade."""
    node, field, value, priority = validate_edit(edit, len(tree['nodes']))
    browser.page.evaluate('''({id,field,value,priority})=>{
      const e=[...document.querySelectorAll('[data-fd-id]')].find(e=>e.getAttribute('data-fd-id')===id);
      if(!e)throw new Error('Missing edit target');
      if(value==='')e.style.removeProperty(field);
      else {
        if(!CSS.supports(field,value))throw new Error('Browser rejected CSS grammar value');
        e.style.setProperty(field,value,priority);
      }
    }''', {'id': tree['nodes'][node]['id'], 'field': field, 'value': value, 'priority': priority})
    browser.executions += 1


def prepare(args):
    """Reuse cached observations, parse CSS once; never rerender the corpus.

    We sample the first edge of a shuffled exact declaration path from each
    cached state. Unlike the reference's online full-program sampler, this
    adapter does NOT invent unseen intermediate images from old screenshots.
    """
    from .html_bridge import HtmlBrowser
    from .visual_data import validate_splits
    source, out = Path(args.rendered).resolve(), Path(args.out).resolve()
    stylesheets=getattr(args,'stylesheets',False)
    from . import css_owners
    contract=css_owners.CONTRACT if stylesheets else CONTRACT
    def parse(browser,html,tree):
        return css_owners.read(browser,tree,html,fixed=True) if stylesheets else extract(browser,html,tree)
    if source == out or source in out.parents: raise ValueError('Use a separate output directory')
    files = [source/f'{kind}-{split}.jsonl' for split in ('train','val','test') for kind in ('policy','pages')]
    config={'kind':contract, 'sources':{str(p):digest(p) for p in files}, 'seed':args.seed}
    if stylesheets:config['max_css_owners']=getattr(args,'max_css_owners',512)
    guard_run(out, config, args.resume)
    report = {}; all_rows = []
    with HtmlBrowser() as browser:
        for split in ('train','val','test'):
            source_rows = list(read_jsonl(source/f'policy-{split}.jsonl')); grouped = defaultdict(list)
            for row in source_rows: grouped[row['id'].rsplit('/',1)[0]].append(row)
            kept = []; failures = []; counts = Counter(); asset_hashes = {}
            for page_index,(page_id, rows) in enumerate(grouped.items()):
                cache=out/'parsed-pages'/split/(hashlib.sha256(page_id.encode()).hexdigest()+'.json')
                signatures={r['current_html']:digest(r['current_html']) for r in rows}
                if args.resume and cache.exists():
                    saved=read_json(cache)
                    if saved['signatures']==signatures:
                        kept.extend(saved['rows']);failures.extend(saved['failures']);counts.update(saved['counts'])
                        asset_hashes.update(signatures);continue
                before_rows=len(kept);before_failures=len(failures);before_counts=counts.copy()
                if page_index%100==0:
                    browser.reset_context()
                    print(f'{split} CSS parsing [{page_index+1}/{len(grouped)}], kept={len(kept)}',flush=True)
                clean = next((r for r in rows if r['id'].endswith('/clean')), None)
                if clean is None:
                    failures.append({'id':page_id,'error':'Missing clean cached HTML state'}); continue
                try:
                    target = parse(browser, Path(clean['current_html']).read_text(), clean['current'])
                    if stylesheets and len(target['owners'])>config['max_css_owners']:
                        raise ValueError(f"CSS owner count {len(target['owners'])} exceeds max_css_owners={config['max_css_owners']}")
                    asset_hashes[clean['current_html']] = digest(clean['current_html'])
                    target_elements = elements(clean['current'], clean['current_boxes'], clean['viewport'])
                except Exception as exc:
                    failures.append({'id':page_id,'error':str(exc)})
                    if 'crash' in str(exc).lower() or 'closed' in str(exc).lower(): browser.reset_context()
                    continue
                for row in rows:
                    try:
                        if row['id'].endswith('/clean'): counts['clean_not_policy_examples'] += 1; continue
                        if row['current']['nodes'] != clean['current']['nodes']:
                            # Geometry-derived props legitimately vary; IDs/parents/roles must not.
                            identity = lambda r:[(n['id'],n.get('parent'),n.get('role')) for n in r['current']['nodes']]
                            if identity(row) != identity(clean): raise ValueError('DOM topology changed')
                        current = parse(browser, Path(row['current_html']).read_text(), row['current'])
                        if stylesheets and current['owners']!=target['owners']: raise ValueError('CSS owner topology changed')
                        asset_hashes[row['current_html']] = digest(row['current_html'])
                        if current['fixed'] != target['fixed']: raise ValueError('Noneditable DOM/CSS differs')
                        seed = int(hashlib.sha256(f'{args.seed}:{row["id"]}'.encode()).hexdigest()[:16],16)
                        path = repair_path(current['state'], target['state'], seed)
                        if not path: counts['already_restored'] += 1; continue
                        # Delete all previous teacher contracts, scores and STOP labels.
                        new = {k:v for k,v in row.items() if k not in (
                            'teacher_edits','improving_edits','tree_path_edits','tree_edit_distance',
                            'teacher_strategy','corruption_edit','policy_subset')}
                        new.update(teacher_strategy=contract, declaration_state=current['state'],
                                   replacement_edit=path[0], symbolic_distance=len(path), target_elements=target_elements)
                        if stylesheets:
                            new.update(css_owners=current['owners'],target_html=clean['current_html'])
                        kept.append(new); counts['edit_rows'] += 1
                        counts['remove_edits' if not path[0][2] else 'set_edits'] += 1
                        counts['property/'+path[0][1]] += 1
                    except Exception as exc:
                        failures.append({'id':row['id'],'error':str(exc)})
                        # A crashed parser page must not poison the rest of the corpus.
                        if 'crash' in str(exc).lower() or 'closed' in str(exc).lower(): browser.reset_context()
                temporary=cache.with_suffix('.tmp')
                write_json(temporary,{'signatures':signatures,'rows':kept[before_rows:],
                                      'failures':failures[before_failures:],'counts':dict(counts-before_counts)})
                temporary.replace(cache)
            by_page={r['id'].rsplit('/',1)[0]:r for r in kept}
            pages = [{**p,'initial_html':by_page[p['id']]['current_html'],
                      'policy_subset':contract} for p in read_jsonl(source/f'pages-{split}.jsonl') if p['id'] in by_page]
            write_jsonl(out/f'policy-{split}.jsonl', kept); write_jsonl(out/f'pages-{split}.jsonl', pages)
            write_jsonl(out/f'rejections-{split}.jsonl', failures)
            write_json(out/f'html-signatures-{split}.json', asset_hashes)
            report[split] = {'source_rows':len(source_rows),'kept':len(kept),'pages':len(pages),
                             'rejected':len(failures),'counts':dict(counts)}
            all_rows.extend(kept); print({split:report[split]},flush=True)
    validate_splits(all_rows)
    write_json(out/'prepare-report.json', report)
    if any(not report[s]['kept'] for s in ('train','val','test')):
        raise ValueError('Empty split; inspect rejections-*.jsonl. Source assets were not changed.')


class EditTokenizer:
    """node → field → replacement value/absence → priority → EOS. No STOP."""
    def __init__(self, max_nodes):
        self.tokens = ['PAD','BOS','EOS','SET','REMOVE','NORMAL','IMPORTANT']
        self.tokens += [f'N{i}' for i in range(max_nodes)] + list(FIELDS)
        self.tokens += list('0123456789.+-%abcdefghijklmnopqrstuvwxyz')
        self.ids = {t:i for i,t in enumerate(self.tokens)}
        self.max_nodes = max_nodes; self.max_length = MAX_VALUE + 5

    def encode(self, edit, node_count):
        node,field,value,priority = validate_edit(edit,node_count)
        text = [f'N{node}',field] + (['SET',*value,'IMPORTANT' if priority else 'NORMAL'] if value else ['REMOVE']) + ['EOS']
        return [self.ids[t] for t in text]

    def allowed(self, prefix, node_count):
        p = [self.tokens[i] for i in prefix]
        if not p: choices = [f'N{i}' for i in range(1,min(node_count,self.max_nodes))]
        elif len(p)==1: choices = list(FIELDS)
        elif len(p)==2: choices = ['SET','REMOVE']
        elif p[-1] == 'EOS': choices = []
        elif p[2]=='REMOVE' or p[-1] in ('NORMAL','IMPORTANT'): choices = ['EOS']
        else:
            value=''.join(p[3:]);field=p[1]
            choices=[c for c in '0123456789.+-%abcdefghijklmnopqrstuvwxyz' if value_prefix(value+c,field)]
            if value_valid(value,field): choices += ['NORMAL','IMPORTANT']
        return [self.ids[t] for t in choices]

    def decode(self, tokens, node_count):
        prefix=[]
        for t in tokens:
            if t not in self.allowed(prefix,node_count): raise ValueError('Invalid edit token sequence')
            prefix.append(t)
        p=[self.tokens[t] for t in tokens]
        if p[-1]!='EOS': raise ValueError('Incomplete replacement')
        edit=[int(p[0][1:]),p[1],'','']
        if p[2]=='SET': edit[2]=''.join(p[3:-2]);edit[3]='important' if p[-2]=='IMPORTANT' else ''
        return validate_edit(edit,node_count)

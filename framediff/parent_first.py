"""Optional parent-first scheduling; no target DOM correspondence is required."""
import copy


def owner_order(tree, owners=None):
    nodes=tree['nodes'];parents={n['id']:n.get('parent') for n in nodes}
    def depth(key):
        seen=set();d=0
        while key in parents and parents[key] is not None:
            if key in seen:raise ValueError('Cyclic DOM hierarchy')
            seen.add(key);key=parents[key];d+=1
        return d
    if owners is None:
        depths={i:depth(n['id']) for i,n in enumerate(nodes) if i}
    else:
        depths={}
        for i,o in enumerate(owners):
            if not i:continue
            ids=[o['id']] if o['kind']=='inline' else o.get('matches',[])
            values=[depth(key) for key in ids if key in parents]
            if values:depths[i]=min(values)
    return sorted(depths,key=lambda i:(depths[i],i))


def improving_teacher(browser, tree, viewport, size, state, edits, target_mask, owners=None):
    """Choose first strictly improving repair in ancestor-first order.

    Target CSS is used ONLY to enumerate supervised training/validation labels.
    Inference never calls this function.
    """
    from .tree_online import abstraction
    from .tree_edits import FIELDS,execute
    from . import css_owners
    order={v:i for i,v in enumerate(owner_order(tree,owners))}
    before,_,_=abstraction(browser,tree,viewport,size)
    distance=float((before-target_mask).abs().mean())
    def apply(edit):
        return css_owners.execute(browser,owners,edit) if owners is not None else execute(browser,tree,edit)
    for edit in sorted(edits,key=lambda e:order.get(e[0],len(order))):
        old=state[edit[0]][FIELDS.index(edit[1])]
        try:
            apply(edit)
            after,_,_=abstraction(browser,tree,viewport,size)
            gain=distance-float((after-target_mask).abs().mean())
        finally:apply([edit[0],edit[1],*old])
        if gain>1e-8:return edit,gain
    return None,0.


def relabel_validation(rows,size):
    """Read-only relabeling of fixed validation states, never test states."""
    from pathlib import Path
    from .html_bridge import HtmlBrowser
    from .tree_online import abstraction
    from .tree_edits import repair_path
    from . import css_owners
    kept=[];excluded=[]
    with HtmlBrowser() as browser:
        for row in rows:
            browser.reset_context();browser.load(Path(row['target_html']).read_text(),row['viewport'])
            mask,_,_=abstraction(browser,row['current'],row['viewport'],size)
            clean=css_owners.read(browser,row['current'])['state']
            browser.load(Path(row['current_html']).read_text(),row['viewport'])
            parsed=css_owners.read(browser,row['current'])
            if parsed['owners']!=row['css_owners']:raise ValueError('Validation owner identity changed')
            edit,gain=improving_teacher(browser,row['current'],row['viewport'],size,
                parsed['state'],repair_path(parsed['state'],clean),mask,parsed['owners'])
            if edit is None:excluded.append(row['id']);continue
            item=copy.copy(row);item.update(replacement_edit=edit,parent_first_gain=gain)
            kept.append(item)
    if not kept:raise ValueError('No improving parent-first validation labels')
    return kept,excluded

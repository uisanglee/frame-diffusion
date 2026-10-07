from pathlib import Path

import pytest

from framediff import css_owners
from framediff.html_bridge import HtmlBrowser
from framediff.plans import dom_tree
from framediff.visual import annotate,elements
from framediff.tree_online import target_pool,OnlineSampler,abstraction,changed


def make_rows(tmp_path):
    bodies={
        'inert':'<div style="display:flex;gap:10px"><button>A</button></div>',
        'hidden':'<button style="display:flex;min-width:200px;min-height:60px;gap:10px"><span>A</span><span>B</span></button>',
        'active':'<div style="display:flex;gap:10px"><button>A</button><button>B</button></div>',
    }
    rows=[]
    with HtmlBrowser() as b:
        for name,body in bodies.items():
            image=tmp_path/f'{name}.png';path=tmp_path/f'{name}.html'
            dom=b.snapshot(f'<html><body>{body}</body></html>',[400,300],image,max_nodes=16)
            tree,boxes,_=dom_tree(dom);annotate(b,tree);parsed=css_owners.read(b,tree)
            path.write_text(dom['html'])
            rows.append(dict(id=name+'/clean',split='train',group=name,source_sha=name,
                current=tree,viewport=[400,300],target_html=str(path),target_image=str(image),
                target_elements=elements(tree,boxes,[400,300]),css_owners=parsed['owners'],
                target_declaration_state=parsed['state']))
    return rows


@pytest.mark.browser
def test_preparsed_inert_and_unobservable_pages_excluded_and_cached(tmp_path,monkeypatch):
    rows=make_rows(tmp_path);cache=tmp_path/'cache'
    pool=target_pool(rows,cache_dir=cache,observation_size=64)
    assert [p['id'] for p in pool]==['active']
    assert {e[1] for e in pool[0]['visible_candidates']}=={'column-gap'}
    import framediff.tree_online as online
    with monkeypatch.context() as patch:
        patch.setattr(online,'HtmlBrowser',lambda:pytest.fail('Cached pages must not open Chromium'))
        assert target_pool(rows,cache_dir=cache,observation_size=64)==pool
        with pytest.raises(ValueError,match='Empty online target pool'):
            target_pool(rows[:2],cache_dir=cache,observation_size=64)
    # Resolution and viewport changes invalidate the eligibility cache.
    count=len(list(cache.glob('*.json')))
    target_pool(rows,cache_dir=cache,observation_size=32)
    assert len(list(cache.glob('*.json')))==count*2
    target_pool([{**r,'viewport':[500,300]} for r in rows],cache_dir=cache,observation_size=64)
    assert len(list(cache.glob('*.json')))==count*3


@pytest.mark.browser
def test_every_sample_and_teacher_change_abstraction_in_both_modes(tmp_path):
    pool=target_pool(make_rows(tmp_path),cache_dir=tmp_path/'cache',observation_size=64)
    states=[]
    for mode in ('abstract','screenshot'):
        sampler=OnlineSampler(pool,mode,seed=9,max_noise=4)
        try:
            local=[]
            for i in range(4):
                row=sampler.generate(i);b=sampler.browser;tree=row['current'];viewport=row['viewport']
                frame,_,_=abstraction(b,tree,viewport,64)
                css_owners.execute(b,row['css_owners'],row['replacement_edit'])
                repaired,_,_=abstraction(b,tree,viewport,64)
                assert changed(frame,repaired)
                assert row['online']['abstraction_max_difference']>0
                assert row['replacement_edit'][1]=='column-gap'
                b.load(Path(row['target_html']).read_text(),viewport)
                previous,_,_=abstraction(b,tree,viewport,64)
                for edit in row['online']['corruptions']:
                    css_owners.execute(b,row['css_owners'],edit)
                    next_frame,_,_=abstraction(b,tree,viewport,64)
                    assert changed(previous,next_frame)
                    previous=next_frame
                local.append((row['declaration_state'],row['replacement_edit']))
            states.append(local)
        finally:sampler.close()
    assert states[0]==states[1]

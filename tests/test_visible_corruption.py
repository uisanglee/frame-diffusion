from pathlib import Path

import pytest

from framediff import css_owners
from framediff.html_bridge import HtmlBrowser
from framediff.plans import dom_tree
from framediff.visual import annotate,elements
from framediff.tree_online import target_pool,OnlineSampler,abstraction,changed


def make_rows(tmp_path):
    bodies={
        'inert':'<span style="width:100px">A</span>',
        'hidden':'<button style="min-width:200px;min-height:60px;max-width:200px;max-height:60px"><span style="margin-left:10px">A</span></button>',
        'active':'<div style="display:flex;gap:10px"><button style="width:80px">A</button><button>B</button></div>',
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
    diagnostics=[]
    pool=target_pool(rows,cache_dir=cache,observation_size=64,diagnostics=diagnostics)
    assert [r['reason'] for r in diagnostics]==['no_visible_candidate','no_visible_candidate','usable']
    assert all(r['sites'] for r in diagnostics)
    assert all(p['outcome'] in ('no_box_change','no_abstraction_change','visible','explicit_constraint_rejected')
               for r in diagnostics for s in r['sites'] for p in s['probes'])
    assert [p['id'] for p in pool]==['active']
    assert {e[1] for e in pool[0]['visible_candidates']}=={'width'}
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
def test_inspection_failures_are_separate_and_not_cached(tmp_path,monkeypatch,capsys):
    import framediff.tree_online as online
    rows=make_rows(tmp_path)
    rows[0]['target_declaration_state']=[[['',''] for _ in online.FIELDS] for _ in rows[0]['current']['nodes']]
    def fail(*args,**kwargs):raise RuntimeError('capture fixture')
    monkeypatch.setattr(online,'inspect_candidates',fail)
    diagnostics=[]
    with pytest.raises(online.EmptyTargetPoolError) as exc:
        target_pool(rows,cache_dir=tmp_path/'cache',diagnostics=diagnostics)
    assert exc.value.diagnostics==diagnostics
    assert [r['reason'] for r in diagnostics]==['no_editable_css','inspection_error','inspection_error']
    assert not list((tmp_path/'cache').glob('*.json'))
    log=capsys.readouterr().out
    assert "'excluded_no_editable_css': 1" in log
    assert "'excluded_no_visible_candidate': 0" in log
    assert "'excluded_inspection_error': 2" in log


@pytest.mark.browser
@pytest.mark.parametrize('saved_input',[True,False])
def test_candidate_audit_reuses_saved_inputs_without_normalizing(tmp_path,monkeypatch,saved_input):
    import hashlib
    from types import SimpleNamespace
    from framediff.ir import write_json,write_jsonl,read_json
    from framediff.candidate_audit import run
    import framediff.explicit_html as explicit
    rows=make_rows(tmp_path)
    root=tmp_path/'normalized';root.mkdir();out=tmp_path/'audit'
    rejections=[]
    for row in rows:
        id=row['id'].rsplit('/',1)[0]
        folder=root/'normalized-pages'/'train'/hashlib.sha256(id.encode()).hexdigest()
        folder.mkdir(parents=True)
        if saved_input:write_json(folder/'inspection-input.json',row)
        else:(folder/'normalized.html').write_text(Path(row['target_html']).read_text())
        rejections.append(dict(id=id,stage='visible_candidate_inspection'))
    write_jsonl(root/'rejections-train.jsonl',rejections)
    source=tmp_path/'source';source.mkdir()
    write_jsonl(source/'policy-train.jsonl',rows)
    write_json(root/'config.json',{'sources':{str(source/'policy-train.jsonl'):'unused'}})
    monkeypatch.setattr(explicit,'convert',lambda *a,**k:pytest.fail('Must not normalize'))
    run(SimpleNamespace(normalized=str(root),out=str(out),split='train',limit=0,
                        source=None,observation_size=64))
    assert read_json(out/'summary.json')['reasons']=={'no_visible_candidate':2,'usable':1}


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
                assert row['replacement_edit'][1]=='width'
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

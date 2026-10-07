import copy
import random
from pathlib import Path

import pytest
import torch

from framediff.tree_online import OnlineSampler,OnlineStream,target_pool,mutation_sites,sample_mutation,task_seed
from framediff.tree_edits import CONTRACT,FIELDS,extract,current_state,execute,repair_path,apply_state
from framediff.ir import write_jsonl,read_jsonl


def test_mutations_only_replace_existing_supported_values_and_preserve_priority():
    state=[[['',''] for _ in FIELDS] for _ in range(3)]
    state[1][0]=['50%',''];state[1][1]=['calc(100px - 2px)','']
    state[2][0]=['80px','important']
    rng=random.Random(2);edits=[sample_mutation(state,[500,300],rng) for _ in range(200)]
    assert {e[0] for e in edits}=={1,2}
    for node,field,value,priority in edits:
        old=state[node][FIELDS.index(field)]
        assert field=='width' and value and value!=old[0]
        assert priority==old[1]
    with pytest.raises(ValueError,match='No supported existing'):
        sample_mutation([[['',''] for _ in FIELDS] for _ in range(3)],[500,300],rng)
    assert task_seed(42,10)==task_seed(42,10)!=task_seed(42,11)


def test_online_pool_rejects_heldout_and_missing_clean_html():
    with pytest.raises(ValueError,match='training pages'): target_pool([{'split':'val'}])
    with pytest.raises(ValueError,match='Missing clean HTML'):
        target_pool([{'id':'a/t0-s0','split':'train'}])


def test_online_pool_filters_permanently_unmutable_preparsed_pages(tmp_path):
    html=tmp_path/'clean.html';html.write_text('<html></html>')
    empty=[[['',''] for _ in FIELDS] for _ in range(2)]
    usable=copy.deepcopy(empty);usable[1][0]=['50%','']
    base={'split':'train','group':'g','source_sha':'x','viewport':[100,100],
          'current':{'nodes':[]},'target_image':str(tmp_path/'target.png'),'target_elements':[],
          'target_html':str(html)}
    rows=[{**base,'id':'bad/t0-s0','target_declaration_state':empty},
          {**base,'id':'good/t0-s0','target_declaration_state':usable}]
    pool=target_pool(rows)
    assert [row['id'] for row in pool]==['good']
    assert pool[0]['target_declaration_state']==usable
    sampler=OnlineSampler(pool,'abstract')
    assert sampler.clean_state('good')==usable
    assert mutation_sites(usable)==[(1,'width','50%','')]
    with pytest.raises(ValueError,match='after filtering'):
        target_pool([rows[0]])


@pytest.fixture
def corpus(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate,elements
    from framediff.html_feedback import refresh_geometry
    from framediff.web_experiment import digest
    html='''<html><body><section style="display:flex;width:300px">
    <div style="width:50%;height:50px;margin:2px">SPLIT</div><div>B</div>
    </section></body></html>'''
    records={}
    with HtmlBrowser() as browser:
        for split in ('train','val'):
            work=tmp_path/split;work.mkdir()
            dom=browser.snapshot(html.replace('SPLIT',split),[500,300],work/'target.png',12)
            tree,boxes,_=dom_tree(dom);annotate(browser,tree)
            clean=work/'target.html';clean.write_text(dom['html'])
            target=extract(browser,dom['html'],tree)['state']
            index=next(i for i,n in enumerate(tree['nodes']) if n['id']=='fd-2')
            execute(browser,tree,[index,'width','200px','important'])
            browser.page.screenshot(path=str(work/'current.png'))
            state=current_state(browser,tree)
            current_boxes=browser.tagged_boxes([n['id'] for n in tree['nodes'][1:]])
            current_boxes[tree['nodes'][0]['id']]=[0,0,500,300]
            row={'id':split+'/t0-s0','group':split,'source_sha':digest(clean),'split':split,
                 'teacher_strategy':CONTRACT,'replacement_edit':repair_path(state,target)[0],
                 'declaration_state':state,'target_html':str(clean),'target_image':str(work/'target.png'),
                 'current_image':str(work/'current.png'),'viewport':[500,300],
                 'current':refresh_geometry(tree,current_boxes),'current_boxes':current_boxes,
                 'target_elements':elements(tree,boxes,[500,300])}
            records[split]=row;write_jsonl(tmp_path/f'policy-{split}.jsonl',[row])
    return records


@pytest.mark.browser
def test_legacy_pool_inspects_external_only_pages_and_keeps_editable_css(tmp_path, capsys, monkeypatch):
    tree={'nodes':[{'id':'__viewport__'},{'id':'fd-0'}]}
    owners=[{'kind':'root','matches':[]},
            {'kind':'inline','id':'fd-0','matches':['fd-0']}]
    base={'split':'train','group':'g','source_sha':'x','viewport':[300,200],
          'current':tree,'css_owners':owners,'target_elements':[],
          'target_image':str(tmp_path/'target.png')}
    rows=[]
    for name,style in [('external','color:red'),('editable','width:120px')]:
        path=tmp_path/f'{name}.html'
        path.write_text(f'<html><head><link rel="stylesheet" href="styles.css"></head>'
                        f'<body><div data-fd-id="fd-0" style="{style}">Box</div></body></html>')
        rows.append({**base,'id':name+'/t0-s0','target_html':str(path)})
    inputs=[rows[0],{**rows[0],'id':'external/t1-s0'},rows[1]]
    cache=tmp_path/'cache'
    pool=target_pool(inputs,cache_dir=cache)
    assert [r['id'] for r in pool]==['editable']
    assert mutation_sites(pool[0]['target_declaration_state'])==[(1,'width','120px','')]
    output=capsys.readouterr().out
    assert "'inspected_missing_states': 2" in output
    assert "'excluded_no_editable_css': 1" in output
    # Both accepted and excluded pages must survive process restarts without
    # opening Chromium. Modalities carry different detector/image metadata.
    import framediff.tree_online as online
    with monkeypatch.context() as patch:
        def unexpected_browser(): raise AssertionError('Cache hit launched a browser')
        patch.setattr(online,'HtmlBrowser',unexpected_browser)
        again=target_pool([{**r,'predicted_target_elements':'another-cache'} for r in inputs],cache_dir=cache)
        assert [r['id'] for r in again]==['editable']
        assert "'cache_hits': 2" in capsys.readouterr().out
    # HTML content changes invalidate only that page, including prior exclusions.
    path=Path(rows[0]['target_html'])
    path.write_text(path.read_text().replace('color:red','width:90px'))
    assert len(target_pool(inputs,cache_dir=cache))==2
    output=capsys.readouterr().out
    assert "'inspected_missing_states': 1" in output and "'cache_hits': 1" in output
    # Parser/eligibility changes invalidate all old entries.
    original=online.inspection_fingerprint
    monkeypatch.setattr(online,'inspection_fingerprint',lambda: {**original(),'test_version':2})
    assert len(target_pool(inputs,cache_dir=cache))==2
    assert "'inspected_missing_states': 2" in capsys.readouterr().out
    path.write_text(path.read_text().replace('width:90px','color:red'))
    with pytest.raises(ValueError,match='Empty online target pool'):
        target_pool([rows[0]],cache_dir=cache)


@pytest.mark.browser
def test_fresh_path_states_and_paired_modalities(corpus):
    pool=target_pool([corpus['train']]);abstract=OnlineSampler(pool,'abstract',max_noise=5)
    raw=OnlineSampler(pool,'screenshot',max_noise=5)
    original=Path(pool[0]['target_html']).read_bytes()
    try:
        rows=[abstract.generate(i) for i in range(12)]
        assert len({str(r['declaration_state']) for r in rows})>1
        assert any(r['online']['reverse_prefix']>0 for r in rows)
        assert all('current_image' not in r for r in rows)
        clean=extract(abstract.browser,original.decode(),pool[0]['current'])['state']
        abstract.close()  # Only one Playwright sync event loop per process.
        same=raw.generate(2)
        assert same['current_boxes']==rows[2]['current_boxes']
        assert same['declaration_state']==rows[2]['declaration_state']
        assert same['replacement_edit']==rows[2]['replacement_edit']
        assert same['current_image'].startswith(b'\x89PNG')
        assert same['online']['screenshot_captures']==1
        assert all(r['online']['screenshot_captures']==0 for r in rows)
        for row in rows:
            assert row['replacement_edit'] in repair_path(row['declaration_state'],clean)
            state=copy.deepcopy(row['declaration_state'])
            for edit in repair_path(state,clean): state=apply_state(state,edit)
            assert state==clean
        assert Path(pool[0]['target_html']).read_bytes()==original
    finally: abstract.close();raw.close()


@pytest.mark.browser
def test_prefetch_bounded_order_and_resume(corpus):
    targets=target_pool([corpus['train']])
    with OnlineStream(targets,'abstract',workers=2,prefetch=2) as stream:
        rows=stream.take(3)
        assert [r['online']['index'] for r in rows]==[0,1,2]
        assert stream.cursor==3 and len(stream.pending)==2
    with OnlineStream(targets,'abstract',workers=1,prefetch=1,start=2) as resumed:
        again=resumed.take(1)[0]
        assert again['declaration_state']==rows[2]['declaration_state']
        assert again['replacement_edit']==rows[2]['replacement_edit']
    assert stream.pool is None and resumed.pool is None


@pytest.mark.browser
@pytest.mark.parametrize('mode',['abstract','screenshot'])
def test_online_training_resume_keeps_fixed_validation(corpus,tmp_path,mode):
    from framediff.cli import main
    validation_before=(tmp_path/'policy-val.jsonl').read_bytes()
    args=['visual-tree-train','--train',str(tmp_path/'policy-train.jsonl'),
          '--val',str(tmp_path/'policy-val.jsonl'),'--out',str(tmp_path/'run'),
          '--mode',mode,'--online-corruption','--online-workers','1','--online-prefetch','1',
          '--device','cpu','--no-pretrained','--size','32','--hidden','16','--layers','1',
          '--heads','2','--token-grid','2','--max-nodes','16','--batch-size','1',
          '--accumulation','1','--eval-every','1','--val-samples','1','--cpu-threads','2']
    main([*args,'--steps','1'])
    checkpoint=tmp_path/'run/last.pt'
    assert torch.load(checkpoint,weights_only=True)['online_cursor']==1
    args[args.index('--online-prefetch')+1]='2'  # Resource tuning preserves the stream.
    main([*args,'--steps','2','--resume',str(checkpoint)])
    assert torch.load(checkpoint,weights_only=True)['online_cursor']==2
    metrics=list(read_jsonl(tmp_path/'run/train.jsonl'))
    assert [r['online_samples'] for r in metrics if 'online_samples' in r]==[1,2]
    assert (tmp_path/'policy-val.jsonl').read_bytes()==validation_before


@pytest.mark.browser
def test_invalid_sources_fail_after_bounded_attempts(corpus):
    target=target_pool([corpus['train']])[0];target['target_html']='/missing/target.html'
    sampler=OnlineSampler([target],'abstract',attempts=2)
    try:
        with pytest.raises(RuntimeError,match='failed after 2 attempts'): sampler.generate(0)
    finally: sampler.close()

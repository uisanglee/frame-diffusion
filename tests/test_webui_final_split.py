from pathlib import Path

import pytest

from framediff import webui_final_split as split
from framediff.ir import read_json,read_jsonl,write_json,write_jsonl
from framediff.visual_data import validate_splits


def corpus(root,n=20):
    master=root/'master';source=[];tables={s:{k:[] for k in kinds} for s,kinds in
        [('rendered',('pages','policy','detector')),('labels',('pages','policy'))]}
    for i in range(n):
        for v in range(2):
            view=f'p{i}/v{v}';folder=master/'assets'/f'p{i}'/f'v{v}';folder.mkdir(parents=True)
            html=folder/'target.html';html.write_text(f'<html><body>{view}</body></html>')
            image=folder/'target.png';image.write_bytes(b'fixture-image')
            row=dict(id=view,page_id=f'p{i}',view=f'v{v}',group='domain:shared.example',
                split='train',source_sha=f'sha-{i}',html=str(html),screenshot=str(image),viewport=[320,240])
            source.append(row)
            for stage in tables:tables[stage]['pages'].append(row)
            clean=dict(row,id=view+'/seed',target_html=str(html),target_image=str(image),clean_target_only=True,
                       visual_geometry_contract='fixture',teacher_strategy='fixture')
            tables['labels']['policy'].append(clean)
            tables['rendered']['policy'].append(dict(clean,id=view+'/clean'))
            tables['rendered']['detector'].append(dict(clean,id=view+'/clean',image=str(image)))
    write_jsonl(master/'source/manifest.jsonl',source)
    for stage,kinds in tables.items():
        for kind,rows in kinds.items():
            for s in split.SPLITS:write_jsonl(master/stage/f'{kind}-{s}.jsonl',rows if s=='train' else [])
    return master


def fake_sampler(monkeypatch,fail=()):
    from framediff import tree_freeze
    calls=[]
    monkeypatch.setattr(split,'timed_page_map',lambda fn,tasks,workers,timeout:
                        ((i,fn(task),None) for i,task in enumerate(tasks)))
    monkeypatch.setattr(tree_freeze,'_inspect_page',lambda task:([
        dict(task[0],id=task[0]['id'].rsplit('/',1)[0])],None))
    class Sampler:
        def __init__(self,rows,*a,**kw):self.row=rows[0]
        def generate(self,index):
            calls.append(self.row['id'])
            if self.row['id'] in fail:raise ValueError('fixture generation failure')
            return dict(self.row,id=self.row['id']+f'/online-{index}',
                current_html_text='<html>Corrupted</html>',current_image=b'png',clean_target_only=False)
        def close(self):pass
    monkeypatch.setattr(tree_freeze,'OnlineSampler',Sampler)
    return calls


def test_page_split_reuses_assets_and_rebalances_failures(tmp_path,monkeypatch):
    master=corpus(tmp_path)
    original={str(p):p.read_bytes() for p in master.rglob('*') if p.is_file()}
    initial,_=split.allocate([f'p{i}' for i in range(20)],42)
    failed_page=next(p for p,s in initial.items() if s=='test')
    calls=fake_sampler(monkeypatch,fail=[failed_page+'/v0',failed_page+'/v1'])
    result=split.publish(tmp_path,workers=2)
    assert result['unique_pages']==19
    assert result['sizes']['7000']=={'train':17,'val':1,'test':1}
    all_rows=[]
    for s in split.SPLITS:
        pages=list(read_jsonl(tmp_path/f'7000/css/pages-{s}.jsonl'))
        rows=list(read_jsonl(tmp_path/f'7000/css/policy-{s}.jsonl'));all_rows+=rows
        assert len(pages)==2*result['sizes']['7000'][s]
        assert {r['page_id'] for r in pages}=={r['page_id'] for r in rows}
        assert all(r['split']==s and r['group']=='webui-page:'+r['page_id'] for r in rows)
        assert all(r['source_group']=='domain:shared.example' for r in rows)
        assert all(r['clean_target_only']==(s=='train') for r in rows)
        if s!='train':assert all(Path(r['initial_html']).is_file() for r in pages)
        detector=list(read_jsonl(tmp_path/f'7000/rendered/detector-{s}.jsonl'))
        assert {r['page_id'] for r in detector}=={r['page_id'] for r in pages}
    validate_splits(all_rows)
    assert failed_page not in {r['page_id'] for r in all_rows}
    assert all(Path(p).read_bytes()==content for p,content in original.items())
    previous_calls=list(calls)
    assert split.publish(tmp_path,workers=2)==result
    assert calls==previous_calls  # Both success and failure checkpoints reused.


def test_old_heldout_corruptions_replaced_but_old_manifests_backed_up(tmp_path,monkeypatch):
    from framediff.tree_online import ONLINE_CONTRACT,asset_signatures
    import hashlib
    master=corpus(tmp_path)
    assignment,_=split.allocate([f'p{i}' for i in range(20)],42)
    calls=fake_sampler(monkeypatch)
    write_json(master/'css/config.json',dict(corruption_contract=ONLINE_CONTRACT,
        seed=90210,max_noise=4,samples=1,observation_size=384))
    for row in read_jsonl(master/'labels/policy-train.jsonl'):
        if assignment[row['page_id']]=='train':continue
        target=dict(row,id=row['id'].rsplit('/',1)[0])
        folder=master/'css/test'/hashlib.sha256(target['id'].encode()).hexdigest()
        folder.mkdir(parents=True);html=folder/'0.html';html.write_text('OLD');png=folder/'0.png';png.write_bytes(b'old')
        saved=dict(row,id=target['id']+'/online-0',current_html=str(html),current_image=str(png),clean_target_only=False)
        write_json(folder/'record.json',dict(signature=asset_signatures([target]),rows=[saved]))
    write_jsonl(tmp_path/'7000/css/pages-test.jsonl',[{'old':True}])
    split.publish(tmp_path)
    assert len(calls)==40  # Eight held-out views times five new profiles.
    assert list((tmp_path/'split-history').glob('*/7000/css/pages-test.jsonl'))
    assert all(Path(r['current_html']).read_text()=='<html>Corrupted</html>'
               for s in ('val','test') for r in read_jsonl(tmp_path/f'7000/css/policy-{s}.jsonl'))


def test_allocation_deterministic_and_small_pool_rejected():
    pages=[str(i) for i in range(100)]
    assert split.allocate(pages,42)==split.allocate(reversed(pages),42)
    assert split.allocate(pages,42)[1]==dict(train=80,val=10,test=10)
    with pytest.raises(ValueError,match='At least 10'):split.allocate(pages[:9],42)


def test_timeout_cache_retries_only_when_budget_increases(tmp_path,monkeypatch):
    master=corpus(tmp_path)
    calls=fake_sampler(monkeypatch)
    row=next(read_jsonl(master/'labels/policy-train.jsonl'))
    settings=dict(seed=90210,max_noise=4,samples_per_page=1,observation_size=384)
    cached=split._heldout_cache(row,master,settings)/'result.json'
    saved=dict(id='p0/v0',rows=[],timeout_s=180,error={'error_type':'TimeoutError'})
    write_json(cached,saved)
    assert split._heldout((row,master,settings,180))==saved
    assert calls==[]
    recovered=split._heldout((row,master,settings,240))
    assert recovered['rows'] and calls==['p0/v0']
    assert split._heldout((row,master,settings,180))==recovered
    assert calls==['p0/v0']  # A successful cache is independent of the time budget.


def test_completion_order_does_not_change_final_manifests(tmp_path,monkeypatch):
    corpus(tmp_path);fake_sampler(monkeypatch)
    first=split.publish(tmp_path)
    paths=list((tmp_path/'7000').rglob('*.jsonl'))
    content={p:p.read_bytes() for p in paths}
    def reversed_map(fn,tasks,workers,timeout):
        tasks=list(tasks)
        return ((i,fn(tasks[i]),None) for i in reversed(range(len(tasks))))
    monkeypatch.setattr(split,'timed_page_map',reversed_map)
    assert split.publish(tmp_path)==first
    assert all(p.read_bytes()==data for p,data in content.items())
    progress=read_json(tmp_path/'master/final-split-progress.json')
    assert progress['stage']=='heldout-round-1'
    assert progress['processed']==progress['total']==progress['kept']==8
    assert progress['percent']==100 and progress['failures']==progress['timeouts']==0


def test_shared_sources_across_page_ids_publish_and_resume(tmp_path, monkeypatch):
    master = corpus(tmp_path)
    initial, _ = split.allocate([f'p{i}' for i in range(20)],42)
    train = next(p for p,s in initial.items() if s=='train')
    test = next(p for p,s in initial.items() if s=='test')
    path = master/'source/manifest.jsonl'
    rows = list(read_jsonl(path))
    for row in rows:
        if row['page_id'] in (train,test): row['source_sha']='shared-source'
    write_jsonl(path,rows)
    calls = fake_sampler(monkeypatch)
    report = split.publish(tmp_path)
    final = list(read_jsonl(tmp_path/'7000/source/manifest.jsonl'))
    validate_splits(final)
    assert len({r['split'] for r in final if r['page_id'] in (train,test)}) == 1
    assert report['unique_pages'] == 20
    previous = list(calls)
    assert split.publish(tmp_path) == report
    assert calls == previous


def test_transitive_source_groups_are_deterministic():
    pages = [str(i) for i in range(20)]
    links = [('0','a'),('1','a'),('1','b'),('2','b')]
    assignment, counts = split.allocate_linked(pages,42,links)
    assert assignment['0'] == assignment['1'] == assignment['2']
    assert sum(counts.values()) == 20 and all(counts.values())
    assert split.allocate_linked(reversed(pages),42,reversed(links)) == (assignment,counts)


def test_preserved_split_never_rebalances_after_failure(tmp_path,monkeypatch):
    master=corpus(tmp_path)
    original=list(read_jsonl(master/'source/manifest.jsonl'))
    for row in original:
        i=int(row['page_id'][1:])
        row['split']='train' if i<16 else 'val' if i<18 else 'test'
    write_jsonl(master/'source/manifest.jsonl',original)
    previous=tmp_path/'previous.jsonl';write_jsonl(previous,original)
    calls=fake_sampler(monkeypatch,fail=['p18/v0','p18/v1'])
    report=split.publish(tmp_path,preserve_splits_from=previous)
    assert report['sizes']['7000']==dict(train=16,val=2,test=1)
    assert report['target_counts']==dict(train=16,val=2,test=2)
    assert report['excluded_pages']==['p18']
    assignment=split.preserved_assignment(original)
    final=list(read_jsonl(tmp_path/'7000/source/manifest.jsonl'))
    assert all(row['split']==assignment[row['page_id']] for row in final)
    before=list(calls)
    assert split.publish(tmp_path,preserve_splits_from=previous)==report
    assert calls==before

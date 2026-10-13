from types import SimpleNamespace

import pytest

from framediff.ir import read_json, read_jsonl, write_json, write_jsonl
from framediff.webui_scaling import counts, nested_sources, subset, summarize


def sources():
    return [dict(id=f'{s}-{i}',group=f'{s}-domain-{i//2}',source_sha=f'{s}-{i}',split=s)
            for s,n in counts(100).items() for i in range(n)]


def test_audited_partial_master_and_nested_subset(tmp_path):
    from PIL import Image
    from framediff.webui_visual import _assign
    from framediff.webui_scaling import prepare
    raw=tmp_path/'raw'
    for i in range(29):
        folder=raw/str(i);folder.mkdir(parents=True)
        Image.new('RGB',(10,10)).save(folder/'phone-screenshot.png')
        (folder/'phone-html.html').write_text(f'<html>{i}</html>')
        (folder/'phone-url.txt').write_text(f'https://d{i}.example/')
    selected,_=_assign({str(i):[{'id':str(i)}] for i in range(29)},
                       counts(40),42,allow_partial=True)
    expected=[len(selected[s]) for s in ('train','val','test')]
    args=SimpleNamespace(root=str(raw),out=str(tmp_path/'out'),sizes=[10,29],
        master_counts=expected,allocation_counts=[32,4,4],views=['phone'],seed=42,resume=True)
    prepare(args)
    prepare(args)
    report=read_json(tmp_path/'out/split-report.json')
    assert list(report['sizes']['29'].values())==expected
    small=list(read_jsonl(tmp_path/'out/10/source/manifest.jsonl'))
    large=list(read_jsonl(tmp_path/'out/29/source/manifest.jsonl'))
    assert len(large)==29 and len(small)==10
    assert {r['id'] for r in small}<={r['id'] for r in large}
    bad=SimpleNamespace(**{**vars(args),'out':str(tmp_path/'bad'),'master_counts':[21,4,4]})
    with pytest.raises(ValueError,match='Allocation does not match'):
        prepare(bad)


def test_nested_domain_disjoint_and_exact_counts():
    rows=sources()
    result=nested_sources(rows,[20,50,100],42)
    assert result==nested_sources(list(reversed(rows)),[20,50,100],42)
    for size,selected in result.items():
        assert {s:sum(r['split']==s for r in selected) for s in counts(size)}==counts(size)
    assert {r['id'] for r in result[20]} < {r['id'] for r in result[50]} < {r['id'] for r in result[100]}
    assert {r['group'] for r in result[20] if r['split']=='test'}.isdisjoint(
        r['group'] for r in result[100] if r['split']=='train')
    with pytest.raises(ValueError,match='Cross-split leakage'):
        nested_sources(rows+[dict(rows[0],id='leak',split='test')],[20],42)
    with pytest.raises(ValueError,match='Insufficient'):
        nested_sources(rows,[200],42)


def test_raw_import_nested_manifests_and_resume(tmp_path):
    from PIL import Image
    from framediff.webui_scaling import prepare
    raw=tmp_path/'raw'
    for i in range(30):
        folder=raw/str(i);folder.mkdir(parents=True)
        stem=folder/'default_1280-720'
        Image.new('RGB',(1280,720)).save(str(stem)+'-screenshot.png')
        (folder/'default_1280-720-html.html').write_text(f'<html><body>page {i}</body></html>')
        (folder/'default_1280-720-url.txt').write_text(f'https://domain{i}.example/')
    out=tmp_path/'scales'
    args=SimpleNamespace(root=str(raw),out=str(out),sizes=[10,30],view='default_1280-720',seed=42,resume=True)
    prepare(args)
    original=(out/'10/source/manifest.jsonl').read_bytes()
    assert len(list(read_jsonl(out/'10/source/manifest-train.jsonl')))==8
    assert len(list(read_jsonl(out/'30/source/manifest-test.jsonl')))==3
    prepare(args)
    assert (out/'10/source/manifest.jsonl').read_bytes()==original


def test_multiview_split_counts_unique_pages_and_keeps_views_together(tmp_path):
    from PIL import Image
    from framediff.webui_scaling import prepare
    raw=tmp_path/'raw'
    views={'default_100-80':(100,80),'iPhone-Test':(40,90)}
    for i in range(30):
        folder=raw/str(i);folder.mkdir(parents=True)
        for view,size in views.items():
            stem=folder/view
            Image.new('RGB',size).save(str(stem)+'-screenshot.png')
            (folder/f'{view}-html.html').write_text(f'<html><body>page {i}</body></html>')
            (folder/f'{view}-url.txt').write_text(f'https://domain{i}.example/')
    out=tmp_path/'scales'
    args=SimpleNamespace(root=str(raw),out=str(out),sizes=[10,30],view=None,
        views=list(views),seed=42,resume=True)
    prepare(args)
    rows=list(read_jsonl(out/'30/source/manifest.jsonl'))
    assert len(rows)==60
    assert len({r['page_id'] for r in rows})==30
    assert all(len({r['split'] for r in rows if r['page_id']==page})==1 for page in {r['page_id'] for r in rows})
    report=read_json(out/'10/source/report.json')
    assert report['selected']=={'train':8,'val':1,'test':1}
    assert sum(report['selected_views'].values())==20


def test_subset_keeps_shared_paths_and_reports_attrition(tmp_path):
    rows=sources();selected=nested_sources(rows,[20,100],42)
    write_json(tmp_path/'config.json',{'sizes':[20,100]})
    for size,items in selected.items():write_jsonl(tmp_path/str(size)/'source/manifest.jsonl',items)
    for split in counts(100):
        original=[r for r in rows if r['split']==split]
        # Remove a small-subset page at normalization, and another at rendering.
        small=[r for r in selected[20] if r['split']==split]
        rejected=small[0]['id'];render_failed=small[1]['id']
        for folder,kinds in [('rendered',('policy','pages','detector')),('labels',('policy','pages')),('css',('policy','pages'))]:
            kept=[dict(r,html='/shared/target.html') for r in original
                  if r['id']!=render_failed and (folder=='rendered' or r['id']!=rejected)]
            for kind in kinds:write_jsonl(tmp_path/'master'/folder/f'{kind}-{split}.jsonl',kept)
            if folder=='labels':
                write_jsonl(tmp_path/'master'/folder/f'rejections-{split}.jsonl',
                            [dict(id=rejected,stage='normalization',error='changed')])
            if folder=='css' and split!='train':
                write_jsonl(tmp_path/'master'/folder/f'rejections-{split}.jsonl',[])
    subset(SimpleNamespace(data=str(tmp_path)))
    coverage=read_json(tmp_path/'20/coverage.json')
    assert coverage['rendering']['train']['excluded']==1
    assert coverage['normalization']['train']['rejected']==1
    assert coverage['fixed_corruption']['train']['pages']==14
    assert all(r['html']=='/shared/target.html' for r in read_jsonl(tmp_path/'20/css/pages-train.jsonl'))
    # Deterministic rerun does not trigger configuration mismatch.
    subset(SimpleNamespace(data=str(tmp_path)))
    runs=tmp_path/'runs'
    for size in (20,100):
        write_json(runs/str(size)/'detector-test-common.json',
                   dict(n=1,precision_iou50=.8,recall_iou50=.6,small_recall_iou50=None))
        write_json(runs/str(size)/'eval-s-abstract-common/summary.json',
                   [dict(n=1,method='abstract-policy',final_box_iou=.9)])
    summarize(SimpleNamespace(data=str(tmp_path),runs=str(runs)))
    reports=read_json(runs/'detector-comparison.json')
    assert len(reports)==2
    assert reports[0]['f1_iou50']==pytest.approx(2*.8*.6/1.4)
    assert len(read_json(runs/'policy-comparison.json'))==2


def test_clean_only_renderer_produces_detector_and_normalization_inputs(tmp_path):
    from framediff.visual_data import build
    html=tmp_path/'page.html'
    html.write_text('<html><body><div style="width:100px;height:40px;background:red">Test</div></body></html>')
    manifest=tmp_path/'manifest.jsonl'
    write_jsonl(manifest,[dict(id='page',group='example',split='train',html=str(html),viewport=[400,200])])
    out=tmp_path/'rendered'
    args=SimpleNamespace(manifest=str(manifest),out=str(out),resume=True,trajectories=3,max_noise=4,
        seed=42,max_nodes=32,abstract_size=64,min_elements=1,max_source_mae=1.,clean_only=True)
    build(args)
    policy=list(read_jsonl(out/'policy-train.jsonl'))
    assert len(policy)==1 and policy[0]['id']=='page/clean'
    assert len(list(read_jsonl(out/'detector-train.jsonl')))==1
    page=list(read_jsonl(out/'pages-train.jsonl'))[0]
    assert page['initial_html']==policy[0]['current_html']
    build(args)
    assert list(read_jsonl(out/'policy-train.jsonl'))==policy

import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from framediff.ir import read_jsonl, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[1]


def test_shell_view_names_preserve_spaces():
    command='source scripts/webui_views.sh; printf "%s\\n" "${views[@]}"'
    env={k:v for k,v in os.environ.items() if k not in ('WEBUI_VIEW','WEBUI_VIEWS')}
    result=subprocess.run(['bash','-c',command],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    assert len(result.stdout.splitlines())==6
    assert result.stdout.splitlines()[-1]=='iPhone-13 Pro'
    env['WEBUI_VIEWS']='default_1280-720,iPhone-13 Pro'
    result=subprocess.run(['bash','-c',command],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    assert result.stdout.splitlines()==['default_1280-720','iPhone-13 Pro']
    env['WEBUI_VIEWS']='default_1280-720 iPhone-13 Pro'
    assert subprocess.run(['bash','-c',command],cwd=ROOT,env=env,capture_output=True).returncode==2


def test_preindexed_sampler_and_unlimited_validation():
    from framediff.visual_train import page_view_pools,sample_page_views,validation_sample
    rows=[dict(id=f'p{i}/v{j}',page_id=f'p{i}',source_sha=str(i)) for i in range(1001) for j in range(3)]
    pools=page_view_pools(iter(rows))
    assert len(pools)==1001
    assert sample_page_views(pools,8,random.Random(42))==sample_page_views(pools,8,random.Random(42))
    assert validation_sample(rows,0)==rows
    assert len(validation_sample(rows,2000))==2000
    with pytest.raises(ValueError):validation_sample(rows,-1)


def test_coverage_unique_pages_not_views_and_partial_views_not_success(tmp_path):
    from framediff.fixed_test import coverage_provenance
    labels=tmp_path/'labels';css=tmp_path/'css'
    write_json(labels/'prepare-report.json',{'test':dict(selected_pages=1,selected_views=2,
        source_pages=1,source_views=2,clean_pages=1,clean_views=2,kept=1,kept_views=2,rejected=0,rejected_views=0)})
    write_json(css/'config.json',{'data':{str(labels/'policy-test.jsonl'):'unused'}})
    write_json(css/'freeze-report.json',{'test':{'pages':1,'page_views':2}})
    rows=[dict(id=f'p/{v}/online-0',page_id='p') for v in ('desktop','mobile')]
    write_jsonl(tmp_path/'source/manifest-test.jsonl',[dict(id=f'p/{v}',page_id='p') for v in ('desktop','mobile')])
    results=[dict(method='abstract',page_id=f'p/{v}',logical_page_id='p',failed=False) for v in ('desktop','mobile')]
    report=coverage_provenance(css/'policy-test.jsonl',rows,results,['abstract'])
    assert report['methods']['abstract']['successful_pages']==1
    assert report['methods']['abstract']['end_to_end_success_rate']==1
    assert report['page_views']['evaluated']==2
    partial=coverage_provenance(css/'policy-test.jsonl',rows[:1],results[:1],['abstract'])
    assert partial['methods']['abstract']['successful_pages']==0


def lazy_descriptor(tmp_path):
    from framediff.visual_data import build
    from framediff.tree_normalize import _prepare_page
    html=tmp_path/'page.html'
    html.write_text('<html><body style="margin:0"><div style="width:100px;height:40px;background:red">Test</div></body></html>')
    manifest=tmp_path/'manifest.jsonl'
    write_jsonl(manifest,[dict(id='page/desktop',page_id='page',view='desktop',group='example',split='train',html=str(html),viewport=[400,200])])
    out=tmp_path/'rendered'
    args=SimpleNamespace(manifest=str(manifest),out=str(out),resume=True,trajectories=1,max_noise=1,
        seed=42,max_nodes=32,abstract_size=64,min_elements=1,max_source_mae=1.,clean_only=True,lazy_train=True)
    build(args)
    assert not list(out.rglob('*.png'))
    args=SimpleNamespace(resume=True,seed=42,max_css_owners=32,observation_size=64,max_pixel_mae=.01,max_box_error=1.)
    clean=list(read_jsonl(out/'policy-train.jsonl'))[0]
    page=list(read_jsonl(out/'pages-train.jsonl'))[0]
    row,_,error=_prepare_page((args,tmp_path/'labels','train',0,page['id'],clean,{page['id']:page}))
    assert error is None
    assert not list((tmp_path/'labels').rglob('*.png'))
    return row


def test_lazy_preparation_does_not_render_train(tmp_path):
    row=lazy_descriptor(tmp_path)
    assert row['lazy_source']['viewport']==[400,200]
    assert row['normalization_contract']


@pytest.mark.browser
def test_lazy_detector_and_policy_materialization(tmp_path):
    from framediff.lazy_webui import TrainViewCache,portable_images
    from framediff.tree_online import target_pool,OnlineSampler
    row=lazy_descriptor(tmp_path)
    cache=TrainViewCache(capacity=1)
    try:
        detector=cache.get(row)
        path=Path(detector['image'])
        assert path.is_file() and detector['elements']
        portable=portable_images(detector,('image',))
        assert isinstance(portable['image'],bytes)
        policy=cache.get(row,policy=True)
        assert policy['replacement_edit'] and policy['page_id']=='page'
        assert not path.exists() # eviction only removes this cache's generated files
        assert Path(row['lazy_source']['html']).is_file()
        assert cache.get(row,policy=True)==policy
    finally:cache.close()
    sampler=OnlineSampler(target_pool([row]),'screenshot',max_noise=1)
    try:
        result=sampler.generate(0)
        assert isinstance(result['target_image'],bytes)
        assert isinstance(result['current_image'],bytes)
        assert result['replacement_edit'] and result['page_id']=='page'
    finally:
        sampler.close();sampler.view_cache.close()


@pytest.mark.browser
def test_lazy_training_cache_stage2_and_resume(tmp_path):
    import torch
    from framediff.cli import main
    from framediff.lazy_webui import TrainViewCache
    from framediff.web_experiment import digest
    row=lazy_descriptor(tmp_path)
    # Independent held-out source, eagerly prepared at a different viewport.
    html=tmp_path/'heldout.html'
    html.write_text('<html><body style="margin:0"><div style="width:120px;height:50px;background:blue">Validation</div></body></html>')
    val_source={**row['lazy_source'],'id':'val/mobile','page_id':'val','group':'heldout',
                'view':'mobile','html':str(html),'source_sha':digest(html),'viewport':[220,300]}
    val_descriptor={**row,'lazy_source':val_source}
    cache=TrainViewCache()
    try:
        val_detection={**cache.get(val_descriptor),'split':'val'}
        val_policy={**cache.get(val_descriptor,policy=True),'split':'val'}
        write_jsonl(tmp_path/'detector-train.jsonl',[row])
        write_jsonl(tmp_path/'detector-val.jsonl',[val_detection])
        write_jsonl(tmp_path/'policy-train.jsonl',[row])
        write_jsonl(tmp_path/'policy-val.jsonl',[val_policy])
        main(['visual-train-detector','--train',str(tmp_path/'detector-train.jsonl'),
              '--val',str(tmp_path/'detector-val.jsonl'),'--out',str(tmp_path/'detector'),
              '--device','cpu','--cpu-threads','2','--steps','1','--batch-size','1','--accumulation','1',
              '--val-samples','0','--eval-every','1','--min-size','64','--max-size','96',
              '--max-detections','8','--no-pretrained'])
        for split in ('train','val'):
            main(['visual-cache-targets','--data',str(tmp_path/f'policy-{split}.jsonl'),
                  '--checkpoint',str(tmp_path/'detector/best.pt'),'--out',str(tmp_path/f'cached-{split}'),
                  '--device','cpu','--cpu-threads','2','--size','64'])
        assert not list((tmp_path/'cached-train').glob('*.png'))
        assert list(read_jsonl(tmp_path/'cached-train/data.jsonl'))[0]['lazy_detector']
        for mode,predicted in [('screenshot',False),('abstract',False),('abstract',True)]:
            out=tmp_path/f'run-{mode}-{predicted}'
            args=['visual-tree-train','--train',str(tmp_path/('cached-train/data.jsonl' if predicted else 'policy-train.jsonl')),
                  '--val',str(tmp_path/('cached-val/data.jsonl' if predicted else 'policy-val.jsonl')),
                  '--out',str(out),'--mode',mode,'--stylesheets','--online-corruption',
                  '--online-workers','1','--online-prefetch','1','--online-max-noise','1',
                  '--device','cpu','--no-pretrained','--size','32','--hidden','16','--layers','1',
                  '--heads','2','--token-grid','2','--max-nodes','32','--batch-size','1','--accumulation','1',
                  '--eval-every','1','--val-samples','0','--policy-metric-samples','0','--cpu-threads','2']
            if predicted:args.append('--predicted-targets')
            main([*args,'--steps','1'])
            checkpoint=out/'last.pt'
            main([*args,'--steps','2','--resume',str(checkpoint)])
            assert torch.load(checkpoint,weights_only=True)['online_cursor']==2
            assert list(read_jsonl(out/'validation.jsonl'))[-1]['validation_complete']
    finally:cache.close()


@pytest.mark.browser
def test_multiview_prepare_pipeline_resume_and_subset(tmp_path):
    from PIL import Image
    from framediff.cli import main
    from framediff.ir import read_json
    raw=tmp_path/'raw';out=tmp_path/'scale'
    for i in range(12):
        page=raw/str(i);page.mkdir(parents=True)
        for view,viewport in [('default_1280-720',(1280,720)),('iPhone-13 Pro',(390,844))]:
            Image.new('RGB',viewport,'white').save(page/f'{view}-screenshot.png')
            (page/f'{view}-html.html').write_text(f'<html><body style="margin:0"><div style="width:100px;height:40px;background:red">Page {i}</div></body></html>')
            (page/f'{view}-url.txt').write_text(f'https://site{i}.example/')
    main(['visual-webui-scales-prepare','--root',str(raw),'--out',str(out),'--sizes','10',
          '--views','default_1280-720','iPhone-13 Pro','--resume'])
    master=out/'master'
    commands=[['visual-build-data','--manifest',str(master/'source/manifest.jsonl'),
               '--out',str(master/'rendered'),'--clean-only','--lazy-train',
               '--max-nodes','32','--abstract-size','64','--workers','2','--resume'],
              ['visual-tree-normalize','--rendered',str(master/'rendered'),'--out',str(master/'labels'),
               '--observation-size','64','--max-css-owners','32','--workers','2','--resume'],
              ['visual-tree-freeze','--rendered',str(master/'labels'),'--out',str(master/'css'),
               '--observation-size','64','--max-noise','1','--workers','2','--resume'],
              ['visual-webui-scales-subset','--data',str(out)]]
    for command in commands:main(command)
    before=(out/'10/css/policy-test.jsonl').read_bytes()
    for command in commands:main(command)
    assert before==(out/'10/css/policy-test.jsonl').read_bytes()
    train=list(read_jsonl(out/'10/css/policy-train.jsonl'))
    test=list(read_jsonl(out/'10/css/policy-test.jsonl'))
    assert len(train)==16 and all(r.get('lazy_contract') for r in train)
    assert len(test)==2 and all(not r.get('lazy_contract') for r in test)
    assert len({r['page_id'] for r in train})==8
    assert {r['view'] for r in test}=={'default_1280-720','iPhone-13 Pro'}
    assert all(Path(r['current_image']).is_file() for r in test)
    assert read_json(out/'10/coverage.json')['rendering']['train']['status']=='deferred'

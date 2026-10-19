import io
import random
import pytest
import torch
from pathlib import Path
from PIL import Image

from framediff.detector_support import TargetPredictionCache


@pytest.mark.parametrize('low_quality',[False,True])
@pytest.mark.parametrize('case',['random','ties','zeros','thresholds'])
def test_block_matching_exactly_matches_torchvision(low_quality,case):
    from torchvision.models.detection._utils import Matcher
    from framediff.detector_support import bounded_box_matches
    rng=torch.Generator().manual_seed(18)
    scores=torch.rand(17,43,generator=rng)
    if case=='ties':scores=(scores*3).floor()/3
    if case=='zeros':scores.zero_()
    if case=='thresholds':scores=(scores*3).floor()*.3
    matcher=Matcher(.6,.3,allow_low_quality_matches=low_quality)
    sizes=[]
    def similarity(gt,anchors):
        sizes.append((len(gt),len(anchors)))
        return scores[gt[:,0].long()][:,anchors[:,0].long()]
    actual=bounded_box_matches(torch.arange(17.).reshape(-1,1),torch.arange(43.).reshape(-1,1),
                               matcher,similarity,gt_chunk=4,anchor_chunk=7)
    assert torch.equal(actual,matcher(scores))
    assert max(g for g,a in sizes)<=4 and max(a for g,a in sizes)<=7


def test_bounded_rpn_preserves_checkpoint_losses_and_gradients():
    from framediff.visual_train import detector
    from framediff.detector_support import BoundedRegionProposalNetwork
    from torchvision.models.detection.rpn import RegionProposalNetwork
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        net=detector({'min_size':32,'max_size':64,'max_detections':10},False).train()
        rpn=net.rpn
        assert isinstance(rpn,BoundedRegionProposalNetwork)
        baseline=RegionProposalNetwork(rpn.anchor_generator,rpn.head,.7,.3,256,.5,
                                      rpn._pre_nms_top_n,rpn._post_nms_top_n,rpn.nms_thresh,rpn.score_thresh)
        assert baseline.load_state_dict(rpn.state_dict(),strict=True).missing_keys==[]
        boxes=[torch.tensor([[0.,0.,10.,10.],[20.,20.,25.,25.]])]
        for gt in (torch.empty(0,4),torch.tensor([[1.,1.,9.,9.]])):
            left=baseline.assign_targets_to_anchors(boxes,[{'boxes':gt}])
            right=rpn.assign_targets_to_anchors(boxes,[{'boxes':gt}])
            for a,b in zip(left,right):assert torch.equal(a[0],b[0])
        image=torch.rand(3,32,48)
        target={'boxes':torch.tensor([[1.,1.,14.,16.]]),'labels':torch.tensor([1])}
        torch.manual_seed(42);losses=net([image],[target]);sum(losses.values()).backward()
        grad=rpn.head.cls_logits.weight.grad.clone()
        net.zero_grad(set_to_none=True);net.rpn=baseline
        torch.manual_seed(42);expected=net([image],[target]);sum(expected.values()).backward()
        for key in expected:torch.testing.assert_close(losses[key],expected[key],rtol=0,atol=0)
        torch.testing.assert_close(grad,baseline.head.cls_logits.weight.grad,rtol=0,atol=0)
    finally:torch.set_num_threads(old_threads)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for peak memory comparison')
def test_bounded_matching_cuda_peak_memory():
    from torchvision.ops import box_iou
    from torchvision.models.detection._utils import Matcher
    from framediff.detector_support import bounded_box_matches
    generator=torch.Generator(device='cuda').manual_seed(21)
    def boxes(n):
        xy=torch.rand(n,2,generator=generator,device='cuda')*1000
        return torch.cat((xy,xy+torch.rand(n,2,generator=generator,device='cuda')*100+1),dim=1)
    gt=boxes(512);anchors=boxes(16384);matcher=Matcher(.7,.3,True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated()
    expected=matcher(box_iou(gt,anchors))
    torch.cuda.synchronize();dense_peak=torch.cuda.max_memory_allocated()-baseline
    torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
    actual=bounded_box_matches(gt,anchors,matcher,box_iou)
    torch.cuda.synchronize();chunk_peak=torch.cuda.max_memory_allocated()-baseline
    assert torch.equal(actual,expected)
    assert chunk_peak<dense_peak
    print({'dense_peak_bytes':dense_peak,'chunk_peak_bytes':chunk_peak})


def test_detector_resume_microbatch_migration(tmp_path):
    import copy
    import json
    from framediff.visual_train import guard_detector_resume
    old={'kind':'visual-detector-v1','config':{'model':'same'},'data':{'train':'same'},
         'settings':{'batch_size':2,'accumulation':4,'lr':.0001}}
    path=tmp_path/'config.json';path.write_text(json.dumps(old))
    new=copy.deepcopy(old);new['settings'].update(batch_size=1,accumulation=8)
    assert guard_detector_resume(tmp_path,new,True)==2
    assert guard_detector_resume(tmp_path,new,True)==2
    saved=path.read_text()
    for change in ({'accumulation':4},{'lr':.01}):
        bad=copy.deepcopy(new);bad['settings'].update(change)
        with pytest.raises(ValueError):guard_detector_resume(tmp_path,bad,True)
        assert path.read_text()==saved
    bad=copy.deepcopy(new);bad['data']['train']='changed'
    with pytest.raises(ValueError):guard_detector_resume(tmp_path,bad,True)
    fresh=tmp_path/'fresh'
    assert guard_detector_resume(fresh,new,False)==1


def detector_rows(tmp_path):
    rows=[]
    for i in range(5):
        path=tmp_path/f'{i}.png';Image.new('RGB',(13+i,9+i),(i*40,70,120)).save(path)
        rows.append({'id':f'p{i}/view','page_id':f'p{i}','image':str(path),
                     'elements':[] if i==0 else [{'box':[1,2,6,7],'label':1}]})
    return rows


@pytest.mark.browser
def test_detector_relabel_script_exports_independent_assets_and_resumes(tmp_path):
    import subprocess
    import sys
    import json
    from framediff.html_bridge import HtmlBrowser
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.visual import DETECTOR_LABEL_CONTRACT
    html=tmp_path/'sample.html';png=tmp_path/'sample.png'
    html.write_text('<html><body style="margin:0">Visible text<button>OK</button></body></html>')
    with HtmlBrowser() as browser:
        browser.load(html.read_text(),[320,200]);pixels,_=browser.stable_capture()
        png.write_bytes(pixels)
    source=tmp_path/'old';out=tmp_path/'new'
    for split in ('train','val','test'):
        folder=source/'10/rendered';folder.mkdir(parents=True,exist_ok=True)
        write_jsonl(folder/f'detector-{split}.jsonl',[dict(id=split,group=split,split=split,
                     source_sha=split,image=str(png),viewport=[320,200],elements=[])])
    train_path = source/'10/rendered/detector-train.jsonl'
    write_jsonl(train_path, list(read_jsonl(train_path)) + [dict(id='broken', image=str(tmp_path/'missing.png'))])
    script=Path(__file__).resolve().parents[1]/'scripts/relabel_detector_data.py'
    command=[sys.executable,str(script),'--data',str(source),'--out',str(out),'--sizes','10','--workers','1']
    subprocess.run(command,check=True,capture_output=True,text=True)
    assert (out/'READY').is_file()
    assert json.loads((out/'summary.json').read_text())['failed'] == 1
    assert list(read_jsonl(out/'rejections.jsonl'))[0]['id'] == 'broken'
    for split in ('train','val','test'):
        row=list(read_jsonl(out/f'10/rendered/detector-{split}.jsonl'))[0]
        assert row['detector_label_contract']==DETECTOR_LABEL_CONTRACT and len(row['elements'])==2
        assert Path(row['image']).is_relative_to(out) and Path(row['audit_html']).is_relative_to(out)
        assert row['reused_pixels']
    before=(out/'10/rendered/detector-train.jsonl').read_bytes()
    legacy = json.loads((out/'relabel-config.json').read_text())
    legacy['code']['script'] = '5576bcf7f09aaf9ef4980ecf9f59785c7548098eb21760ac278f6a83c772de39'
    (out/'relabel-config.json').write_text(json.dumps(legacy))
    png.unlink();html.unlink()
    subprocess.run(command,check=True,capture_output=True,text=True)
    assert (out/'10/rendered/detector-train.jsonl').read_bytes()==before
    assert json.loads((out/'relabel-config.json').read_text())['label_contract']==DETECTOR_LABEL_CONTRACT
    (tmp_path/'missing.html').write_text('<html><body>Recovered</body></html>')
    Image.new('RGB',(320,200),'white').save(tmp_path/'missing.png')
    subprocess.run(command,check=True,capture_output=True,text=True)
    assert json.loads((out/'summary.json').read_text())['failed'] == 0
    assert len(list(read_jsonl(out/'10/rendered/detector-train.jsonl'))) == 2
    assert (out/'rejections.jsonl').read_text() == ''


@pytest.mark.browser
def test_detector_relabel_lazy_single_element(tmp_path):
    import importlib.util
    import hashlib
    from framediff.lazy_webui import CONTRACT
    script = Path(__file__).resolve().parents[1]/'scripts/relabel_detector_data.py'
    spec = importlib.util.spec_from_file_location('relabel', script)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    html = tmp_path/'one.html'
    html.write_text('<html><body><p>One text</p></body></html>')
    row = {'id': 'one/clean', 'lazy_contract': CONTRACT,
           'lazy_source': {'id': 'one', 'group': 'one', 'split': 'train', 'html': str(html),
                           'source_sha': hashlib.sha256(html.read_bytes()).hexdigest(), 'viewport': [320,200]},
           'lazy_settings': {'seed':42, 'max_nodes':512, 'abstract_size':384, 'min_elements':3,
                             'max_source_mae':1., 'resume':True, 'clean_only':True, 'lazy_train':False}}
    result, error = module.process_checked((row, str(tmp_path/'out')))
    assert error is None, error
    assert len(result['elements']) == 1
    assert row['lazy_settings']['min_elements'] == 3
    current = {'inputs': {}, 'code': {'script':'new', 'visual':'same'}}
    assert not module.compatible_config({'inputs': {}, 'code': {'script':'unknown', 'visual':'same'}}, current)


def test_prefetched_detector_pixels_targets_loss_and_gradients_match(tmp_path):
    from framediff.detector_support import prepare_detector_batch,detector_batches
    from framediff.visual_train import detector_input,detection_losses
    rows=detector_rows(tmp_path)
    prepared=prepare_detector_batch(rows)
    for old,new in zip(rows,prepared):
        assert torch.equal(new['_detector_pixels'].float()/255,detector_input(old['image']))
        assert new['_detector_pixels'].dtype==torch.uint8
        assert '_detector_pixels' not in old
        assert new['_detector_target']['boxes'].shape==(len(old['elements']),4)
    class Net(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.))
        def forward(self,images,targets):
            return {'loss':sum(x.mean()*self.weight+y['boxes'].sum()*self.weight for x,y in zip(images,targets))}
    net=Net();a=detection_losses(net,rows,'cpu')['loss'];a.backward();grad=net.weight.grad.clone()
    net.zero_grad();b=detection_losses(net,prepared,'cpu')['loss'];b.backward()
    assert torch.equal(a,b) and torch.equal(grad,net.weight.grad)
    batches=list(detector_batches(rows,2,workers=2,prefetch=2))
    assert [len(b) for b in batches]==[2,2,1]
    assert [r['id'] for b in batches for r in b]==[r['id'] for r in rows]
    assert list(detector_batches(rows,2,workers=0))==[rows[:2],rows[2:4],rows[4:]]


def test_detector_loader_preserves_sampling_and_resume_rng(tmp_path):
    from framediff.detector_support import DetectorLoader
    from framediff.visual_train import page_view_pools,sample_page_views
    rows=detector_rows(tmp_path);pools=page_view_pools(rows)
    rng=random.Random(42);reference=random.Random(42)
    loader=DetectorLoader(pools,rng,2,workers=2,prefetch=3)
    try:
        for _ in range(3):
            batch=loader.take();expected=sample_page_views(pools,2,reference)
            assert [r['id'] for r in batch]==[r['id'] for r in expected]
            assert rng.getstate()==reference.getstate()
        saved=rng.getstate();expected=loader.take()
    finally:loader.close()
    resumed=random.Random();resumed.setstate(saved)
    loader=DetectorLoader(pools,resumed,2,workers=1,prefetch=1)
    try:assert [r['id'] for r in loader.take()]==[r['id'] for r in expected]
    finally:loader.close()
    assert resumed.getstate()==rng.getstate()


def test_detector_prefetch_propagates_decode_failure(tmp_path):
    from framediff.detector_support import detector_batches
    rows=detector_rows(tmp_path);rows[1]['image']=str(tmp_path/'missing.png')
    with pytest.raises(FileNotFoundError):list(detector_batches(rows,1,workers=2,prefetch=2))


def test_detector_render_worker_decodes_and_stops_with_pending_batches(tmp_path):
    from framediff.detector_support import DetectorStream
    from framediff.visual_train import page_view_pools,detector_input
    pools=page_view_pools(detector_rows(tmp_path))
    with DetectorStream(pools,random.Random(42),2,tmp_path/'cache',tmp_path/'errors',
                        workers=1,prefetch=3,decode=True) as stream:
        rows=stream.take();processes=list(stream.pool._processes.values())
        for row in rows:
            assert torch.equal(row['_detector_pixels'].float()/255,detector_input(row['image']))
    assert all(not process.is_alive() for process in processes)


def test_detector_training_prefetch_and_resume_preserve_checkpoint(tmp_path,monkeypatch):
    from framediff.cli import main
    from framediff.ir import write_jsonl
    from framediff.visual import CONTRACT
    import framediff.visual_train as train
    class TinyDetector(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.randn(()))
        def forward(self,images,targets=None):
            if self.training:
                return {'loss_classifier':sum((image.mean()*self.weight-torch.rand(())).square() for image in images)}
            return [{'boxes':image.new_tensor([[1,2,6,7]]),'labels':torch.tensor([1]),
                     'scores':image.new_tensor([.9])} for image in images]
    monkeypatch.setattr(train,'detector',lambda *a,**kw:TinyDetector())
    rows=detector_rows(tmp_path)
    for split in ('train','val'):
        write_jsonl(tmp_path/f'{split}.jsonl',[{**r,'id':split+'/'+r['id'],'page_id':split+'/'+r['page_id'],
                                               'split':split,'group':split,'source_sha':split,
                                               'contract':CONTRACT} for r in rows])
    def run(name,workers,steps,resume=False,batch_size=2,accumulation=2):
        out=tmp_path/name
        args=['visual-train-detector','--train',str(tmp_path/'train.jsonl'),'--val',str(tmp_path/'val.jsonl'),
              '--out',str(out),'--device','cpu','--no-pretrained','--cpu-threads','1','--steps',str(steps),
              '--batch-size',str(batch_size),'--accumulation',str(accumulation),'--eval-every','1','--val-samples','3',
              '--loader-workers',str(workers),'--loader-prefetch','2']
        if resume:args+=['--resume',str(out/'last.pt')]
        main(args)
        return torch.load(out/'last.pt',weights_only=True)
    direct=run('direct',0,2);prefetched=run('prefetched',2,2)
    run('resumed',2,1);resumed=run('resumed',1,2,True)
    for checkpoint in (prefetched,resumed):
        assert checkpoint['rng']==direct['rng']
        assert torch.equal(checkpoint['torch_rng'],direct['torch_rng'])
        assert checkpoint['best']==direct['best']
        for key,value in direct['model'].items():assert torch.equal(checkpoint['model'][key],value)
        for key,value in direct['optimizer']['state'][0].items():
            assert torch.equal(checkpoint['optimizer']['state'][0][key],value)
    before=run('microbatch',2,1,batch_size=2,accumulation=4)
    after=run('microbatch',2,2,True,batch_size=1,accumulation=8)
    assert before['step']==1 and after['step']==2
    assert after['optimizer']['state'][0]['step'].item()==2
    assert torch.isfinite(after['model']['weight'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for pinned transfer')
def test_detector_pinned_transfer_matches_cpu(tmp_path):
    from framediff.detector_support import prepare_detector_batch
    from framediff.visual_train import detector_input
    rows=detector_rows(tmp_path)
    for row in prepare_detector_batch(rows,pin=True):
        pixels=row['_detector_pixels'];assert pixels.is_pinned()
        torch.testing.assert_close(pixels.to('cuda',non_blocking=True).float().div(255).cpu(),
                                   detector_input(row['image']),rtol=1e-6,atol=1e-7)


def test_shared_prediction_cache_reuses_legacy_and_is_bounded(tmp_path,monkeypatch):
    old=TargetPredictionCache(tmp_path/'old','sha',.4)
    items=[dict(box=[0,0,1,1],label=4,score=.9)]
    import json
    (old.folder/'key.json').write_text(json.dumps({'key':'key','items':items}))
    monkeypatch.setenv('TUIDE_SHARED_CACHE',str(tmp_path/'shared'))
    cache=TargetPredictionCache(old.folder,'sha',.4,capacity=2)
    assert cache.read('key')==items
    assert (cache.folder/'key.json').stat().st_ino==(old.folder/'key.json').stat().st_ino
    other=TargetPredictionCache(tmp_path/'other','sha',.4)
    assert other.read('key')==items
    for i in range(1000):cache.remember(str(i),items)
    assert len(cache.memory)==2
    assert len(list(cache.folder.glob('*.json')))==1


def test_loader_close_releases_prefetched_batches():
    from framediff.detector_support import DetectorLoader
    from concurrent.futures import Future
    loader=DetectorLoader({},random.Random(0),1)
    future=Future();future.set_result(bytearray(1000000))
    loader.pending.append((future,None))
    loader.close()
    assert not loader.pending


def test_prediction_cache_batches_unique_misses_and_survives_restart(tmp_path,monkeypatch):
    import framediff.visual_train as train
    calls=[]
    def detect(net,images,threshold):
        calls.append(len(images))
        return [[dict(box=[0,0,1,1],label=4,score=.9)] for _ in images]
    monkeypatch.setattr(train,'detect_batch',detect)
    images=[]
    for color in ('red','blue'):
        buf=io.BytesIO();Image.new('RGB',(5,5),color).save(buf,format='PNG');images.append(buf.getvalue())
    cache=TargetPredictionCache(tmp_path,'checkpoint-a',.4)
    first=cache.predict(None,[images[0],images[1],images[0]])
    assert calls==[2] and first[0]==first[2]
    cache=TargetPredictionCache(tmp_path,'checkpoint-a',.4)
    assert cache.predict(None,images)==first[:2] and calls==[2]
    other=TargetPredictionCache(tmp_path,'checkpoint-b',.4)
    other.predict(None,images)
    assert calls==[2,2]
    for path in tmp_path.glob('*.json'):path.write_text('{bad')
    cache=TargetPredictionCache(tmp_path,'checkpoint-a',.4)
    assert cache.predict(None,images)==first[:2] and calls==[2,2,2]

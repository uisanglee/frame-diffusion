import io
import random
import pytest
import torch
from PIL import Image

from framediff.detector_support import TargetPredictionCache


def detector_rows(tmp_path):
    rows=[]
    for i in range(5):
        path=tmp_path/f'{i}.png';Image.new('RGB',(13+i,9+i),(i*40,70,120)).save(path)
        rows.append({'id':f'p{i}/view','page_id':f'p{i}','image':str(path),
                     'elements':[] if i==0 else [{'box':[1,2,6,7],'label':1}]})
    return rows


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
    def run(name,workers,steps,resume=False):
        out=tmp_path/name
        args=['visual-train-detector','--train',str(tmp_path/'train.jsonl'),'--val',str(tmp_path/'val.jsonl'),
              '--out',str(out),'--device','cpu','--no-pretrained','--cpu-threads','1','--steps',str(steps),
              '--batch-size','2','--accumulation','2','--eval-every','1','--val-samples','3',
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


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for pinned transfer')
def test_detector_pinned_transfer_matches_cpu(tmp_path):
    from framediff.detector_support import prepare_detector_batch
    from framediff.visual_train import detector_input
    rows=detector_rows(tmp_path)
    for row in prepare_detector_batch(rows,pin=True):
        pixels=row['_detector_pixels'];assert pixels.is_pinned()
        torch.testing.assert_close(pixels.to('cuda',non_blocking=True).float().div(255).cpu(),
                                   detector_input(row['image']),rtol=1e-6,atol=1e-7)


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

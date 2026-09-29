import copy
import random
import pytest
import torch
from framediff.ir import apply_edit, execute, validate
from framediff.data import synthetic,corrupt,differences,make_record,assert_disjoint
from framediff.model import ModelConfig,EditDenoiser,collate
from framediff.metrics import box_metrics
from framediff.train import loss_for,sample
from framediff.search import repair

def test_roundtrip_valid_edits_and_recovery():
    clean=synthetic(1);current=corrupt(clean,random.Random(1),5)
    assert execute(current)!=execute(clean)
    for edit in differences(current,clean):current=apply_edit(current,edit)
    assert execute(current)==execute(clean)

def test_cycle_and_illegal_edits():
    clean=synthetic(1);bad=copy.deepcopy(clean);bad['nodes'][1]['parent']=bad['nodes'][1]['id']
    with pytest.raises(ValueError):validate(bad)
    with pytest.raises(ValueError):apply_edit(clean,(0,'width',3))
    with pytest.raises(ValueError):apply_edit(clean,(1,'width',999))

def test_loss_backward_padding_no_leak():
    torch.set_num_threads(2)
    cfg=ModelConfig(32,1,4,0)
    model=EditDenoiser(cfg)
    records=[make_record(synthetic(i),str(i),str(i),i) for i in (1,2)]
    samples=[sample(r,random.Random(i),cfg,3,clean_probability=0,online=False) for i,r in enumerate(records)]
    loss,_=loss_for(model,samples,'cpu','edits');assert torch.isfinite(loss);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    # Encoding uses current + observations only; target source props never enter model.
    batch=collate([s[0] for s in samples]);out=model(batch)
    assert torch.isneginf(out['logits'][~batch['legal']]).all()

def test_coordinate_repair_improves_known_offset():
    clean=synthetic(2);current=apply_edit(clean,(1,'dx',36))
    obs=[{'viewport':[1024,768],'target':execute(clean)}]
    result,trace=repair(current,obs,'coordinate',steps=3,beam=1,budget=400)
    assert trace['history'][-1]['score']<trace['history'][0]['score']
    assert box_metrics(execute(result),obs[0]['target'],(1024,768))['box_iou']>.99

def test_group_leakage_and_missing_metric():
    with pytest.raises(ValueError):assert_disjoint([{'id':'a','group':'site'}],[{'id':'b','group':'site'}])
    m=box_metrics({}, {'x':[0,0,100,100]},(1024,768))
    assert m['box_iou']==0 and m['recall']==0

@pytest.mark.browser
def test_proxy_browser_parity():
    from framediff.browser import Browser
    with Browser() as browser:
        for seed in range(3):
            tree=corrupt(synthetic(seed),random.Random(seed+1),5)
            for viewport in [(1024,768),(390,844)]:
                expected=execute(tree,viewport);actual=browser.render(tree,viewport)
                error=max(abs(actual[k][j]-expected[k][j]) for k in expected for j in range(4))
                assert error<.2,(seed,viewport,error)

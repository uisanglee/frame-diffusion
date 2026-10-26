"""Stopping is a stage boundary; resume must not restart a finished stage."""
import torch
import pytest

from framediff.cli import main
from framediff.ir import write_jsonl,read_jsonl
from framediff.visual import CONTRACT
from framediff.visual_train import update_early_stopping


def test_patience_delta_and_disabled():
    state={'reference_loss':1.,'bad_validations':0,'stopped':False}
    state=update_early_stopping(state,.95,2,.1)
    assert state['bad_validations']==1
    state=update_early_stopping(state,.89,2,.1)
    assert state['bad_validations']==0 and state['reference_loss']==.89
    state=update_early_stopping(state,.89,2,.1)
    assert update_early_stopping(state,.89,2,.1)['stopped']
    assert not update_early_stopping(state,1.,0,0.)['stopped']


def test_detector_early_stop_resume_best_and_next_stage(tmp_path,monkeypatch):
    import framediff.visual_train as training
    from PIL import Image
    class TinyDetector(torch.nn.Module):
        def __init__(self,*args):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
    monkeypatch.setattr(training,'detector',TinyDetector)
    validations=iter([1.,.9,.91,.92])
    monkeypatch.setattr(training,'detection_loss',lambda net,*args:net.weight.square().mean())
    monkeypatch.setattr(training,'detection_losses',lambda *args:{'classifier':torch.tensor(next(validations))})
    monkeypatch.setattr(training,'detector_quality',lambda *args:{})
    image=tmp_path/'image.png';Image.new('RGB',(32,32),'white').save(image)
    for split in ('train','val'):
        write_jsonl(tmp_path/f'{split}.jsonl',[{'split':split,'group':split,'source_sha':split,
                                             'contract':CONTRACT,'image':str(image),'elements':[]}])
    common=['visual-train-detector','--train',str(tmp_path/'train.jsonl'),'--val',str(tmp_path/'val.jsonl'),
            '--device','cpu','--batch-size','1','--accumulation','1','--loader-workers','0',
            '--eval-every','1','--log-every','1','--early-stop-patience','2','--no-pretrained']
    out=tmp_path/'stage1'
    main([*common,'--out',str(out),'--steps','3'])
    def last():return torch.load(out/'last.pt',weights_only=True)
    assert last()['early_stopping']['bad_validations']==1
    main([*common,'--out',str(out),'--steps','10','--resume',str(out/'last.pt')])
    assert last()['step']==4 and last()['early_stopping']['stopped']
    best=torch.load(out/'best.pt',weights_only=True)
    assert best['step']==2 and best['best']==pytest.approx(.9)
    logs=list(read_jsonl(out/'train.jsonl'))
    # A terminal resume must not instantiate a model or consume another validation.
    def unexpected(*args):raise AssertionError('Completed stage must be skipped')
    monkeypatch.setattr(training,'detector',unexpected)
    main([*common,'--out',str(out),'--steps','20','--resume',str(out/'last.pt')])
    assert list(read_jsonl(out/'train.jsonl'))==logs
    monkeypatch.setattr(training,'detector',TinyDetector)
    validations=iter([1.1,1.2,1.3])
    next_out=tmp_path/'stage2'
    main([*common,'--out',str(next_out),'--steps','10','--init-checkpoint',str(out/'best.pt')])
    next_ck=torch.load(next_out/'last.pt',weights_only=True)
    assert next_ck['step']==3 and next_ck['early_stopping']['stopped']
    assert torch.load(next_out/'best.pt',weights_only=True)['step']==1


@pytest.mark.parametrize('flag,value',[('--early-stop-patience','-1'),('--early-stop-min-delta','-1'),
                                     ('--early-stop-min-delta','nan')])
def test_invalid_early_stop_settings_rejected_before_data(flag,value):
    with pytest.raises(ValueError,match='nonnegative'):
        main(['visual-train-detector','--train','missing','--val','missing','--out','unused',flag,value])

from types import SimpleNamespace

import pytest
import torch

from framediff.ir import write_json, write_jsonl, read_jsonl
from framediff.scale_report import build


def test_scale_report_shares_baselines_and_rejects_different_trials(tmp_path):
    for scale in ('s','m','l'):
        path=tmp_path/scale
        (path/'evaluation').mkdir(parents=True);(path/'repair').mkdir()
        write_jsonl(path/'evaluation/metrics.jsonl',[
            {'id':'page/repeat-1','method':method,'failed':False,'dom_box_iou':.5}
            for method in ('initial','screenshot-policy','abstract-policy')])
        write_json(path/'repair/timing-summary.json',{'model_parameters':{'abstract-policy':10}})
    build(tmp_path)
    rows=list(read_jsonl(tmp_path/'comparison/metrics.jsonl'))
    assert [r['method'] for r in rows].count('initial')==1
    assert {r['method'] for r in rows}=={'initial','screenshot-policy','TUIDE-S','TUIDE-M','TUIDE-L'}
    write_jsonl(tmp_path/'l/evaluation/metrics.jsonl',[{'id':'different','method':'abstract-policy'}])
    with pytest.raises(ValueError,match='identical'):build(tmp_path)


def test_abstract_only_evaluation_does_not_load_rgb(tmp_path,monkeypatch):
    from framediff import visual_experiment as experiment
    from framediff.cli import main
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.cfg=SimpleNamespace(mode='abstract')
    loaded=[]
    def load(path,device):
        loaded.append(path);return Policy(),{}
    monkeypatch.setattr(experiment,'load_policy',load)
    monkeypatch.setattr(experiment,'load_detector',lambda *a:(None,{}))
    class Browser:
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(experiment,'HtmlBrowser',Browser)
    html=tmp_path/'initial.html';html.write_text('<html></html>')
    for name in ('policy.pt','detector.pt'):(tmp_path/name).write_text('fixture')
    data=tmp_path/'prepared.jsonl'
    write_jsonl(data,[{'id':'page','group':'test','methods':{'initial':{
        'html':str(html),'failed':True,'error':'initial failed','seconds':0,'vlm_calls':0}}}])
    main(['visual-evaluate','--data',str(data),'--abstract-only',
          '--abstract-checkpoint',str(tmp_path/'policy.pt'),'--detector-checkpoint',str(tmp_path/'detector.pt'),
          '--out',str(tmp_path/'out'),'--device','cpu','--warmup','0','--repeats','1'])
    assert loaded==[str(tmp_path/'policy.pt')]
    methods=list(read_jsonl(tmp_path/'out/results.jsonl'))[0]['methods']
    assert methods['abstract-policy']['failed'] and 'screenshot-policy' not in methods

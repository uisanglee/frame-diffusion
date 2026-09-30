from types import SimpleNamespace

import pytest
import torch

from framediff.data import make_record, synthetic
from framediff.ir import read_json, write_jsonl
from framediff.model import EditDenoiser, ModelConfig
from framediff.render_compare import compare_renderers, summarize, truth_for


def test_paired_loss_sign_and_repeats():
    rows=[]
    for repeat in range(3):
        for backend,t,iou,calls in [('proxy',1,.8,0),('browser',4,.9,12)]:
            rows.append(dict(id='a',group='site',repeat=repeat,backend=backend,repair_seconds=t,
                             pipeline_seconds=t+9,browser_iou=iou,search_browser_calls=calls))
    pairs,summary=summarize(rows,42)
    assert len(pairs)==1 and summary[0]['pages']==1
    assert summary[0]['repair_speedup']==4
    assert summary[0]['pipeline_speedup']==1.3
    assert summary[0]['iou_loss_pp']==pytest.approx(10)
    assert summary[0]['browser_calls_saved']==12


def test_independent_truth_required_and_legacy_webui():
    with pytest.raises(ValueError,match='independent'):
        truth_for({'target_kind':'predicted_frames','observations':[]})
    r={'observations':[{'viewport':[100,100],'target':{'a':[0,0,5,5]}}],
       'original_target':{'a':[0,0,6,6]}}
    assert truth_for(r)[0]['target']==r['original_target']


@pytest.mark.browser
def test_paired_browser_smoke(tmp_path):
    from dataclasses import asdict
    cfg=ModelConfig(hidden=16,layers=1,heads=2)
    model=EditDenoiser(cfg)
    checkpoint=tmp_path/'model.pt'
    torch.save({'config':asdict(cfg),'model':model.state_dict(),'objective':'edits','training_groups':[]},checkpoint)
    record=make_record(synthetic(4),'a','site',1,viewports=[[320,240]])
    write_jsonl(tmp_path/'data.jsonl',[record])
    args=SimpleNamespace(data=str(tmp_path/'data.jsonl'),checkpoint=str(checkpoint),out=str(tmp_path/'out'),
                         device='cpu',cpu_threads=1,limit=1,seed=42,steps=1,beam=1,topk=1,budget=1,
                         repeats=1,warmup=0,raster=True)
    compare_renderers(args)
    summary=read_json(tmp_path/'out/summary.json')
    assert len(summary)==2
    assert all(s['browser_calls_saved']>=1 for s in summary)
    assert read_json(tmp_path/'out/manifest.json')['sample_ids']==['a']

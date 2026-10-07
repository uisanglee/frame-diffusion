from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from framediff.ir import node,read_jsonl,write_jsonl


def sample(tmp_path):
    target=tmp_path/'target.png';Image.new('RGB',(100,80),'white').save(target)
    initial=tmp_path/'initial.html';initial.write_text('<html><body><div>A</div></body></html>')
    reference=tmp_path/'reference.html';reference.write_text('<html><body><div>A</div></body></html>')
    tree={'version':1,'nodes':[node('fd-0',None,'page'),node('fd-1','fd-0','text')]}
    tree['nodes'][1]['visual_class']=1
    return {'id':'page','group':'page','screenshot':str(target),'html':str(reference),'viewport':[100,80],
            'current':tree,'original_boxes':{'fd-0':[0,0,100,80],'fd-1':[10,10,30,20]},
            'methods':{'initial':{'html':str(initial),'seconds':2.,'vlm_calls':1,'failed':False}}}


def test_cache_and_abstract_revision_are_separate_stages(tmp_path,monkeypatch):
    import framediff.abstract_vlm_revision as module
    data=tmp_path/'prepared.jsonl';write_jsonl(data,[sample(tmp_path)])
    checkpoint=tmp_path/'detector.pt';checkpoint.write_bytes(b'checkpoint')
    monkeypatch.setattr(module,'load_detector',lambda *a:(object(),{}))
    monkeypatch.setattr(module,'detector_input',lambda *a:object())
    monkeypatch.setattr(module,'detect',lambda *a:[{'box':[8,8,45,35],'label':1,'score':.9}])
    cached=module.cache_abstractions(SimpleNamespace(data=str(data),detector_checkpoint=str(checkpoint),
        out=str(tmp_path/'cache'),device='cpu',threshold=.4,size=64,resume=False))
    assert Path(cached[0]['abstract_target']).is_file() and Path(cached[0]['abstract_current']).is_file()
    class Browser:
        executions=0
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def snapshot(self,*a,**kw):self.executions+=1;return {}
    monkeypatch.setattr(module,'HtmlBrowser',Browser)
    monkeypatch.setattr(module.vlm,'load_qwen_runtime',lambda *a:object())
    monkeypatch.setattr(module.vlm,'images_for',lambda *a:[])
    monkeypatch.setattr(module.vlm,'generate',lambda *a:('<html><body><div>A</div></body></html>',
        {'input_tokens':120,'output_tokens':40,'generation_seconds':.1}))
    revised=module.revise(SimpleNamespace(data=str(tmp_path/'cache/data.jsonl'),out=str(tmp_path/'revision'),
        backend='qwen',model='test',revision='main',endpoint='',api_key_env='KEY',four_bit=False,resume=False,
        max_new_tokens=100,max_pixels=10000,seed=1,vlm_retries=0))
    method=revised[0]['methods']['abstract-vlm-self-revision']
    assert method['repair_vlm_calls']==1 and method['repair_input_tokens']==120 and method['repair_output_tokens']==40
    assert list(read_jsonl(tmp_path/'revision/prepared.jsonl'))[0]['methods']['abstract-vlm-self-revision']['failed'] is False

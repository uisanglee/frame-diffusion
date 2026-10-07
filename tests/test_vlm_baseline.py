import json
from PIL import Image

from framediff import vlm
from framediff.cli import main
from framediff.data import make_record, synthetic
from framediff.ir import read_json, read_jsonl, write_jsonl


def test_hf_backend_dispatches_to_generic_runtime(monkeypatch):
    marker=object();captured=[]
    monkeypatch.setattr(vlm,'hf_generate',lambda args,prompt,images,runtime:(captured.append(runtime) or ('ok',{})))
    args=type('Args',(),{'backend':'hf'})()
    assert vlm.generate(args,'prompt',[],marker)[0]=='ok'
    assert captured==[marker]


class FakeBrowser:
    def __enter__(self):
        self.executions=0
        return self

    def __exit__(self,*args):
        pass

    def render(self,tree,viewport,path):
        self.executions+=1
        Image.new('RGB',tuple(viewport)).save(path)


def test_baseline_truth_is_private_and_failures_retained(tmp_path,monkeypatch):
    monkeypatch.setattr('framediff.vlm_baseline.Browser',FakeBrowser)
    record=make_record(synthetic(1),'a','test-site',1,viewports=[[320,240]])
    record['evaluation_observations']=[{'viewport':[320,240],'target':{'SECRET_TRUTH':[1,2,3,4]}}]
    write_jsonl(tmp_path/'input.jsonl',[record])
    captured=[]
    def generate(args,prompt,images):
        captured.append(prompt)
        assert 'SECRET_TRUTH' not in prompt
        assert len(images)==1
        return 'malformed output',{}
    monkeypatch.setattr(vlm,'api_generate',generate)
    command=['vlm-baseline','--data',str(tmp_path/'input.jsonl'),'--out',str(tmp_path/'baseline'),
             '--backend','openai-compatible']
    main(command)
    updated=list(read_jsonl(tmp_path/'baseline/test.jsonl'))[0]
    assert updated['current']==record['current']
    assert updated['vlm_baseline']['result']==record['current']
    assert updated['vlm_baseline']['status']=='failed'
    main(command+['--resume'])
    assert len(captured)==1
    main(['evaluate','--data',str(tmp_path/'baseline/test.jsonl'),'--out',str(tmp_path/'eval'),
          '--methods','none,vlm-revise','--device','cpu'])
    summary=read_json(tmp_path/'eval/summary.json')
    assert summary[0]['box_iou']==summary[1]['box_iou']
    assert summary[1]['vlm_failure_rate']==1
    assert summary[1]['repair_seconds']>=updated['vlm_baseline']['seconds']


def test_screenshot_baseline_does_not_receive_target_boxes(tmp_path,monkeypatch):
    monkeypatch.setattr('framediff.vlm_baseline.Browser',FakeBrowser)
    record=make_record(synthetic(1),'a','test-site',1,viewports=[[320,240]])
    record['screenshot']=str(tmp_path/'reference.png')
    Image.new('RGB',(320,240)).save(record['screenshot'])
    write_jsonl(tmp_path/'input.jsonl',[record])
    def generate(args,prompt,images):
        assert args.frames is None and len(images)==2
        assert 'Image 1 is the TARGET' in prompt
        return json.dumps(record['clean']),{}
    monkeypatch.setattr(vlm,'api_generate',generate)
    main(['vlm-baseline','--data',str(tmp_path/'input.jsonl'),'--out',str(tmp_path/'baseline'),
          '--backend','openai-compatible'])
    updated=list(read_jsonl(tmp_path/'baseline/test.jsonl'))[0]
    assert updated['vlm_baseline']['result']==record['clean']
    assert updated['current']==record['current']
    assert updated['vlm_baseline']['input_mode']=='screenshot'

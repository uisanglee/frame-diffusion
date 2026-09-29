import copy
import json
import pytest
from framediff.cli import main
from framediff.data import make_record,synthetic
from framediff.ir import write_jsonl,read_json,write_json

def test_train_resume_and_independent_evaluation(tmp_path):
    train=make_record(synthetic(1),'train','train-domain',1,viewports=[[1024,768]])
    val=make_record(synthetic(2),'val','val-domain',2,viewports=[[1024,768]])
    write_jsonl(tmp_path/'train.jsonl',[train]);write_jsonl(tmp_path/'val.jsonl',[val])
    args=['train','--train',str(tmp_path/'train.jsonl'),'--val',str(tmp_path/'val.jsonl'),
          '--out',str(tmp_path/'run'),'--hidden','32','--heads','4','--layers','1',
          '--batch-size','1','--steps','2','--device','cpu']
    main(args)
    args[args.index('--steps')+1]='3'
    main(args+['--resume',str(tmp_path/'run/last.pt')])
    import torch
    assert torch.load(tmp_path/'run/last.pt',weights_only=True)['step']==3
    predicted=copy.deepcopy(val)
    predicted['target_kind']='predicted_frames'
    predicted['evaluation_observations']=copy.deepcopy(val['observations'])
    for box in predicted['observations'][0]['target'].values():box[0]+=200
    write_jsonl(tmp_path/'test.jsonl',[predicted])
    main(['evaluate','--data',str(tmp_path/'test.jsonl'),'--out',str(tmp_path/'eval'),'--methods','copy-boxes','--device','cpu'])
    assert read_json(tmp_path/'eval/summary.json')[0]['box_iou']<.5
    # Ground truth must never be silently replaced by predicted frames.
    predicted.pop('evaluation_observations');write_jsonl(tmp_path/'bad.jsonl',[predicted])
    with pytest.raises(ValueError,match='independent'):
        main(['evaluate','--data',str(tmp_path/'bad.jsonl'),'--out',str(tmp_path/'bad-eval'),'--methods','none'])

def test_one_shot(tmp_path):
    for split in ('train','val'):
        record=make_record(synthetic(3),split,split,3,viewports=[[1024,768]])
        write_jsonl(tmp_path/f'{split}.jsonl',[record])
    main(['train','--train',str(tmp_path/'train.jsonl'),'--val',str(tmp_path/'val.jsonl'),
          '--out',str(tmp_path/'run'),'--hidden','32','--heads','4','--layers','1',
          '--batch-size','1','--steps','2','--device','cpu','--objective','boxes'])
    main(['evaluate','--data',str(tmp_path/'val.jsonl'),'--out',str(tmp_path/'eval'),
          '--checkpoint',str(tmp_path/'run/best.pt'),'--methods','one-shot','--device','cpu'])
    assert read_json(tmp_path/'eval/summary.json')[0]['n']==1

def test_vlm_api_adapter_mock(tmp_path,monkeypatch):
    from framediff import vlm
    from PIL import Image
    tree=synthetic(3);write_json(tmp_path/'current.json',tree)
    Image.new('RGB',(100,100)).save(tmp_path/'reference.png')
    captured={}
    def generate(args,prompt,images):
        captured['prompt']=prompt
        assert len(images)==1
        return json.dumps(tree),{'usage':{'total_tokens':1}}
    monkeypatch.setattr(vlm,'api_generate',generate)
    main(['vlm','--backend','openai-compatible','--task','revise-ir','--current',str(tmp_path/'current.json'),
          '--image',str(tmp_path/'reference.png'),'--out',str(tmp_path/'revised.json')])
    assert read_json(tmp_path/'revised.json')==tree
    assert 'preserving exact node list' in captured['prompt']

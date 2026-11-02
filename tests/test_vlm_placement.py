from types import SimpleNamespace
import sys
import pytest
import torch
from PIL import Image
from framediff.vlm import pin_revision, verify_cuda_placement

@pytest.mark.parametrize('pad,eos,expected', [
    (None, 2, {'pad_token_id': 2}),
    (None, [2,3], {'pad_token_id': 2}),
    (0, 2, {}), (None, None, {}), (None, [], {}),
])
def test_explicit_padding_preserves_existing_config(pad, eos, expected):
    from framediff.vlm import explicit_padding
    config = SimpleNamespace(pad_token_id=pad, eos_token_id=eos)
    assert explicit_padding(SimpleNamespace(generation_config=config)) == expected
    assert config.pad_token_id == pad and config.eos_token_id == eos

@pytest.mark.parametrize('model,expected', [
    ('Qwen/Qwen3-VL-8B-Instruct', .7),
    ('zai-org/GLM-4.1V-9B-Thinking', .8),
    ('zai-org/UI2Code_N', .8),
    ('google/gemma-3-12b-it', 0.),
    ('llava-hf/llava-v1.6-mistral-7b-hf', .2),
    ('mistral-experimental/pixtral-12b', .35),
])
def test_paper_temperature_resolution(model, expected):
    from framediff.vlm import initial_temperature
    assert initial_temperature(model) == expected
    assert initial_temperature(model, '0.1') == .1


def test_unknown_and_invalid_temperature_fail_explicitly():
    from framediff.vlm import initial_temperature
    for model, value in [('unknown', 'auto'), ('unknown', 'nan'), ('unknown', '-1')]:
        with pytest.raises(ValueError):
            initial_temperature(model, value)


def test_metadata_records_inherited_values_without_mutating_model():
    from framediff.vlm import generation_metadata
    base = {'temperature': .8, 'top_p': .6, 'top_k': 2, 'eos_token_id': [1,2], 'do_sample': True}
    model = SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda:dict(base)))
    result = generation_metadata(model, {'do_sample': False, 'stopping_criteria': [object()]}, 16384)
    assert result['top_p'] == .6 and result['top_k'] == 2
    assert not result['sampling_active']
    assert result['eos_token_id'] == [1,2]
    assert 'stopping_criteria' not in result
    assert base['do_sample'] is True


def test_revision_pin_reused_without_hub(tmp_path, monkeypatch):
    calls=[]
    def info(model, revision):
        calls.append((model, revision))
        return SimpleNamespace(sha='a'*40)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda:SimpleNamespace(model_info=info)))
    assert pin_revision('model', 'main', tmp_path)=='a'*40
    assert pin_revision('model', 'main', tmp_path)=='a'*40
    assert len(calls)==1
    with pytest.raises(ValueError, match='changed'):
        pin_revision('other', 'main', tmp_path)
    with pytest.raises(ValueError, match='changed'):
        pin_revision('model', 'other', tmp_path)


def test_unpinned_old_run_rejected(tmp_path):
    (tmp_path/'prepare').mkdir()
    (tmp_path/'prepare/config.json').write_text('{}')
    with pytest.raises(ValueError, match='no revision pin'):
        pin_revision('model', 'main', tmp_path)


@pytest.mark.parametrize('device,mapping,valid', [
    ('cuda:0', {'':0}, True), ('cpu', {'':'cpu'}, False),
    ('cuda:1', {'':1}, False), ('cuda:0', {'':'disk'}, False),
    ('meta', {'':'cpu'}, False),
])
def test_placement(device, mapping, valid, monkeypatch):
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda d:'test GPU')
    model=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device=torch.device(device))]),
                          buffers=lambda:iter([]),hf_device_map=mapping)
    if valid:
        verify_cuda_placement(model)
        assert model._framediff_placement['device']=='cuda:0'
    else:
        with pytest.raises(RuntimeError):verify_cuda_placement(model)


def test_glm_loader_pins_processor_and_single_gpu(monkeypatch):
    from framediff import vlm
    captured={}
    model=SimpleNamespace(config=SimpleNamespace(_commit_hash='a'*40))
    model.eval=lambda:model
    def load(name,**kwargs):
        captured['model']=(name,kwargs)
        return model
    def processor(name,**kwargs):
        captured['processor']=kwargs
        return object()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a,**k:pytest.fail('wrong loader')),
        Glm4vForConditionalGeneration=SimpleNamespace(from_pretrained=load),
        AutoProcessor=SimpleNamespace(from_pretrained=processor),
        BitsAndBytesConfig=lambda **kw:kw))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(vlm,'verify_cuda_placement',lambda m:captured.update(checked=m))
    args=SimpleNamespace(model='zai-org/GLM-4.1V-9B-Thinking',revision='a'*40,seed=42,four_bit=True)
    vlm.load_hf_runtime(args)
    assert captured['model'][1]['device_map']=={'':0}
    assert captured['model'][1]['quantization_config']['load_in_4bit']
    assert captured['processor']['revision']=='a'*40
    assert captured['checked'] is model


def test_llava_next_loader_uses_nf4_and_one_gpu(monkeypatch):
    from framediff import vlm
    captured={}
    model=SimpleNamespace(config=SimpleNamespace(_commit_hash='b'*40))
    model.eval=lambda:model
    def load(name,**kwargs):
        captured['model']=(name,kwargs)
        return model
    def processor(name,**kwargs):
        captured['processor']=(name,kwargs)
        return object()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a,**k:pytest.fail('wrong loader')),
        LlavaNextForConditionalGeneration=SimpleNamespace(from_pretrained=load),
        AutoProcessor=SimpleNamespace(from_pretrained=processor),
        BitsAndBytesConfig=lambda **kw:kw))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(vlm,'verify_cuda_placement',lambda m:captured.update(checked=m))
    args=SimpleNamespace(model='llava-hf/llava-v1.6-mistral-7b-hf',revision='b'*40,
        seed=42,four_bit=True)
    vlm.load_hf_runtime(args)
    name,kwargs=captured['model']
    assert name==args.model
    assert kwargs['revision']=='b'*40
    assert kwargs['device_map']=={'':0}
    assert kwargs['quantization_config']=={
        'load_in_4bit':True,'bnb_4bit_quant_type':'nf4',
        'bnb_4bit_compute_dtype':torch.bfloat16,'bnb_4bit_use_double_quant':True}
    assert captured['processor']==(args.model,{'revision':'b'*40})
    assert captured['checked'] is model


@pytest.mark.parametrize('backend', ['hf', 'qwen'])
def test_two_image_revision_keeps_target_first_and_allows_short_output(monkeypatch, backend):
    from framediff import vlm
    seen={}
    class Inputs(dict):
        def to(self,device):
            seen['device']=str(device)
            return self
    class Processor:
        tokenizer=SimpleNamespace(get_vocab=lambda:{})
        def apply_chat_template(self,messages,**kwargs):
            seen['content']=messages[0]['content']
            return Inputs(input_ids=torch.tensor([[1,2]]))
        def batch_decode(self,tokens,**kwargs):
            return ['<html><body>ok</body></html>']
    class Model:
        device=torch.device('cpu')
        config=SimpleNamespace(_commit_hash='b'*40)
        _framediff_placement={'device':'cuda:0'}
        def generate(self,**inputs):
            seen['generation']=inputs
            return torch.tensor([[1,2,3]])
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:123)
    args=SimpleNamespace(image_labels=['Reference Webpage:','Current Webpage:'],
        prompt_first=True,max_new_tokens=16384,four_bit=True)
    images=[Image.new('RGB',(4,4),'red'),Image.new('RGB',(4,4),'blue')]
    answer,meta=getattr(vlm,backend+'_generate')(args,'Revise the HTML',images,(Model(),Processor()))
    assert seen['generation']['max_new_tokens']==16384
    assert seen['generation']['min_new_tokens']==0
    assert 'min_length' not in seen['generation']  # Avoid conflicting minimum-length options.
    assert 'eos_token_id' not in seen['generation']  # Preserve each model's EOS configuration.
    assert meta['output_tokens']==1
    content=seen['content']
    assert [part['type'] for part in content]==['text','text','image','text','image']
    assert [part['text'] for part in content if part['type']=='text']==[
        'Revise the HTML','Reference Webpage:','Current Webpage:']
    assert content[2]['image'] is images[0] and content[4]['image'] is images[1]
    assert answer=='<html><body>ok</body></html>'
    assert meta['weight_precision']=='nf4-4bit'


@pytest.mark.parametrize('ids,expected', [([1,4,2,5],[5]),([1,4],None),([5],[5])])
def test_reasoning_not_executed(ids,expected):
    from framediff.vlm import decode_hf_answer
    decoded=[]
    processor=SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda:{'<think>':1,'</think>':2}),
        batch_decode=lambda tokens,**kwargs:decoded.append(tokens[0].tolist()) or ['HTML'])
    answer=decode_hf_answer(processor,torch.tensor([ids]))
    assert answer==('' if expected is None else 'HTML')
    assert decoded==([] if expected is None else [expected])

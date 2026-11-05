from pathlib import Path
from types import SimpleNamespace

import pytest

from framediff.design2code_sat import (OFFICIAL_SETTINGS, design2code_generate,
                                      runtime_settings, validate_request)


def test_official_contract_and_generation_settings():
    assert OFFICIAL_SETTINGS == {'max_length':4096, 'temperature':.5, 'top_p':1.,
                                'top_k':1, 'repetition_penalty':1.1}
    settings = runtime_settings()
    assert settings['query'] == '' and settings['text_processor_length'] == 2048
    assert settings['model_parallel_size'] == 2 and settings['weight_precision'] == 'bf16'


def test_does_not_mislabel_repair_as_generation():
    args = SimpleNamespace(task='revise-html', four_bit=False)
    with pytest.raises(ValueError, match='self-revision'):
        validate_request(args, [object(), object()])
    args.task = 'generate-html'
    args.four_bit = True
    with pytest.raises(ValueError, match='BF16'):
        validate_request(args, [object()])


def test_runtime_reused_and_generic_prompt_not_forwarded():
    calls = []
    runtime = SimpleNamespace(generate=lambda args, images: (calls.append(images) or 'HTML', {'output_tokens':3}))
    args = SimpleNamespace(task='generate-html', four_bit=False)
    answer, metadata = design2code_generate(args, '', ['image'], runtime)
    assert answer == 'HTML' and metadata['output_tokens'] == 3
    assert calls == [['image']]


def test_torchrun_worker_records_actual_tokens_and_uses_official_model_class():
    # Static guard for accidental generic AutoModel loading; smoke tests need two GPUs.
    path = Path(__file__).parents[1]/'framediff/design2code_sat_worker.py'
    source = path.read_text()
    compile(source, str(path), 'exec')
    assert 'FineTuneTestCogAgentModel.from_pretrained_base' in source
    assert "query='" not in source


def test_official_zip_extracted_once_with_nested_directory(tmp_path, monkeypatch):
    import zipfile
    from framediff.design2code_sat import checkpoint_directory
    snapshot = tmp_path/'hub'
    snapshot.mkdir()
    cache = tmp_path/'cache'
    monkeypatch.setenv('DESIGN2CODE_CHECKPOINT_CACHE', str(cache))
    with zipfile.ZipFile(snapshot/'design2code-18b-v0.zip', 'w') as archive:
        archive.writestr('design2code-18b-v0/model_config.json', '{}')
        archive.writestr('design2code-18b-v0/1/mp_rank_00_model_states.pt', 'weights')
    result = checkpoint_directory(snapshot, 'commit')
    assert (result/'model_config.json').is_file()
    (result/'sentinel').write_text('keep')
    assert checkpoint_directory(snapshot, 'commit') == result
    assert (result/'sentinel').read_text() == 'keep'


def test_archive_path_traversal_rejected(tmp_path, monkeypatch):
    import zipfile
    from framediff.design2code_sat import checkpoint_directory
    monkeypatch.setenv('DESIGN2CODE_CHECKPOINT_CACHE', str(tmp_path/'cache'))
    with zipfile.ZipFile(tmp_path/'design2code-18b-v0.zip', 'w') as archive:
        archive.writestr('../escape', 'bad')
    with pytest.raises(ValueError, match='Unsafe path'):
        checkpoint_directory(tmp_path, 'commit')


def test_nonofficial_query_and_temperature_rejected():
    args = SimpleNamespace(task='generate-html', four_bit=False, temperature=.5)
    with pytest.raises(ValueError, match='empty query'):
        design2code_generate(args, 'unofficial instruction', [object()])
    args.temperature = .7
    with pytest.raises(ValueError, match='temperature is 0.5'):
        validate_request(args, [object()])


@pytest.mark.parametrize('partitions', [1, 3])
def test_cpu_lora_merge_retains_linear_and_matches_factor_computation(partitions):
    torch = pytest.importorskip('torch')
    from framediff.design2code_sat_worker import merge_sat_lora_cpu
    # Distinct base class must survive merging so SAT can recognize/shard it.
    class ParallelLinear(torch.nn.Linear):
        pass
    class LoraLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original = ParallelLinear(6, 4*partitions)
            self.matrix_A = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(2, 6)) for _ in range(partitions)])
            self.matrix_B = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(4, 2)) for _ in range(partitions)])
            self.scaling = .125
    torch.manual_seed(42)
    layer = LoraLinear()
    x = torch.randn(5, 6)
    reference = layer.original(x) + torch.cat([(x@a.T)@b.T*layer.scaling for a,b in zip(layer.matrix_A, layer.matrix_B)], -1)
    model = torch.nn.Sequential(layer)
    assert merge_sat_lora_cpu(model) == 1
    assert isinstance(model[0], ParallelLinear)
    torch.testing.assert_close(model(x), reference)
    # Reconstruct row- or column-partitioned matrix multiplication after merge.
    w, bias = model[0].weight, model[0].bias
    row_result = x[:,:3]@w[:,:3].T + x[:,3:]@w[:,3:].T + bias
    column_result = torch.cat([x@part.T for part in w.chunk(2, dim=0)], dim=-1)+bias
    torch.testing.assert_close(row_result, reference)
    torch.testing.assert_close(column_result, reference)

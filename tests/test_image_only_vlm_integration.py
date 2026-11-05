from types import SimpleNamespace
import sys

from PIL import Image
import pytest

from framediff import vlm
from framediff.runtime_errors import InferenceWorkerError, raise_if_fatal
from framediff.web_experiment import prepare


@pytest.mark.parametrize('backend', vlm.IMAGE_ONLY_BACKENDS)
def test_official_images_are_not_resized_or_alpha_discarded(tmp_path, backend):
    path = tmp_path/'source.png'
    Image.new('RGBA', (2000, 1200), (0,0,0,0)).save(path)
    args = SimpleNamespace(backend=backend, image=[path], max_pixels=16)
    result = vlm.images_for(args)[0]
    assert result.size == (2000, 1200) and result.mode == 'RGBA'


@pytest.mark.parametrize('backend', vlm.IMAGE_ONLY_BACKENDS)
def test_image_only_rejects_self_revision_before_any_generation(backend):
    args = SimpleNamespace(backend=backend, rounds=2, initial_mode='direct')
    with pytest.raises(ValueError, match='rounds=0'):
        prepare(args)


def test_worker_failure_is_fatal_not_cached_as_page_generation_failure(monkeypatch):
    def fail(*a):
        raise RuntimeError('worker incompatible with custom model')
    monkeypatch.setitem(sys.modules, 'framediff.websight_backend', SimpleNamespace(websight_generate=fail))
    args = SimpleNamespace(backend='websight')
    with pytest.raises(InferenceWorkerError) as caught:
        vlm.generate(args, '', [])
    with pytest.raises(InferenceWorkerError):
        raise_if_fatal(caught.value)


def test_verified_image_only_temperature_defaults():
    assert vlm.initial_temperature('HuggingFaceM4/VLM_WebSight_finetuned') == 0
    assert vlm.initial_temperature('SALT-NLP/Design2Code-18B-v0') == .5


@pytest.mark.browser
@pytest.mark.parametrize('backend,model', [('websight', 'HuggingFaceM4/VLM_WebSight_finetuned'),
                                         ('design2code-sat', 'SALT-NLP/Design2Code-18B-v0')])
def test_official_initial_pipeline_empty_prompt_reuse_and_worker_close(tmp_path, monkeypatch, backend, model):
    import json
    root = tmp_path/'dataset'
    root.mkdir()
    Image.new('RGB', (320,240), 'white').save(root/'a.png')
    html = '<html><body><h1>Header</h1><p>Visible content</p></body></html>'
    (root/'a.html').write_text(html.replace('Visible content', 'SECRET_REFERENCE_TEXT'))
    calls = []
    runtime = SimpleNamespace(close=lambda: calls.append('close'))
    monkeypatch.setattr(vlm, 'load_runtime', lambda args: runtime)
    def generate(args, prompt, images, loaded):
        assert loaded is runtime and prompt == '' and len(images) == 1
        assert args.task == 'generate-html' and args.current is None
        calls.append('generate')
        return html, {'input_tokens':66, 'output_tokens':30}
    monkeypatch.setattr(vlm, 'generate', generate)
    args = SimpleNamespace(root=str(root), manifest=None, out=str(tmp_path/'prepared'),
        backend=backend, model=model, rounds=0, initial_mode='direct', revision_protocol='shared',
        repair_conditioning='visual', vlm_retries=0, max_nodes=16, max_pixels=1048576,
        max_new_tokens=16384, limit=0, seed=42, resume=False, four_bit=False)
    rows = prepare(args)
    assert not rows[0]['errors']
    assert set(rows[0]['methods']) == {'initial'}
    assert rows[0]['uses_reference_text'] is False
    assert calls == ['generate', 'close']
    prompt_file = next((tmp_path/'prepared/pages').glob('*/initial.prompt.json'))
    assert json.loads(prompt_file.read_text())['prompt'] == ''
    args.resume = True
    assert prepare(args) == rows
    assert calls == ['generate', 'close']

import importlib.util
import json
from pathlib import Path
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location('ui2coden_runner', Path(__file__).parents[1] / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
HTML = '<html><body><div style="width:100px;height:40px">hello</div></body></html>'


def test_polishing_prompt_preserves_apps_and_uses_supplied_image_assets():
    prompt = runner.polishing_prompt(HTML)
    assert 'complete, runnable HTML document' in prompt
    assert 'viewport' not in prompt and 'compact' not in prompt
    assert 'preserve the existing text, supplied assets, and application behavior' in prompt
    assert 'React, JSX, Babel, and Tailwind are allowed' in prompt
    assert 'rick.jpg' in prompt and 'Do not generate new images' in prompt
    assert prompt.endswith('\nCurrent HTML:\n' + HTML)


@pytest.fixture
def record(tmp_path, monkeypatch):
    # Unit tests exercise handoff/adoption without networks or model inference.
    # Real JS/static capture equivalence is covered in test_app_browser.py.
    def freeze(html, viewport, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        source=folder/'source.html'; source.write_text(html)
        static=folder/'static.html'; static.write_text(html)
        png=folder/'render.png'; Image.new('RGB', viewport, 'white').save(png)
        return dict(source_html=str(source),static_html=str(static),screenshot=str(png),
                    pixel_mae=0.,max_box_error_px=0.,seconds=.01,
                    browser_executions=2,browser_screenshots=2,contract=runner.APPLICATION_CONTRACT)
    monkeypatch.setattr(runner, 'freeze_application', freeze)
    (tmp_path/'initial.html').write_text(HTML)
    (tmp_path/'reference.html').write_text(HTML.replace('hello', 'SECRET_REFERENCE'))
    Image.new('RGB', (320, 240), 'white').save(tmp_path/'target.png')
    manifest = tmp_path/'input.jsonl'
    manifest.write_text(json.dumps(dict(id='test/a', screenshot='target.png',
        initial_html='initial.html', html='reference.html', viewport=[320, 240])))
    return runner.load_records(manifest, tmp_path, 0)[0]


class Browser:
    executions = 0
    screenshots = 0
    def reset_context(self): pass
    def snapshot(self, html, viewport, png, max_nodes, require_content=False):
        self.executions += 1
        Image.new('RGB', viewport, 'white').save(png)
        self.screenshots += 1


def test_application_polishing_does_not_prohibit_react():
    from framediff.html_generation import STATIC_HTML_PROMPT
    assert STATIC_HTML_PROMPT not in runner.PROMPT
    assert 'Do not use React' not in runner.PROMPT
    assert 'disables JavaScript' not in runner.PROMPT


def test_failed_application_capture_keeps_previous_html(record, tmp_path, monkeypatch):
    react = '<html><body><div id="root"></div><script type="text/babel">ReactDOM.render(jsx, root)</script></body></html>'
    runtime = Runtime([react, HTML])
    freeze=runner.freeze_application
    def verify(html,*args,**kwargs):
        if 'text/babel' in html:raise ValueError('application failed to mount')
        return freeze(html,*args,**kwargs)
    monkeypatch.setattr(runner,'freeze_application',verify)
    result = runner.run_page(record, tmp_path/'page', 2, runtime, Browser())
    first = result['methods']['ui2coden-round-1']
    assert first['failed'] and 'failed to mount' in first['error']
    assert first['html'] == record['initial_html']
    assert runtime.calls[1] == HTML
    assert not result['methods']['ui2coden-round-2']['failed']


class Runtime:
    def __init__(self, answers): self.answers = iter(answers); self.calls = []
    def generate(self, target, current, html):
        assert 'SECRET_REFERENCE' not in html
        self.calls.append(html)
        answer = next(self.answers)
        if isinstance(answer, Exception): raise answer
        return answer, dict(input_tokens=100, output_tokens=20)


def test_rounds_resume_and_metrics_schema(record, tmp_path):
    runtime = Runtime([HTML.replace('100px', '120px'), HTML.replace('100px', '140px')])
    folder = tmp_path/'page'
    result = runner.run_page(record, folder, 2, runtime, Browser())
    assert '120px' in runtime.calls[1]
    last = result['methods']['ui2coden-round-2']
    assert last['repair_vlm_calls'] == 2
    assert last['repair_input_tokens'] == 200
    assert last['browser_executions'] == 6
    assert last['repair_seconds'] >= result['methods']['ui2coden-round-1']['repair_seconds']
    assert runner.run_page(record, folder, 2, None, Browser()) == result
    Path(last['html']).write_text('tampered')
    with pytest.raises(ValueError, match='Cached output changed'):
        runner.run_page(record, folder, 2, None, Browser())


def test_failed_generation_keeps_initial_and_counts_tokens(record, tmp_path):
    result = runner.run_page(record, tmp_path/'page', 1, Runtime(['<html>truncated']), Browser())
    method = result['methods']['ui2coden-round-1']
    assert method['failed'] and method['html'] == record['initial_html']
    assert method['repair_output_tokens'] == 20
    assert method['candidate_failure_count'] == 1


def test_cuda_failure_aborts_without_poisoning_resume(record, tmp_path):
    with pytest.raises(RuntimeError, match='CUDA'):
        runner.run_page(record, tmp_path/'page', 1, Runtime([RuntimeError('CUDA out of memory')]), Browser())
    assert not (tmp_path/'page/round-1.json').exists()


def test_policy_adapter(record, tmp_path):
    path = tmp_path/'policy.jsonl'
    path.write_text(json.dumps(dict(id='policy-test', target_image=record['screenshot'],
        current_html=record['initial_html'], target_html=record['html'])))
    r = runner.load_records(path, tmp_path, 0)[0]
    assert r['viewport'] == [320, 240]
    assert r['initial_html'] == record['initial_html']


def test_reference_render_protocol_survives_adapter(record, tmp_path):
    observation = tmp_path / 'target-dom.json'; observation.write_text('{}')
    manifest = tmp_path / 'prepared.jsonl'
    manifest.write_text(json.dumps({
        'id': 'test/a', 'screenshot': record['screenshot'], 'html': record['html'],
        'viewport': record['viewport'], 'target_image_source': 'reference-render',
        'benchmark_protocol': 'local-reference-render-v1',
        'reference_observation': str(observation),
        'methods': {'initial': {'html': record['initial_html']}}}) + '\n')
    loaded = runner.load_records(manifest, tmp_path, 0)[0]
    assert loaded['benchmark_protocol'] == 'local-reference-render-v1'
    assert loaded['target_image_source'] == 'reference-render'
    assert loaded['reference_observation'] == str(observation)


def test_common_evaluator_with_real_browser(record, tmp_path):
    """Exercises actual Chromium + existing reporting, no model/network/GPU."""
    from argparse import Namespace
    from framediff.web_experiment import evaluate_pages
    with runner.HtmlBrowser() as browser:
        browser.snapshot(HTML, record['viewport'], record['screenshot'], 2048)
        result = runner.run_page(record, tmp_path/'page', 1, Runtime([HTML]), browser)
    manifest = tmp_path/'results.jsonl'
    manifest.write_text(json.dumps(result)+'\n')
    out = tmp_path/'evaluation'
    evaluate_pages(Namespace(data=str(manifest), out=str(out), max_nodes=2048, official_repo=None, resume=False))
    metrics = [json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    assert len(metrics) == 2
    assert all(r['evaluation_error'] is None for r in metrics)
    assert metrics[1]['pixel_mae'] == 0
    assert metrics[1]['dom_box_iou'] == 1
    assert (out/'paper-report.md').is_file()


def test_application_capture_inside_active_browser(record,tmp_path,monkeypatch):
    """Real JS capture must coexist with the parent's active Playwright context."""
    from framediff.app_browser import freeze_application
    monkeypatch.setattr(runner,'freeze_application',freeze_application)
    app='''<html><body><div id="root"></div><script>
      document.getElementById('root').innerHTML='<button style="width:100px">Hello</button>';
    </script></body></html>'''
    with runner.HtmlBrowser() as browser:
        result=runner.run_page(record,tmp_path/'page',1,Runtime([app]),browser)
        # Capture cleanup must not close the parent browser/context.
        browser.snapshot(HTML,record['viewport'],tmp_path/'after.png',2048)
    method=result['methods']['ui2coden-round-1']
    assert not method['failed'], method['error']
    assert method['application_capture']['pixel_mae']==0
    assert method['browser_executions']==3  # input + application + static verification
    assert '<button' in Path(method['html']).read_text()
    assert 'innerHTML' in Path(method['application_html']).read_text()


def test_later_failures_keep_last_success_and_resume(record, tmp_path):
    repaired = HTML.replace('100px', '150px')
    runtime = Runtime([repaired, '<html>truncated', ValueError('render failure')])
    result = runner.run_page(record, tmp_path/'page', 3, runtime, Browser())
    first = result['methods']['ui2coden-round-1']['html']
    assert first != record['initial_html']
    for step in (2, 3):
        method = result['methods'][f'ui2coden-round-{step}']
        assert method['failed'] and method['html'] == first
        assert '150px' in Path(method['html']).read_text()
    assert runner.run_page(record, tmp_path/'page', 3, None, Browser()) == result


def test_rounds_keep_application_source_separate_from_static_capture(record, tmp_path, monkeypatch):
    source='<html><body><div id="root"></div><script>mountApp()</script></body></html>'
    original_freeze=runner.freeze_application
    def freeze(html,*args,**kwargs):
        capture=original_freeze(html,*args,**kwargs)
        Path(capture['static_html']).write_text(HTML)
        return capture
    monkeypatch.setattr(runner,'freeze_application',freeze)
    class StaticBrowser(Browser):
        def snapshot(self,html,*args,**kwargs):
            assert 'mountApp' not in html
            return super().snapshot(html,*args,**kwargs)
    runtime=Runtime([source,'<html>truncated',source])
    result=runner.run_page(record,tmp_path/'page',3,runtime,StaticBrowser())
    assert runtime.calls==[HTML,source,source]
    first=result['methods']['ui2coden-round-1']
    failed=result['methods']['ui2coden-round-2']
    assert Path(first['html']).read_text()==HTML
    assert Path(first['application_html']).read_text()==source
    assert failed['failed']
    assert failed['application_html']==first['application_html']
    assert failed['application_capture']==first['application_capture']
    assert result['methods']['ui2coden-round-3']['application_attempt']['pixel_mae']==0
    Path(first['application_html']).write_text('tampered app')
    with pytest.raises(ValueError,match='application source changed'):
        runner.run_page(record,tmp_path/'page',3,None,StaticBrowser())


def test_adapter_preserves_and_resolves_initial_application(record,tmp_path):
    (tmp_path/'application.html').write_text('<html><body><script>mount()</script></body></html>')
    path=tmp_path/'app-input.jsonl'
    path.write_text(json.dumps(dict(id='app',screenshot=record['screenshot'],
        methods={'initial':dict(html='initial.html',application_html='application.html',
            application_capture={'verified':True})}))+'\n')
    loaded=runner.load_records(path,tmp_path,0)[0]
    assert loaded['initial_metadata']['application_html']==str(tmp_path/'application.html')
    assert loaded['initial_application_html']==str(tmp_path/'application.html')
    assert loaded['initial_metadata']['application_capture']=={'verified':True}


def test_application_renderer_receives_benchmark_placeholder(record,tmp_path):
    Image.new('RGB',(10,10),'blue').save(tmp_path/'rick.jpg')
    generated='<html><body><img src="rick.jpg"></body></html>'
    result=runner.run_page(record,tmp_path/'page',1,Runtime([generated]),Browser())
    method=result['methods']['ui2coden-round-1']
    assert not method['failed']
    assert 'data:image/jpeg;base64,' in Path(method['application_html']).read_text()
    assert (tmp_path/'page/answer-1.txt').read_text()==generated


def test_webui_view_ids_do_not_collapse(record, tmp_path):
    path = tmp_path/'views.jsonl'
    path.write_text(json.dumps(dict(id='page/mobile', page_id='page',
        screenshot=record['screenshot'], initial_html=record['initial_html'],
        html=record['html']))+'\n')
    assert runner.load_records(path, tmp_path, 0)[0]['page_id'] == 'page/mobile'


def test_long_html_keeps_full_output_budget():
    from types import SimpleNamespace
    config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=65536),
                             vision_config=SimpleNamespace(max_position_embeddings=1024))
    limit = runner.context_limit(config)
    assert limit == 65536
    # 17k input must not consume the separate 16k output allowance.
    runner.check_budget(17000, 16384, limit)
    runner.check_budget(49152, 16384, limit)
    with pytest.raises(ValueError, match='no input truncation'):
        runner.check_budget(49153, 16384, limit)
    assert runner.context_limit(config, 32768) == 32768
    with pytest.raises(ValueError, match='model limit'):
        runner.context_limit(config, 100000)
    with pytest.raises(ValueError, match='max_position_embeddings'):
        runner.context_limit(SimpleNamespace())


@pytest.mark.parametrize('produced_tokens', [1, 16384])
def test_polishing_dispatch_keeps_long_input_and_bf16_metadata(record, tmp_path, monkeypatch, produced_tokens):
    from types import SimpleNamespace
    import torch
    seen = {}
    class Inputs(dict):
        def to(self, device):
            seen['device'] = str(device)
            return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen['messages'] = messages
            return Inputs(input_ids=torch.zeros((1, 17000), dtype=torch.long))
        def decode(self, ids, **kwargs): return HTML
    class Model:
        device=torch.device('cpu')
        _framediff_input_device=torch.device('cuda:1')
        generation_config=SimpleNamespace(to_dict=lambda: dict(
            do_sample=True,temperature=.8,top_p=.6,top_k=2,
            eos_token_id=[151329,151336,151338,151348],pad_token_id=151329))
        def forward(self, input_ids, logits_to_keep=0): pass
        def generate(self, **kwargs):
            seen['generation'] = kwargs
            ids = torch.zeros((1, 17000 + produced_tokens), dtype=torch.long)
            if produced_tokens < kwargs['max_new_tokens']: ids[0, -1] = 151329
            return ids
    runtime = runner.Runtime.__new__(runner.Runtime)
    runtime.args = SimpleNamespace(max_pixels=262144, max_new_tokens=16384, protocol='layout')
    runtime.context_limit = 65536
    runtime.model, runtime.processor = Model(), Processor()
    monkeypatch.setattr(runner, 'start_cuda_measurement', lambda m: None)
    monkeypatch.setattr(runner, 'finish_cuda_measurement', lambda m: dict(
        peak_cuda_bytes=200, peak_cuda_bytes_by_device={'0':100,'1':200}, sum_device_peak_cuda_bytes=300))
    answer, stats = runtime.generate(record['screenshot'], record['screenshot'], HTML)
    assert answer == HTML and seen['device'] == 'cuda:1'
    assert seen['generation']['input_ids'].shape[1] == 17000
    assert seen['generation']['max_new_tokens'] == 16384
    assert set(seen['generation']) == {'input_ids','max_new_tokens','stopping_criteria'}
    assert len(seen['generation']['stopping_criteria']) == 1
    assert isinstance(seen['generation']['stopping_criteria'][0], runner.HTMLCompletionStop)
    assert stats['html_completion_stop'] is True
    settings=stats['generation_settings']
    assert settings['do_sample'] is True and settings['sampling_active'] is True
    assert (settings['temperature'],settings['top_p'],settings['top_k']) == (.8,.6,2)
    assert settings['eos_token_id']==[151329,151336,151338,151348]
    assert stats['generation_contract']==runner.GENERATION_CONTRACT
    assert stats['input_tokens'] == 17000 and stats['output_tokens'] == produced_tokens
    assert stats['hit_output_limit'] is (produced_tokens == 16384)
    assert stats['generation_stop_reason'] == ('max_new_tokens' if produced_tokens == 16384 else 'eos')
    result = runner.run_page(record, tmp_path/'bf16-page', 1, Runtime([HTML]), Browser(), precision='bf16')
    assert result['vlm_weight_precision'] == 'bf16'


@pytest.mark.parametrize('document', [
    HTML,
    '<think>The answer might contain <html><body>draft</body></html>.</think>' + HTML,
    '<html><body><!-- </html> -->visible<script>const end="</html>";</script></body></html>',
])
def test_polishing_stops_at_completed_html_before_trailing_explanation(record, monkeypatch, document):
    """Exercise the actual runner criterion through incremental mocked decoding."""
    import torch
    from types import SimpleNamespace
    class Inputs(dict):
        def to(self, device): return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            # An existing document in the input must not trigger output stopping.
            return Inputs(input_ids=torch.tensor([[ord(c) for c in HTML]]))
        def decode(self, ids, **kwargs): return ''.join(chr(int(i)) for i in ids)
        def batch_decode(self, ids, **kwargs): return [self.decode(row, **kwargs) for row in ids]
    class Model:
        device = torch.device('cpu')
        generation_config = SimpleNamespace(to_dict=lambda: dict(do_sample=True,temperature=.8))
        def generate(self, input_ids, max_new_tokens, stopping_criteria):
            assert max_new_tokens == 16384
            ids = input_ids
            for c in document + '\nUnwanted explanation after the HTML.':
                ids = torch.cat((ids, torch.tensor([[ord(c)]])), dim=1)
                if any(stop(ids, None) for stop in stopping_criteria): break
            return ids
    runtime = runner.Runtime.__new__(runner.Runtime)
    runtime.args = SimpleNamespace(max_pixels=262144, max_new_tokens=16384)
    runtime.context_limit = 65536
    runtime.model, runtime.processor = Model(), Processor()
    monkeypatch.setattr(runner, 'start_cuda_measurement', lambda model: None)
    monkeypatch.setattr(runner, 'finish_cuda_measurement', lambda model: {})
    answer, stats = runtime.generate(record['screenshot'], record['screenshot'], HTML)
    assert answer == document
    assert stats['generation_stop_reason'] == 'html_complete'
    assert stats['output_tokens'] == len(document)
    assert not stats['hit_output_limit']


def test_failed_initial_never_polished_even_with_old_round_cache(record, tmp_path, capsys):
    folder = tmp_path/'page'
    runner.run_page(record, folder, 2, Runtime([HTML, HTML]), Browser())
    old_round = (folder/'round-1.json').read_bytes()
    record['initial_metadata'] = dict(failed=True, error='truncated', seconds=10., vlm_calls=1)
    browser = Browser()
    result = runner.run_page(record, folder, 2, None, browser)
    assert browser.executions == 0
    for step in (1, 2):
        method = result['methods'][f'ui2coden-round-{step}']
        assert method['skipped'] and method['failed']
        assert method['html'] == record['initial_html']
        assert method['repair_vlm_calls'] == method['repair_seconds'] == 0
        assert method['visual_timing']['actions'] == 0
    assert (folder/'round-1.json').read_bytes() == old_round
    assert 'initial_generation_failed' in capsys.readouterr().out


def test_adapter_preserves_legacy_initial_error(record, tmp_path):
    manifest = tmp_path/'legacy.jsonl'
    manifest.write_text(json.dumps(dict(id='x', screenshot=record['screenshot'],
        initial_html=record['initial_html'], errors={'initial': 'truncated'}))+'\n')
    loaded = runner.load_records(manifest, tmp_path, 0)[0]
    assert loaded['initial_metadata']['failed']
    result = runner.run_page(loaded, tmp_path/'skipped', 2, None, Browser())
    assert result['methods']['ui2coden-round-2']['skip_reason'] == 'initial_generation_failed'

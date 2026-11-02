from types import SimpleNamespace
import torch
from framediff.html_generation import CSSRepeatStop, GeneratedPresencePenalty, repeated_css
from framediff.vlm import html_generation_options


def test_presence_penalty_is_once_per_generated_token_and_ignores_prompt():
    penalty = GeneratedPresencePenalty(2)
    original = torch.zeros((1, 5))
    result = penalty(torch.tensor([[0, 1, 2, 2, 3]]), original)
    assert result.tolist() == [[0, 0, -1.5, -1.5, 0]]
    assert original.sum() == 0


def test_sampling_options_and_non_html_scope():
    args = SimpleNamespace(task='generate-html', temperature=0.7)
    processor = SimpleNamespace(batch_decode=lambda *a, **k: [''])
    opts = html_generation_options(args, processor, 10, qwen=True)
    assert opts['top_p'] == .8 and opts['top_k'] == 20
    assert isinstance(opts['stopping_criteria'][0], CSSRepeatStop)
    assert len(opts['logits_processor']) == 1
    args.temperature = 0
    opts = html_generation_options(args, processor, 10, qwen=True)
    assert 'top_p' not in opts and 'logits_processor' not in opts
    assert isinstance(opts['stopping_criteria'][0], CSSRepeatStop)
    args.task = 'extract-frames'
    assert html_generation_options(args, processor, 10, qwen=True) == {}


def test_repeat_guard_stops_long_css_cycle_but_allows_normal_html():
    cycle = ('.a { width: 10px; }\n.b { height: 20px; }\n') * 15
    assert repeated_css('<style>' + cycle)
    assert not repeated_css('<style>' + cycle + '</style><body>ok</body>')
    assert not repeated_css('<style>.a{width:10px}.b{height:20px}</style>')
    assert not repeated_css('<style>' + ''.join(
        f'.item:nth-child({i}) {{ width: 10px; }}\n' for i in range(30)))
    assert not repeated_css('<style>@media (max-width: 800px) {' + cycle + '}')
    calls=[]
    processor=SimpleNamespace(batch_decode=lambda *a, **k: calls.append(1) or ['<style>'+cycle])
    guard=CSSRepeatStop(processor, prompt_tokens=10)
    assert not guard(torch.zeros((1, 1000), dtype=torch.long), None)
    assert calls == []
    assert guard(torch.zeros((1, 1034), dtype=torch.long), None)
    assert guard.reason == 'repeated_css_generation'
    assert len(calls) == 1

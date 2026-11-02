from types import SimpleNamespace
import torch
from framediff.html_generation import repeated_css, CSSRepeatStop, GeneratedPresencePenalty
from framediff.vlm import html_generation_options


def test_repeated_css_and_normal_html():
    footer = '.footer a{color:red}.footer{display:flex}.footer a:hover{color:blue}'
    assert repeated_css('<style>' + footer * 16)
    assert repeated_css('<style>' + ''.join(f'.x:nth-child({n}){{color:{["red","blue","green"][n%3]}}}' for n in range(48)))
    assert not repeated_css('<style>' + footer * 2)
    assert not repeated_css('<style>' + footer * 16 + '</style><body>')
    assert not repeated_css('<body>' + '<tr><td>data</td></tr>' * 200)
    assert not repeated_css('<style>' + ''.join(f'.x{n}{{width:{n}px}}' for n in range(60)))


def test_guard_checks_only_generated_text_at_intervals():
    calls = []
    def decode(tokens, **kwargs):
        calls.append(tokens.shape[1])
        return ['<style>' + '.a{color:red}' * 48]
    guard = CSSRepeatStop(SimpleNamespace(batch_decode=decode), 100)
    assert not guard(torch.zeros((1, 600), dtype=torch.long), None)
    assert guard(torch.zeros((1, 612), dtype=torch.long), None)
    assert calls == [512]
    assert guard.reason == 'repeated_css_generation'


def test_presence_penalty_is_once_per_generated_token_and_ignores_prompt():
    penalty = GeneratedPresencePenalty(2)
    original = torch.zeros((1, 5))
    result = penalty(torch.tensor([[0, 1, 2, 2, 3]]), original)
    assert result.tolist() == [[0, 0, -1.5, -1.5, 0]]
    assert original.sum() == 0


def test_sampling_options_and_non_html_scope():
    args = SimpleNamespace(task='generate-html', temperature=0.7)
    opts, guard = html_generation_options(args, None, 10, qwen=True)
    assert opts['top_p'] == .8 and opts['top_k'] == 20
    assert opts['stopping_criteria'] == [guard]
    assert len(opts['logits_processor']) == 1
    args.temperature = 0
    opts, _ = html_generation_options(args, None, 10, qwen=True)
    assert 'top_p' not in opts and 'logits_processor' not in opts
    args.task = 'extract-frames'
    assert html_generation_options(args, None, 10) == ({}, None)

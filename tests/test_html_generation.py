from types import SimpleNamespace
import torch
from framediff.html_generation import GeneratedPresencePenalty
from framediff.vlm import html_generation_options


def test_presence_penalty_is_once_per_generated_token_and_ignores_prompt():
    penalty = GeneratedPresencePenalty(2)
    original = torch.zeros((1, 5))
    result = penalty(torch.tensor([[0, 1, 2, 2, 3]]), original)
    assert result.tolist() == [[0, 0, -1.5, -1.5, 0]]
    assert original.sum() == 0


def test_sampling_options_and_non_html_scope():
    args = SimpleNamespace(task='generate-html', temperature=0.7)
    opts = html_generation_options(args, None, 10, qwen=True)
    assert opts['top_p'] == .8 and opts['top_k'] == 20
    assert 'stopping_criteria' not in opts
    assert len(opts['logits_processor']) == 1
    args.temperature = 0
    opts = html_generation_options(args, None, 10, qwen=True)
    assert 'top_p' not in opts and 'logits_processor' not in opts
    args.task = 'extract-frames'
    assert html_generation_options(args, None, 10) == {}

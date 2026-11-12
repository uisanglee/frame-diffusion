from types import SimpleNamespace
import torch
import pytest
from framediff.html_generation import CSSRepeatStop, GeneratedPresencePenalty, HTMLCompletionStop, repeated_css
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
    assert isinstance(opts['stopping_criteria'][0], HTMLCompletionStop)
    assert isinstance(opts['stopping_criteria'][1], CSSRepeatStop)
    assert len(opts['logits_processor']) == 1
    args.temperature = 0
    opts = html_generation_options(args, processor, 10, qwen=True)
    assert 'top_p' not in opts and 'logits_processor' not in opts
    assert isinstance(opts['stopping_criteria'][0], HTMLCompletionStop)
    assert isinstance(opts['stopping_criteria'][1], CSSRepeatStop)
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


class CharDecoder:
    def decode(self, ids, **kwargs):
        assert not kwargs['skip_special_tokens']
        return ''.join(chr(n) for n in ids)


@pytest.mark.parametrize('body', [
    '<html><style>.x{width:12px}</style><body>ok</body></html>',
    '<HTML lang="en"><body>ok</body></HTML >',
    '<think>Example <html>draft</html> is not final.</think><html><body>final</body></html>',
    '<html><body><!-- </html> --><script>const x="</html>";</script>ok</body></html>',
    '<html><body><textarea></html></textarea><div title="</html>">x</div></body></html>',
    '<html><style>/* </html> */</style><body>' + 'long text ' * 100 + '</body></html>',
])
def test_document_completion_only_stops_at_actual_end(body):
    prompt = '<html><body>current HTML</body></html> Please repair.'
    guard = HTMLCompletionStop(CharDecoder(), len(prompt))
    for index in range(1, len(body)+1):
        ids = torch.tensor([[ord(c) for c in prompt + body[:index]]])
        assert guard(ids) == (index == len(body))
    assert guard.reason == 'html_complete'


def test_document_completion_respects_template_reasoning_prefix():
    prompt = '<|assistant|><think>\n'
    body = 'a draft <html></html> remains reasoning</think><html><body>x</body></html>'
    guard = HTMLCompletionStop(CharDecoder(), len(prompt))
    for index in range(1, len(body)+1):
        assert guard(torch.tensor([[ord(c) for c in prompt + body[:index]]])) == (index == len(body))


def test_incomplete_document_does_not_stop_and_decoding_stays_bounded():
    decoded_lengths = []
    class TrackingDecoder(CharDecoder):
        def decode(self, ids, **kwargs):
            decoded_lengths.append(len(ids))
            return super().decode(ids, **kwargs)
    guard = HTMLCompletionStop(TrackingDecoder(), 0)
    body = '<html><body>' + 'x'*1500
    for index in range(1, len(body)+1):
        assert not guard(torch.tensor([[ord(c) for c in body[:index]]]))
    assert guard.reason is None
    assert max(decoded_lengths) <= 64


@pytest.mark.parametrize('prompt', ['', '<|assistant|><think>\n'])
def test_non_monotonic_decoder_rebuilds_parser_without_premature_stop(prompt):
    class RevisingDecoder(CharDecoder):
        def decode(self, ids, **kwargs):
            text = super().decode(ids, **kwargs)
            # Simulate a tokenizer revising the previously decoded suffix.
            if text.endswith('AB'):
                return text[:-2] + 'Z'
            return text
    reasoning = 'draft <html></html></think>' if prompt else ''
    body = reasoning + '<html><body>' + 'x'*90 + 'ABCD<script>"</html>"</script></body></html>'
    guard = HTMLCompletionStop(RevisingDecoder(), len(prompt))
    for index in range(1, len(body)+1):
        ids = torch.tensor([[ord(c) for c in prompt + body[:index]]])
        assert guard(ids) == (index == len(body))
    assert guard.reason == 'html_complete'


def test_revised_prefix_does_not_invent_document_opening():
    class Decoder:
        def decode(self, ids, **kwargs):
            return {0: '', 1: '<html>', 2: 'plain text', 3: 'plain text</html>'}[len(ids)]
    guard = HTMLCompletionStop(Decoder(), 0)
    for n in range(1, 4):
        assert not guard(torch.zeros((1, n), dtype=torch.long))
    assert not guard.parser.document

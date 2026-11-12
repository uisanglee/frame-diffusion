"""Compact HTML instructions and bounded guards for generated HTML."""
import re
from collections import Counter
from html.parser import HTMLParser

CONTRACT = 'html-completion-stop-v5'
STATIC_HTML_PROMPT = (
    'The renderer disables JavaScript and external network requests. '
    'Put all visible page content directly in the HTML body and all CSS in a <style> block or inline styles. '
    'Do not use React, JSX, Babel, Tailwind CDN, external stylesheets, or JavaScript to create the page. '
    'Use only the supplied local image resources or existing data URIs. '
    'Return only the HTML document, without reasoning, explanations, or Markdown fences.\n')
PROMPT = ('Return one complete HTML document ending with </html>. Keep CSS compact: '
          'use concise markup, reuse shared classes and grouped selectors, and omit comments '
          'and avoid introducing redundant wrappers. Preserve all required content. Do not repeat identical CSS rules '
          'or enumerate hundreds of nth-child selectors; use reusable or periodic rules. '
          'Finish the body and close the document within the output budget.\n'
          'Return a complete, runnable HTML document.\n' + STATIC_HTML_PROMPT)


def viewport_instruction(viewport):
    """Describe the evaluation viewport, independent of model image resizing."""
    width, height = map(int, viewport)
    if width <= 0 or height <= 0:
        raise ValueError('Viewport dimensions must be positive')
    return (f'The target webpage is evaluated at a viewport of {width} by {height} '
            'CSS pixels. Match the reference layout at this viewport.\n')


class _DocumentCompletionParser(HTMLParser):
    # Closing-tag text in JS, CSS, comments and text-only elements is not a
    # document terminator. HTMLParser keeps these contents out of tag callbacks.
    CDATA_CONTENT_ELEMENTS = ('script', 'style', 'textarea', 'title', 'xmp',
                              'iframe', 'noembed', 'noframes')

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.thinking = False
        self.document = False
        self.complete = False

    def handle_starttag(self, tag, attrs):
        if tag == 'think':
            self.thinking = True
        elif tag == 'html' and not self.thinking:
            self.document = True

    def handle_endtag(self, tag):
        if tag == 'think':
            self.thinking = False
        elif tag == 'html' and self.document and not self.thinking:
            self.complete = True


class HTMLCompletionStop:
    """Batch-one generation stops at a real closing HTML tag, before EOS.

    Decode bounded incremental chunks, not the entire growing answer each step.
    Retain special tokens so a draft inside <think> cannot trigger early stop.
    This detects document closure; downstream validation still checks content.
    """
    def __init__(self, processor, prompt_tokens):
        self.processor = processor
        self.prompt_tokens = prompt_tokens
        self.parser = _DocumentCompletionParser()
        self.reason = None
        self.decode_start = prompt_tokens
        self.decoded = ''
        self.seen_tokens = prompt_tokens
        self.initialized = False
        self.initial_thinking = False

    def _decode(self, ids):
        options = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
        decoder = getattr(self.processor, 'decode', None)
        if decoder is None:
            decoder = getattr(getattr(self.processor, 'tokenizer', None), 'decode', None)
        if decoder is not None:
            return decoder(ids, **options)
        return self.processor.batch_decode([ids], **options)[0]

    def __call__(self, input_ids, scores=None, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError('HTMLCompletionStop requires batch size 1')
        if self.reason:
            return True
        if not self.initialized:
            # Some chat templates open the reasoning span in the assistant prefix.
            prefix = self._decode(input_ids[0, max(0, self.prompt_tokens-32):self.prompt_tokens].tolist())
            self.parser.thinking = prefix.rstrip().endswith('<think>')
            self.initial_thinking = self.parser.thinking
            self.initialized = True
        end = input_ids.shape[1]
        if end <= self.seen_tokens:
            return False
        text = self._decode(input_ids[0, self.decode_start:end].tolist())
        # A partial UTF-8 token can decode as U+FFFD and change on the next step.
        # Defer that suffix until it is stable rather than feeding it twice.
        stable = text.rstrip('\ufffd')
        if not stable.startswith(self.decoded):
            # Token decoding need not be prefix-monotonic (byte fallback,
            # whitespace handling, or a chunk starting inside a character).
            # Rebuild from the authoritative full generated text only on this
            # exceptional path; do not retain stale parser/CDATA/think state.
            self.decode_start = self.prompt_tokens
            text = self._decode(input_ids[0, self.prompt_tokens:end].tolist())
            stable = text.rstrip('\ufffd')
            self.parser = _DocumentCompletionParser()
            self.parser.thinking = self.initial_thinking
            self.decoded = ''
        self.parser.feed(stable[len(self.decoded):])
        self.decoded = stable
        self.seen_tokens = end
        if stable == text and end - self.decode_start >= 64:
            # Retain token overlap: SentencePiece can omit the leading space
            # when a new decoding chunk starts with a word token.
            self.decode_start = end - 16
            self.decoded = self._decode(input_ids[0, self.decode_start:end].tolist())
        if self.parser.complete:
            self.reason = 'html_complete'
        return self.reason is not None


def repeated_css(text):
    """Detect sustained exact rule duplication in an unclosed, simple style block."""
    lower = text.lower()
    start = lower.rfind('<style')
    if start < 0 or lower.rfind('</style') > start:
        return False
    css = re.sub(r'^<style[^>]*>', '', text[start:], count=1, flags=re.I)
    # Numbered selectors and nested conditional rules can intentionally repeat
    # declarations. Treat them as legitimate rather than normalizing selectors.
    if re.search(r'@(?:media|supports|container|layer|keyframes)\b', css, flags=re.I):
        return False
    rules = re.findall(r'([^{}]+)\{([^{}]*)\}', css)[-48:]
    if len(rules) < 24:
        return False
    normalized = []
    for selector, body in rules:
        normalized.append(re.sub(r'\s+', '', selector + '{' + body + '}'))
    counts = Counter(normalized)
    return len(counts) <= 4 and min(counts.values()) >= 6


class CSSRepeatStop:
    """Inspect only a bounded generated suffix every 512 tokens, never the prompt."""
    def __init__(self, processor, prompt_tokens, interval=512, window=4096):
        self.processor, self.prompt_tokens = processor, prompt_tokens
        self.interval, self.window = interval, window
        self.last_check = 0
        self.in_style = False
        self.reason = None

    def __call__(self, input_ids, scores, **kwargs):
        length = input_ids.shape[1] - self.prompt_tokens
        if length < 1024 or length - self.last_check < self.interval:
            return False
        self.last_check = length
        suffix = input_ids[:, max(self.prompt_tokens, input_ids.shape[1] - self.window):]
        text = self.processor.batch_decode(suffix, skip_special_tokens=True)[0]
        lower = text.lower()
        if '<style' in lower or '</style' in lower:
            self.in_style = lower.rfind('<style') > lower.rfind('</style')
        elif self.in_style:
            text = '<style>' + text
        if self.in_style and repeated_css(text):
            self.reason = 'repeated_css_generation'
        return self.reason is not None


class GeneratedPresencePenalty:
    """Presence penalty on generated tokens only, without penalizing the prompt."""
    def __init__(self, prompt_tokens, penalty=1.5):
        self.prompt_tokens, self.penalty = prompt_tokens, penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_tokens:]
        if generated.shape[1]:
            scores = scores.clone()
            values = scores.gather(1, generated) - self.penalty
            scores.scatter_(1, generated, values)
        return scores

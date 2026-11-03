"""Compact HTML instructions and bounded guards for generated HTML."""
import re
from collections import Counter

CONTRACT = 'compact-html-css-viewport-v2'
PROMPT = ('Return one complete HTML document ending with </html>. Keep CSS compact: '
          'use concise markup, reuse shared classes and grouped selectors, and omit comments '
          'and avoid introducing redundant wrappers. Preserve all required content. Do not repeat identical CSS rules '
          'or enumerate hundreds of nth-child selectors; use reusable or periodic rules. '
          'Finish the body and close the document within the output budget.\n')


def viewport_instruction(viewport):
    """Describe the evaluation viewport, independent of model image resizing."""
    width, height = map(int, viewport)
    if width <= 0 or height <= 0:
        raise ValueError('Viewport dimensions must be positive')
    return (f'The target webpage is evaluated at a viewport of {width} by {height} '
            'CSS pixels. Match the reference layout at this viewport.\n')


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

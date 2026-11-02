"""Bounded, conservative checks for runaway HTML generation (single sample)."""
import re
from collections import Counter

CONTRACT = 'compact-html-css-repeat-v1'
PROMPT = ('Return one complete HTML document ending with </html>. Keep CSS compact: '
          'reuse shared classes and grouped selectors. Do not repeat identical CSS rules '
          'or enumerate hundreds of nth-child selectors; use reusable or periodic rules. '
          'Finish the body and close the document within the output budget.\n')


def repeated_css(text):
    # Inspect only the current unclosed style block; normal repeated HTML rows are allowed.
    start = text.lower().rfind('<style')
    if start < 0 or '</style' in text[start:].lower():
        return False
    css = re.sub(r'^<style[^>]*>', '', text[start:], count=1, flags=re.I)
    rules = re.findall(r'([^{}]+)\{([^{}]*)\}', css)[-48:]
    if len(rules) < 24:
        return False
    normalized = []
    for selector, body in rules:
        selector = re.sub(r':nth-child\(\d+\)', ':nth-child(N)', selector)
        normalized.append(re.sub(r'\s+', '', selector + '{' + body + '}'))
    counts = Counter(normalized)
    # Require a sustained tail dominated by at most four repeated rules.
    return len(counts) <= 4 and min(counts.values()) >= 6


class CSSRepeatStop:
    def __init__(self, processor, prompt_tokens):
        self.processor = processor
        self.prompt_tokens = prompt_tokens
        self.last_check = 0
        self.reason = None
        self.in_style = False

    def __call__(self, input_ids, scores, **kwargs):
        length = input_ids.shape[1] - self.prompt_tokens
        if length < 512 or length - self.last_check < 128:
            return False
        self.last_check = length
        # Bounded CPU decoding every 128 tokens, not per-token full-prefix decoding.
        tokens = input_ids[:, self.prompt_tokens:]
        text = self.processor.batch_decode(tokens[:, -8192:], skip_special_tokens=True)[0]
        lower = text.lower()
        if '<style' in lower or '</style' in lower:
            self.in_style = lower.rfind('<style') > lower.rfind('</style')
        elif self.in_style:
            text = '<style>' + text
        if repeated_css(text):
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

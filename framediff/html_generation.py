"""Compact HTML instructions and generated-token presence penalty."""

CONTRACT = 'compact-html-css-repeat-v1'
PROMPT = ('Return one complete HTML document ending with </html>. Keep CSS compact: '
          'reuse shared classes and grouped selectors. Do not repeat identical CSS rules '
          'or enumerate hundreds of nth-child selectors; use reusable or periodic rules. '
          'Finish the body and close the document within the output budget.\n')


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

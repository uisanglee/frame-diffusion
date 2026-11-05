def compatible_browser_digest(sha):
    """Keep caches for the audited context-recycling change only."""
    if sha == '285ecaab41ff4079a26e8615d3213e9cfb620bb89cac817e6df0f64cf37d6dc9':
        return '4e3e43aef0e8deb0bae424d2dc5e8a438d86d7ec5149c09d836c48cdde3b8882'
    return sha  # Future rendering changes must still invalidate caches.


def compatible_visual_digest(sha):
    """Removing the unused action_size helper leaves visual labels unchanged."""
    if sha == 'bada89efed13ab8e51ecf3ee1659ffd4f0631dda6e88a0b23fba91a107507360':
        return '81ac6d0d50303868ee4f3958d9dec22818c75c025926e72153b0da3f06dc884f'
    return sha

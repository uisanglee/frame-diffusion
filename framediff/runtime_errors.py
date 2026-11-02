"""Do not turn a broken inference runtime into cached benchmark page failures."""


def raise_if_fatal(error):
    text = str(error).lower()
    if isinstance(error, (MemoryError, ImportError)) or any(term in text for term in (
            'cuda', 'out of memory', 'cublas', 'cudnn', 'device-side assert')):
        raise error

"""Do not turn a broken inference runtime into cached benchmark page failures."""

class InferenceWorkerError(RuntimeError):
    """A dedicated inference worker failed; stop rather than score a missing model."""


def raise_if_fatal(error):
    text = str(error).lower()
    if isinstance(error, (MemoryError, ImportError, InferenceWorkerError)) or any(term in text for term in (
            'cuda', 'out of memory', 'cublas', 'cudnn', 'device-side assert')):
        raise error

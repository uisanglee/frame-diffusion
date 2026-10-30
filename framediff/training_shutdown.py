"""Let training context managers clean up workers on a termination request."""
from functools import wraps
import signal
import threading


def cleanup_on_termination(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            return function(*args, **kwargs)
        previous = {}

        def terminate(signum, frame):
            # A second TERM/INT must not interrupt descendant cleanup.
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            raise SystemExit(128 + signum)

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, terminate)
            return function(*args, **kwargs)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return wrapped

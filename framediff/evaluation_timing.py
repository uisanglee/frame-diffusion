"""Wall-clock evaluation diagnostics; nested stages report exclusive time too."""
from contextlib import contextmanager
import json
import time


class EvaluationTiming:
    def __init__(self, **context):
        self.context = context
        self.seconds = {}
        self.inclusive_seconds = {}
        self.calls = {}
        self._stack = []

    def event(self, stage, status, **extra):
        print(json.dumps({'evaluation_stage': stage, 'status': status,
                          **self.context, **extra}), flush=True)

    @contextmanager
    def stage(self, name):
        self.event(name, 'started')
        frame = [time.perf_counter(), 0.]
        self._stack.append(frame)
        status = 'completed'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            elapsed = time.perf_counter() - frame[0]
            self._stack.pop()
            if self._stack:
                self._stack[-1][1] += elapsed
            exclusive = max(0., elapsed - frame[1])
            self.seconds[name] = self.seconds.get(name, 0.) + exclusive
            self.inclusive_seconds[name] = self.inclusive_seconds.get(name, 0.) + elapsed
            self.calls[name] = self.calls.get(name, 0) + 1
            self.event(name, status, seconds=elapsed, exclusive_seconds=exclusive)

    def report(self):
        return {'stage_seconds': dict(self.seconds),
                'stage_inclusive_seconds': dict(self.inclusive_seconds),
                'stage_calls': dict(self.calls)}

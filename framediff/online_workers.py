"""Bounded, ordered online jobs with independently restartable browser workers."""
from collections import deque
from concurrent.futures import TimeoutError
from multiprocessing.connection import wait
from types import SimpleNamespace
import multiprocessing as mp
import math
import time


def _run(connection, initializer, initargs, stop):
    from .cache_lock import set_progress
    try:
        if initializer: initializer(*initargs, stop)
        while True:
            command=connection.recv()
            if command is None: break
            job,function,args=command
            def progress(stage, path):
                connection.send(('progress',job,time.monotonic(),stage,path))
            set_progress(progress)
            try:
                value=function(*args)
                connection.send(('result',job,value,None))
            except Exception as exc:
                # Send only the error, not its traceback retaining browser state.
                try: connection.send(('result',job,None,exc))
                except (TypeError, AttributeError):
                    connection.send(('result',job,None,RuntimeError(f'{type(exc).__name__}: {exc}')))
            finally: set_progress(None)
    except (EOFError, BrokenPipeError): pass
    finally: connection.close()


class _Job:
    def __init__(self,pool,number,function,args):
        self.pool=pool;self.number=number;self.function=function;self.args=args
        self.done=False;self.value=None;self.error=None
        self.stage='queued';self.mark=time.monotonic();self.active_s=0.;self.wait_s=0.;self.path=None

    def result(self,timeout=None):
        return self.pool.result(self,timeout)

    def transition(self,stage,now):
        elapsed=max(0.,now-self.mark)
        if self.stage=='cache_wait': self.wait_s+=elapsed
        elif self.stage=='working': self.active_s+=elapsed
        self.stage=stage;self.mark=now


class OnlineWorkerPool:
    def __init__(self,workers,initializer=None,initargs=(),cache_wait_timeout=3600):
        if workers<1 or not math.isfinite(cache_wait_timeout) or cache_wait_timeout<=0:
            raise ValueError('Invalid online worker limits')
        self.context=mp.get_context('spawn');self.initializer=initializer;self.initargs=initargs
        self.cache_wait_timeout=cache_wait_timeout;self.queue=deque();self.slots=[];self.serial=0;self.closed=False
        try:
            for _ in range(workers): self.slots.append(self._start())
        except BaseException:
            self.shutdown();raise

    def _start(self):
        parent,child=self.context.Pipe();stop=self.context.Event()
        process=self.context.Process(target=_run,args=(child,self.initializer,self.initargs,stop))
        try: process.start()
        except BaseException:
            parent.close();child.close();raise
        child.close()
        return SimpleNamespace(process=process,connection=parent,stop=stop,job=None)

    def submit(self,function,*args):
        if self.closed: raise RuntimeError('Online worker pool is closed')
        job=_Job(self,self.serial,function,args);self.serial+=1;self.queue.append(job);self._dispatch()
        return job

    def _dispatch(self):
        for slot in self.slots:
            if slot.job is None and self.queue:
                job=self.queue.popleft();slot.job=job
                # Include worker startup, but never time waiting in the queue.
                job.transition('working',time.monotonic())
                slot.connection.send((job.number,job.function,job.args))

    def _receive(self,slot):
        try: message=slot.connection.recv()
        except (EOFError, OSError):
            if slot.job:
                slot.job.error=RuntimeError(f'Online worker {slot.process.pid} exited unexpectedly')
                slot.job.done=True
            return False
        kind,number,*payload=message;job=slot.job
        if job is None or job.number!=number: raise RuntimeError('Online worker result identity changed')
        if kind=='progress':
            now,stage,path=payload;job.transition(stage,now);job.path=path
        elif kind=='result':
            job.transition('complete',time.monotonic());job.value,job.error=payload;job.done=True;slot.job=None
        else: raise RuntimeError('Unknown online worker message')
        return True

    def result(self,job,timeout):
        while not job.done:
            ready=wait([s.connection for s in self.slots],timeout=.05)
            for slot in self.slots:
                if slot.connection in ready:
                    # A GPU/validation interval can leave several status events
                    # buffered. Observe the latest state before checking clocks.
                    while slot.connection.poll():
                        if not self._receive(slot):break
            self._dispatch()
            if job.done: break
            now=time.monotonic()
            if job.stage=='cache_wait':
                if job.wait_s+now-job.mark>=self.cache_wait_timeout:
                    raise TimeoutError(f'Shared cache lock wait exceeded {self.cache_wait_timeout}s: {job.path}')
            elif job.stage=='working' and timeout is not None and job.active_s+now-job.mark>=timeout:
                raise TimeoutError(f'Online sample active work exceeded {timeout}s')
        if job.error is not None: raise job.error
        return job.value

    @staticmethod
    def _terminate(slots):
        from .preparation_workers import stop_executor
        # Reuse ancestry-aware cleanup on only the selected workers.
        processes=[s.process for s in slots if s.process.is_alive()]
        if processes:
            stop_executor(SimpleNamespace(shutdown=lambda **kwargs:None),processes)
        for slot in slots:
            slot.process.join(timeout=2);slot.connection.close()

    def restart(self,job,retry=True):
        if job.done: return
        for i,slot in enumerate(self.slots):
            if slot.job is job:
                slot.stop.set();self._terminate([slot]);self.slots[i]=self._start()
                if retry:
                    job.stage='queued';job.mark=time.monotonic();job.active_s=0.;job.wait_s=0.
                    self.queue.appendleft(job)
                else: job.done=True
                self._dispatch();return
        raise RuntimeError('Timed-out online job has no active worker')

    def shutdown(self):
        if self.closed:return
        self.closed=True;self.queue.clear()
        for slot in self.slots:
            slot.stop.set()
            try: slot.connection.send(None)
            except (BrokenPipeError, EOFError, OSError): pass
        # Give cooperative samplers time to close their browser leases.
        deadline=time.monotonic()+2
        for slot in self.slots: slot.process.join(timeout=max(0.,deadline-time.monotonic()))
        self._terminate(self.slots)

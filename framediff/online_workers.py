"""Bounded, ordered online jobs with independently restartable browser workers."""
from collections import deque
from concurrent.futures import TimeoutError
from types import SimpleNamespace
import queue
import threading
import signal
import os
import multiprocessing as mp
import math
import time


def _run(connection, stop):
    from .cache_lock import set_progress
    try:
        initializer,initargs=connection.recv()
        connection.send(('started',os.getpid()))
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
                try:connection.send(('result',job,value,None))
                finally:del value  # Do not retain the previous PNG/DOM while producing the next.
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
        self.stage='queued';self.mark=time.monotonic();self.active_s=0.;self.wait_s=0.;self.path=None;self.target=None

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
        from .worker_lifecycle import supervise
        parent,child=self.context.Pipe();stop=self.context.Event()
        process=self.context.Process(target=supervise,args=(child,stop))
        try: process.start()
        except BaseException:
            parent.close();child.close();raise
        child.close()
        slot=SimpleNamespace(process=process,connection=parent,stop=stop,job=None,
                             worker_pid=None,inbox=queue.Queue(maxsize=8),reader_stop=threading.Event())
        def read_messages():
            while not slot.reader_stop.is_set():
                try:message=parent.recv()
                except (EOFError,OSError) as exc:message=exc
                while not slot.reader_stop.is_set():
                    try:slot.inbox.put(message,timeout=.05);break
                    except queue.Full:pass
                if isinstance(message,BaseException):break
        slot.reader=threading.Thread(target=read_messages,daemon=True)
        slot.reader.start()
        try:parent.send((self.initializer,self.initargs))
        except BaseException:
            self._terminate([slot]);raise
        return slot

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
        try: message=slot.inbox.get_nowait()
        except queue.Empty:return False
        if isinstance(message,BaseException):
            if slot.job:
                slot.job.error=RuntimeError(f'Online worker {slot.process.pid} exited unexpectedly')
                slot.job.done=True
            return False
        if message[0]=='started':
            slot.worker_pid=message[1];return True
        kind,number,*payload=message;job=slot.job
        if job is None or job.number!=number: raise RuntimeError('Online worker result identity changed')
        if kind=='progress':
            now,stage,path=payload
            if stage=='target':job.target=path
            else:job.transition(stage,now);job.path=path
        elif kind=='result':
            job.transition('complete',time.monotonic());job.value,job.error=payload;job.done=True;slot.job=None
        else: raise RuntimeError('Unknown online worker message')
        return True

    def result(self,job,timeout):
        while not job.done:
            received=False
            for slot in self.slots:
                while self._receive(slot):received=True
            self._dispatch()
            if job.done: break
            now=time.monotonic()
            if job.stage=='cache_wait':
                if job.wait_s+now-job.mark>=self.cache_wait_timeout:
                    raise TimeoutError(f'Shared cache lock wait exceeded {self.cache_wait_timeout}s: {job.path}')
            elif job.stage=='working' and timeout is not None and job.active_s+now-job.mark>=timeout:
                raise TimeoutError(f'Online sample active work exceeded {timeout}s')
            if not received:time.sleep(.01)
        if job.error is not None: raise job.error
        return job.value

    @staticmethod
    def _terminate(slots):
        from .preparation_workers import stop_executor
        # Let the owner reap adopted children even if its worker already died.
        for slot in slots:
            slot.reader_stop.set()
            if slot.process.is_alive():
                try:os.kill(slot.process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+6
        for slot in slots:slot.process.join(timeout=max(0.,deadline-time.monotonic()))
        # Reuse ancestry-aware cleanup on only the selected workers.
        processes=[s.process for s in slots if s.process.is_alive()]
        if processes:
            stop_executor(SimpleNamespace(shutdown=lambda **kwargs:None),processes)
        for slot in slots:
            slot.process.join(timeout=2);slot.connection.close()
            slot.reader.join(timeout=1)
            slot.job=None
            while True:
                try:slot.inbox.get_nowait()
                except queue.Empty:break

    def restart(self,job,retry=True):
        if job.done: return
        for i,slot in enumerate(self.slots):
            if slot.job is job:
                slot.stop.set();self._terminate([slot]);self.slots[i]=self._start()
                if retry:
                    job.stage='queued';job.mark=time.monotonic();job.active_s=0.;job.wait_s=0.
                    job.target=None
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
        try:self._terminate(self.slots)
        finally:self.slots.clear()

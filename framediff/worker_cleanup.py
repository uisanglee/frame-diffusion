"""Reap only descendants of this executor's workers before killing workers."""
import os
import signal
import subprocess
import time


def process_table():
    output=subprocess.check_output(['ps','-axo','pid=,ppid=,lstart=,command='],text=True)
    table={}
    for line in output.splitlines():
        parts=line.split(None,7)
        if len(parts)==8:table[int(parts[0])]=(int(parts[1]),' '.join(parts[2:]))
    return table


def stop_executor(pool,processes):
    # Capture ancestry before any worker dies and its children are reparented.
    # Freeze workers first so queued work cannot launch a new browser while
    # the captured descendants are being reaped.
    try:
        for process in processes:
            if process.is_alive():
                try:os.kill(process.pid,signal.SIGSTOP)
                except ProcessLookupError:pass
        _stop_descendants(processes)
    finally:
        pool.shutdown(wait=False,cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
                try:os.kill(process.pid,signal.SIGCONT)
                except ProcessLookupError:pass
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():process.kill();process.join(timeout=2)


def _stop_descendants(processes):
    table=process_table();roots={p.pid for p in processes if p.pid and p.is_alive()}
    descendants=set();front=roots
    while front:
        front={pid for pid,(parent,_) in table.items() if parent in front}-descendants-roots
        descendants.update(front)
    tracked={pid:table[pid] for pid in descendants}
    def alive():
        current=process_table()
        return {pid for pid,identity in tracked.items() if pid in current and current[pid][1]==identity[1]}
    def send(pids,sig):
        for pid in pids:
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
    send(alive(),signal.SIGTERM)
    deadline=time.monotonic()+2
    while alive() and time.monotonic()<deadline:time.sleep(.05)
    send(alive(),signal.SIGKILL)
    # Give driver/browser parents time to reap their subprocesses.
    deadline=time.monotonic()+2
    while alive() and time.monotonic()<deadline:time.sleep(.05)
    remaining=alive()
    if remaining:raise RuntimeError(f'Worker descendants did not exit: {sorted(remaining)}')

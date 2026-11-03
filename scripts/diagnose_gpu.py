#!/usr/bin/env python3
"""Collect bounded, read-only Linux GPU diagnostics; never reset or stress GPUs."""
import argparse
import csv
import datetime
import os
from pathlib import Path
import re
import subprocess
import time


def collect(command, log, timeout=20):
    log.write('\n$ ' + ' '.join(command) + '\n')
    log.flush()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        output = result.stdout.decode(errors='replace')
        log.write(output + f'\n[exit={result.returncode}]\n')
        return output if result.returncode == 0 else ''
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b''
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        log.write(output + f'\n[TIMEOUT after {timeout}s; status unknown]\n')
    except OSError as error:
        log.write(f'[unavailable: {error}]\n')
    finally:
        log.flush()
    return ''


def pci_address(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value.strip())
    if not match:
        raise ValueError(f'Unexpected PCI address: {value!r}')
    domain, bus, device, function = match.groups()
    return f'{int(domain, 16):04x}:{bus.lower()}:{device.lower()}.{function}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=3, help='Physical nvidia-smi index, not remapped CUDA index')
    parser.add_argument('--out', type=Path, default=Path('log/gpu3-diagnosis.log'))
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves earlier failure evidence.
    with args.out.open('x') as log:
        log.write(f'GPU diagnostic snapshot: {datetime.datetime.now().astimezone().isoformat()}\n')
        log.write(f'Requested physical GPU index: {args.gpu}\nRead-only; no stress test or GPU reset.\n')
        admin = [] if os.geteuid() == 0 else ['sudo', '-n']
        for command in [['uname', '-a'], ['uptime'], ['free', '-h'], ['df', '-h', '/'],
                        ['cat', '/proc/driver/nvidia/version']]:
            collect(command, log)
        collect(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,driver_version', '--format=csv'], log)
        identity = collect(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid,pci.bus_id',
                            '--format=csv,noheader,nounits'], log)
        target = str(args.gpu)
        if identity.strip():
            try:
                uuid, bus = next(csv.reader(identity.strip().splitlines()))
                if not uuid.strip().startswith('GPU-'):
                    raise ValueError('Missing GPU UUID')
                target = uuid.strip()
                address = pci_address(bus)
                log.write(f'Physical target pinned to {target}, PCI {address}\n')
                collect(admin + ['lspci', '-s', address, '-vvnn'], log)
                device = Path('/sys/bus/pci/devices') / address
                for name in ('current_link_speed', 'current_link_width', 'max_link_speed',
                             'max_link_width', 'aer_dev_correctable', 'aer_dev_nonfatal', 'aer_dev_fatal'):
                    path = device / name
                    if path.exists():
                        collect(['cat', str(path)], log)
            except (ValueError, StopIteration) as error:
                log.write(f'[identity parsing failed: {error}]\n')
        collect(['nvidia-smi', '-q', '-i', target], log)
        collect(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], log)
        for sample in range(3):
            log.write(f'\nSnapshot {sample + 1}: {datetime.datetime.now().astimezone().isoformat()}\n')
            collect(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total', '--format=csv'], log)
            if sample < 2:
                time.sleep(2)
        collect(['sensors'], log)
        collect(['lspci', '-nn'], log)
        collect(admin + ['journalctl', '--list-boots', '--no-pager'], log)
        errors = re.compile(r'NVRM|Xid|fallen off|AER|PCIe Bus Error|out of memory|oom|killed process|panic|watchdog|lockup|thermal|hardware error|mce|I/O error', re.I)
        for boot in ('0', '-1'):
            kernel = collect(admin + ['journalctl', '-k', '-b', boot, '--no-pager', '-o', 'short-iso'], log)
            log.write(f'\n=== Boot {boot}: keyword matches (boot initialization messages may also match) ===\n')
            matches = [line for line in kernel.splitlines() if errors.search(line)]
            log.write('\n'.join(matches) + '\n' if matches else 'No matching lines retrieved; this does NOT establish hardware health.\n')
            collect(admin + ['journalctl', '-b', boot, '-n', '150', '--no-pager', '-o', 'short-iso'], log)
        collect(admin + ['ls', '-lah', '/sys/fs/pstore', '/var/lib/systemd/pstore'], log)
        log.write('\nMissing commands, permissions, and timeouts are recorded above. Software logs cannot directly certify PSU health.\n')
    print(f'Diagnostic log saved: {args.out.resolve()}')


if __name__ == '__main__':
    main()

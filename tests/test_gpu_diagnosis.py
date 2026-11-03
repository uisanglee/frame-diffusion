import importlib.util
import io
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('gpu_diagnosis', Path(__file__).parents[1] / 'scripts/diagnose_gpu.py')
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def test_pci_domain_normalization():
    assert diagnosis.pci_address('00000000:41:00.0') == '0000:41:00.0'
    with pytest.raises(ValueError):
        diagnosis.pci_address('unexpected')


@pytest.mark.parametrize('error', [FileNotFoundError('missing'), subprocess.TimeoutExpired(['nvidia-smi'], 20, output=b'partial')])
def test_unavailable_gpu_does_not_abort_collection(error):
    log = io.StringIO()
    with patch.object(diagnosis.subprocess, 'run', side_effect=error):
        assert diagnosis.collect(['nvidia-smi'], log) == ''
    assert 'TIMEOUT' in log.getvalue() or 'unavailable' in log.getvalue()


def test_permission_failure_is_not_reported_as_success():
    log = io.StringIO()
    with patch.object(diagnosis.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, b'Permission denied')):
        assert diagnosis.collect(['journalctl'], log) == ''
    assert 'Permission denied' in log.getvalue()
    assert 'exit=1' in log.getvalue()

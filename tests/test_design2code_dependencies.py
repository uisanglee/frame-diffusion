import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).parents[1]/'baselines/design2code/check_dependencies.py'
spec = importlib.util.spec_from_file_location('design2code_dependency_check', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture_repo(tmp_path, code):
    folder = tmp_path/'utils/utils'
    folder.mkdir(parents=True)
    (folder/'dataset.py').write_text(code)
    (tmp_path/'finetune_demo').mkdir()
    (tmp_path/'finetune_demo/inference_design2code.py').write_text('import torch\n')
    return tmp_path


def test_audit_finds_nested_dataset_import_and_ignores_local_modules(tmp_path):
    root = fixture_repo(tmp_path, 'import os, jsonlines\nfrom .chat import chat\ndef f():\n from datasets import load_from_disk\n')
    imports = audit.source_imports(root)
    assert set(imports) == {'jsonlines', 'datasets', 'torch'}


def test_new_unmapped_dependency_fails_source_audit(tmp_path, monkeypatch):
    root = fixture_repo(tmp_path, 'import future_upstream_dependency\n')
    monkeypatch.setattr(sys, 'argv', ['audit', '--repo', str(root), '--source-only'])
    with pytest.raises(SystemExit, match='Unmapped upstream import future_upstream_dependency'):
        audit.main()


def test_all_import_failures_reported_together(tmp_path, monkeypatch):
    root = fixture_repo(tmp_path, 'import jsonlines\n')
    monkeypatch.setattr(sys, 'argv', ['audit', '--repo', str(root)])
    monkeypatch.setattr(audit, 'MODULES', {'jsonlines':'jsonlines', 'datasets':'datasets'})
    def fail(name):
        raise ImportError('missing '+name)
    monkeypatch.setattr(audit.importlib, 'import_module', fail)
    with pytest.raises(SystemExit) as caught:
        audit.main()
    assert 'jsonlines' in str(caught.value) and 'missing datasets' in str(caught.value)
    assert 'utils.models' in str(caught.value) and 'utils.utils' in str(caught.value)

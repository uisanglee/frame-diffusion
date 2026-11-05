"""Public HTML generation commands agree on the default output budget."""
import importlib

import pytest

from framediff.cli import main


@pytest.mark.parametrize('command,module,entry,required', [
    ('vlm', 'framediff.vlm', 'run', ['--task', 'generate-html']),
    ('run-design2code', 'framediff.benchmarks', 'run_design2code_vlm', ['--manifest', 'pages.jsonl']),
    ('web-prepare', 'framediff.web_experiment', 'prepare', ['--root', 'dataset']),
    ('web-abstract-self-revision', 'framediff.abstract_vlm_revision', 'revise', ['--data', 'pages.jsonl']),
])
def test_html_generation_cli_defaults_and_explicit_overrides(monkeypatch, command, module, entry, required):
    calls = []
    monkeypatch.setattr(importlib.import_module(module), entry, lambda args: calls.append(args))
    argv = [command, *required, '--out', 'unused']
    main(argv)
    main([*argv, '--max-new-tokens', '512'])
    assert [args.max_new_tokens for args in calls] == [16384, 512]

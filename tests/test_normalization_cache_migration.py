import pytest

from framediff.ir import read_json
from framediff.preparation_workers import write_json
from framediff.safe_sizes import COMPATIBLE_IMPLEMENTATION_SHAS
from framediff.tree_normalize import _guard_normalization_run


def test_only_audited_safe_size_implementation_migrates(tmp_path):
    old_sha=next(iter(COMPATIBLE_IMPLEMENTATION_SHAS))
    old={'kind':'css','normalization':'flow-preserved-safe-size-margin-v1',
         'safe_sizes_sha':old_sha,'seed':42}
    current={**old,'safe_sizes_sha':'current-fast-path-sha'}
    write_json(tmp_path/'config.json',old)
    _guard_normalization_run(tmp_path,current,True)
    saved=read_json(tmp_path/'config.json')
    assert saved['safe_sizes_sha']=='current-fast-path-sha'
    assert saved['compatible_safe_sizes_shas']==sorted([old_sha,'current-fast-path-sha'])


def test_cache_migration_rejects_semantic_setting_change(tmp_path):
    old_sha=next(iter(COMPATIBLE_IMPLEMENTATION_SHAS))
    write_json(tmp_path/'config.json',{'normalization':'flow-preserved-safe-size-margin-v1',
        'safe_sizes_sha':old_sha,'seed':42})
    with pytest.raises(ValueError,match='configuration/input changed'):
        _guard_normalization_run(tmp_path,{'normalization':'flow-preserved-safe-size-margin-v1',
            'safe_sizes_sha':'current-fast-path-sha','seed':7},True)

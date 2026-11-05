from types import SimpleNamespace
import sys
import types

import pytest

from framediff.websight_attention import (configure_attention, allow_optional_flash_imports,
                                         _only_guarded_flash_imports)


def test_missing_flash_uses_both_official_eager_paths_without_changing_model_settings():
    config = SimpleNamespace(
        _flash_attn_2_enabled=True,
        vision_config=SimpleNamespace(_flash_attn_2_enabled=True, image_size=960),
        perceiver_config=SimpleNamespace(resampler_n_latents=64),
        sliding_window=4096,
        max_position_embeddings=32768,
    )
    result, metadata = configure_attention(config, False)
    assert result is config
    assert not config._flash_attn_2_enabled
    assert not config.vision_config._flash_attn_2_enabled
    assert config.sliding_window == 4096
    assert config.max_position_embeddings == 32768
    assert config.vision_config.image_size == 960
    assert config.perceiver_config.resampler_n_latents == 64
    assert not hasattr(config.perceiver_config, '_flash_attn_2_enabled')
    assert metadata['attention_fallback_components'] == ['text', 'vision']
    assert metadata['published_legacy_attention_flags'] == {'text': True, 'vision': True}
    assert metadata['effective_legacy_attention_flags'] == {'text': False, 'vision': False}


def test_installed_flash_preserves_all_published_attention_flags():
    config = SimpleNamespace(
        _flash_attn_2_enabled=True,
        vision_config=SimpleNamespace(_flash_attn_2_enabled=False),
        perceiver_config=SimpleNamespace(_flash_attn_2_enabled=True),
    )
    _, metadata = configure_attention(config, True)
    assert config._flash_attn_2_enabled
    assert not config.vision_config._flash_attn_2_enabled
    assert config.perceiver_config._flash_attn_2_enabled
    assert not metadata['attention_compatibility_fallback']
    assert metadata['attention_fallback_components'] == []
    assert metadata['effective_legacy_attention_flags'] == metadata['published_legacy_attention_flags']


def test_missing_flash_preserves_existing_eager_and_handles_perceiver_flag():
    config = SimpleNamespace(
        _flash_attn_2_enabled=False,
        perceiver_config=SimpleNamespace(_flash_attn_2_enabled=True),
    )
    _, metadata = configure_attention(config, False)
    assert not config._flash_attn_2_enabled
    assert not config.perceiver_config._flash_attn_2_enabled
    assert metadata['attention_fallback_components'] == ['perceiver']


@pytest.mark.parametrize('source,expected', [
    ('if is_flash_attn_2_available():\n    from flash_attn import flash_attn_func\n', True),
    ('from flash_attn import flash_attn_func\n', False),
    ('if is_flash_attn_2_available():\n    pass\nelse:\n    import flash_attn\n', False),
    ('if is_flash_attn_2_available():\n    import flash_attn\nimport flash_attn.bert_padding\n', False),
    ('if some_other_condition():\n    import flash_attn\n', False),
    ('import torch\n', False),
])
def test_only_actual_availability_guard_makes_flash_optional(tmp_path, source, expected):
    path = tmp_path/'modeling_vmistral.py'
    path.write_text(source)
    assert _only_guarded_flash_imports(path) is expected
    other = tmp_path/'other_model.py'
    other.write_text(source)
    assert not _only_guarded_flash_imports(other)


def test_import_scanner_preserves_other_requirements_and_restores_after_error(monkeypatch, tmp_path):
    path = tmp_path/'vision.py'
    path.write_text('import einops\nif is_flash_attn_2_available():\n    import flash_attn\n')
    original = lambda filename: ['torch', 'einops', 'flash_attn']
    dynamic = SimpleNamespace(get_imports=original)
    transformers = types.ModuleType('transformers')
    transformers.dynamic_module_utils = dynamic
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    metadata = {'flash_attention_2_available': False, 'attention_compatibility_fallback': True}
    with pytest.raises(RuntimeError, match='failed loading model'):
        with allow_optional_flash_imports(metadata):
            assert dynamic.get_imports(path) == ['torch', 'einops']
            assert dynamic.get_imports(tmp_path/'other.py') == ['torch', 'einops', 'flash_attn']
            raise RuntimeError('failed loading model')
    assert dynamic.get_imports is original


def test_flash_enabled_does_not_modify_import_scanner(monkeypatch):
    original = lambda filename: ['torch', 'flash_attn']
    dynamic = SimpleNamespace(get_imports=original)
    transformers = types.ModuleType('transformers')
    transformers.dynamic_module_utils = dynamic
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    with allow_optional_flash_imports({'flash_attention_2_available': True,
                                       'attention_compatibility_fallback': False}):
        assert dynamic.get_imports is original

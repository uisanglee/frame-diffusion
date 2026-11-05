"""Select the published VMistral attention path supported by this environment.

The legacy remote model uses ``_flash_attn_2_enabled`` on its text and vision
configs. Modern Transformers ``attn_implementation`` does not control those
flags. Its own eager implementations use the same checkpoint parameters.
"""
import ast
from contextlib import contextmanager
from pathlib import Path


def _only_guarded_flash_imports(filename):
    """Recognize optional imports in the two published VMistral source files."""
    path = Path(filename)
    if path.name not in ('modeling_vmistral.py', 'vision.py'):
        return False
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except (OSError, SyntaxError, UnicodeError):
        return False
    flash_imports = []

    def visit(node, guarded=False):
        if isinstance(node, ast.Import):
            if any(alias.name.split('.')[0] == 'flash_attn' for alias in node.names):
                flash_imports.append(guarded)
        elif isinstance(node, ast.ImportFrom):
            if not node.level and (node.module or '').split('.')[0] == 'flash_attn':
                flash_imports.append(guarded)
        if isinstance(node, ast.If):
            condition = node.test
            optional = (isinstance(condition, ast.Call)
                        and isinstance(condition.func, ast.Name)
                        and condition.func.id == 'is_flash_attn_2_available'
                        and not condition.args and not condition.keywords)
            for child in node.body:
                visit(child, guarded or optional)
            for child in node.orelse:
                visit(child, guarded)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, guarded)

    visit(tree)
    return bool(flash_imports) and all(flash_imports)


@contextmanager
def allow_optional_flash_imports(attention_metadata):
    """Correct the old HF import scanner while loading the official eager path.

    Transformers 4.37 scans imports inside availability guards as mandatory.
    Only the guarded Flash Attention dependency is omitted; normal Python still
    executes the original downloaded model source, and all other checks remain.
    """
    if (attention_metadata.get('flash_attention_2_available')
            or not attention_metadata.get('attention_compatibility_fallback')):
        yield
        return
    from transformers import dynamic_module_utils

    original = dynamic_module_utils.get_imports

    def get_imports(filename):
        imports = original(filename)
        if 'flash_attn' in imports and _only_guarded_flash_imports(filename):
            return [name for name in imports if name != 'flash_attn']
        return imports

    dynamic_module_utils.get_imports = get_imports
    try:
        yield
    finally:
        dynamic_module_utils.get_imports = original


def configure_attention(config, flash_available):
    """Keep published flags where possible; report any explicit eager fallback."""
    nodes = {'text': config}
    for name in ('vision', 'perceiver'):
        node = getattr(config, f'{name}_config', None)
        if node is not None:
            nodes[name] = node
    published = {}
    effective = {}
    disabled = []
    for name, node in nodes.items():
        flag = getattr(node, '_flash_attn_2_enabled', None)
        if flag is None:
            continue
        published[name] = bool(flag)
        if flag and not flash_available:
            node._flash_attn_2_enabled = False
            disabled.append(name)
        effective[name] = bool(node._flash_attn_2_enabled)
    metadata = {
        'attention_contract': 'websight-published-attention-with-eager-fallback-v1',
        'flash_attention_2_available': bool(flash_available),
        'published_legacy_attention_flags': published,
        'effective_legacy_attention_flags': effective,
        'attention_compatibility_fallback': bool(disabled),
        'attention_fallback_components': disabled,
    }
    return config, metadata


def load_model_config(model_id, revision):
    """Read only the remote config, without downloading model weight shards."""
    from transformers import AutoConfig
    from transformers.utils import is_flash_attn_2_available

    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    return configure_attention(config, is_flash_attn_2_available())

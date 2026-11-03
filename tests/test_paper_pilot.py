import json
from pathlib import Path
import pytest
import framediff.paper_matrix as matrix


def test_pilot_freezes_ten_unique_pages_and_prefers_desktop(tmp_path, monkeypatch):
    script = Path('scripts/run_paper_pilot10.sh').read_text()
    selection = script.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    rows = [dict(id=f'{i:03d}/{view}', page_id=f'{i:03d}', view=view)
            for i in range(12) for view in ('iPad-Pro', 'default_1280-720')]
    monkeypatch.setattr(matrix, 'check_sources', lambda **kw: rows.copy())
    monkeypatch.setattr('sys.argv', ['pilot', 'webui', 'd2c', 'hard', str(tmp_path)])
    exec(selection, {})
    path = tmp_path/'inputs/webui.jsonl'
    saved = path.read_text()
    selected = [json.loads(line) for line in saved.splitlines()]
    assert len(selected) == len({r['page_id'] for r in selected}) == 10
    assert all(r['view'] == 'default_1280-720' for r in selected)
    for name in ('design2code', 'design2code-hard'):
        assert len((tmp_path/f'inputs/{name}.jsonl').read_text().splitlines()) == 10
    exec(selection, {})
    assert path.read_text() == saved
    rows[0]['page_id'] = 'changed'
    rows[1]['page_id'] = 'changed'
    with pytest.raises(ValueError, match='Pilot inputs changed'):
        exec(selection, {})

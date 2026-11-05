import json
from pathlib import Path

from framediff.inspection_storage import read_inspection


def test_existing_detailed_cache_reused_without_probe_json_reads(tmp_path, monkeypatch):
    path = tmp_path / 'existing.json'
    saved = dict(state=[[['10px', '']]], visible_candidates=[[1, 'width', '20px', '']],
                 inspection_diagnostics=dict(sites=[dict(probes=['large audit data'])]))
    original = json.dumps(saved)
    path.write_text(original)
    compact = read_inspection(path)
    assert compact == {k:v for k,v in saved.items() if k!='inspection_diagnostics'}
    assert path.read_text() == original  # Migration never rewrites successful caches.
    read = Path.read_text
    with monkeypatch.context() as patch:
        def forbid_original(p, *a, **kw):
            assert p != path, 'Training reread the detailed probe cache'
            return read(p, *a, **kw)
        patch.setattr(Path, 'read_text', forbid_original)
        assert read_inspection(path) == compact
    assert read_inspection(path, detailed=True) == saved
    saved['state'] = [[['30px', '']]]
    path.write_text(json.dumps(saved))
    assert read_inspection(path)['state'] == saved['state']


def test_online_worker_disables_html_export(monkeypatch):
    from types import SimpleNamespace
    from framediff import tree_online, browser
    calls=[]
    class Sampler:
        def __init__(self,*a,**kw):
            calls.append(kw);self.view_cache=SimpleNamespace(close=lambda:None)
        def close(self):pass
    monkeypatch.setattr(tree_online,'OnlineSampler',Sampler)
    monkeypatch.setattr(browser,'enable_worker_browser',lambda:None)
    monkeypatch.setattr(tree_online,'Finalize',lambda *a,**kw:None)
    monkeypatch.setattr(tree_online,'_sampler',None)
    tree_online._initialize([], 'abstract', 42, 4, 8, False, None)
    assert calls == [{'export_html':False}]

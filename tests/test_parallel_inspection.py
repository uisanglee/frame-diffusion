import threading
from concurrent.futures import ThreadPoolExecutor

from framediff import tree_online as online


def test_parallel_inspection_reuses_cache_and_invalidates_changed_html(tmp_path, monkeypatch):
    created = []
    probes = []
    barrier = threading.Barrier(2)

    class Browser:
        def __enter__(self):
            self.owner = threading.get_ident()
            created.append(self)
            self.closed = False
            return self
        def reset_context(self): assert self.owner == threading.get_ident()
        def __exit__(self, *args):
            assert self.owner == threading.get_ident()
            self.closed = True

    def inspect(browser, target, html, detail):
        assert browser.owner == threading.get_ident()
        probes.append(target['id'])
        if len(probes) <= 2: barrier.wait(timeout=5)
        return [[['', ''] for _ in range(6)]], dict(visible_candidates=[[1, 'width', '20px', '']],
                                     target_elements=[], visual_classes=[0])

    monkeypatch.setattr(online, 'HtmlBrowser', Browser)
    monkeypatch.setattr(online, 'inspect_candidates', inspect)
    monkeypatch.setattr(online, 'inspection_fingerprint', lambda: {'test': 1})
    rows = []
    for i in range(6):
        html = tmp_path / f'{i}.html'
        html.write_text(f'<html>{i}</html>')
        rows.append(dict(id=f'{i}/clean', split='train', group=str(i), source_sha=str(i),
                         viewport=[100, 100], current={'nodes': [{'id': 'root'}]},
                         target_image='unused.png', target_elements=[], target_html=str(html)))
    cache = tmp_path / 'cache'
    first = online.target_pool(rows + rows, cache_dir=cache, workers=2)
    assert len(probes) == 6
    assert len(created) == 2 and all(b.closed for b in created)
    # Both concurrent runs must use existing entries without opening a browser.
    monkeypatch.setattr(online, 'HtmlBrowser', lambda: (_ for _ in ()).throw(AssertionError('browser on cache hit')))
    with ThreadPoolExecutor(2) as executor:
        runs = list(executor.map(lambda _: online.target_pool(rows, cache_dir=cache, workers=2), range(2)))
    assert runs == [first, first]
    assert online.target_pool(rows, cache_dir=cache, workers=1) == first
    monkeypatch.setattr(online, 'HtmlBrowser', Browser)
    (tmp_path / '2.html').write_text('<html>changed</html>')
    assert online.target_pool(rows, cache_dir=cache, workers=2) == first
    assert probes.count('2') == 2 and len(probes) == 7
    assert all(b.closed for b in created)


def test_simultaneous_cache_miss_probes_once(tmp_path, monkeypatch):
    import time
    count = []
    class Browser:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def inspect(*args):
        count.append(1)
        time.sleep(.05)
        return [[['', ''] for _ in range(6)]], dict(visible_candidates=[], target_elements=[], visual_classes=[0])
    monkeypatch.setattr(online, 'HtmlBrowser', Browser)
    monkeypatch.setattr(online, 'inspect_candidates', inspect)
    html = tmp_path / 'target.html'
    html.write_text('<html>same</html>')
    target = dict(target_html=str(html), viewport=[100, 100], current={'nodes': [{'id': 'root'}]})
    def run(_):
        return list(online._inspection_results({'p': target}, tmp_path / 'cache', {'test': 1}, False, 1))
    with ThreadPoolExecutor(2) as executor:
        values = list(executor.map(run, range(2)))
    assert len(count) == 1
    assert sorted(v[0][1][2] for v in values) == [False, True]

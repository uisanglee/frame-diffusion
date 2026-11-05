import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from framediff.official_metrics import OfficialMetrics, EVALUATION_CONTRACT
from framediff.reevaluate_paper import refresh_cache


def test_scoring_never_calls_destructive_preprocess(tmp_path):
    html = '<html><style>' + '.a{color:red}' * 100 + '</style><body>' + '<p>same</p>' * 100 + '</body></html>'
    scorer = OfficialMetrics.__new__(OfficialMetrics)
    scorer.source_sha = 'fixed'
    scorer.compatibility_patch = None
    scorer._block_cache = OrderedDict()
    scorer.ocr = SimpleNamespace()
    def destructive(path):
        Path(path).write_text('')
        raise AssertionError('Upstream truncation must never run')
    scorer.module = SimpleNamespace(pre_process=destructive)
    def evaluate(args):
        pred = args[0][0]
        scorer.module.pre_process(pred)
        assert Path(pred).read_text() == html
        return [[None, None, [1.] * 5]]
    scorer.module.visual_eval_v3_multi = evaluate
    scorer.score(None, html, html, [320, 240], tmp_path/'scores')
    assert json.loads((tmp_path/'scores/official.json').read_text())['evaluation_contract'] == EVALUATION_CONTRACT


def test_refresh_archives_only_evaluation_and_resumes_new_contract(tmp_path):
    source = tmp_path/'prepare/prepared.jsonl'
    source.parent.mkdir()
    source.write_text('original outputs')
    out = tmp_path/'evaluation'
    out.mkdir()
    (out/'config.json').write_text('{"max_nodes": 2048}')
    (out/'metrics.jsonl').write_text('old scores')
    assert refresh_cache(out)['max_nodes'] == 2048
    assert not out.exists()
    assert source.read_text() == 'original outputs'
    assert len(list(tmp_path.glob('evaluation.before-*'))) == 1
    out.mkdir()
    (out/'config.json').write_text(json.dumps({'official_evaluation_contract': EVALUATION_CONTRACT}))
    (out/'metrics.jsonl').write_text('new scores')
    refresh_cache(out)
    assert (out/'metrics.jsonl').read_text() == 'new scores'
    assert len(list(tmp_path.glob('evaluation.before-*'))) == 1


def test_reevaluation_keeps_legacy_roster_without_new_generation(tmp_path, monkeypatch):
    import framediff.reevaluate_paper as reevaluate
    from framediff.paper_matrix import LEGACY_MODELS
    root = tmp_path/'old-run'
    aliases = [a for a, _ in LEGACY_MODELS]
    for dataset in ('design2code', 'design2code-hard'):
        for alias in aliases:
            base = root/dataset/alias/'seeds/42'
            for path in (base/'prepare/prepared.jsonl', base/'ui2coden/results.jsonl'):
                path.parent.mkdir(parents=True)
                path.write_text('{}\n')
    official = tmp_path/'official'
    source = official/'Design2Code/metrics/visual_score.py'
    source.parent.mkdir(parents=True); source.write_text('')
    calls = []
    monkeypatch.setattr(reevaluate.subprocess, 'run', lambda options, **kw: calls.append(options))
    monkeypatch.setattr('sys.argv', ['reevaluate', '--root', str(root), '--official-repo', str(official)])
    reevaluate.main()
    assert len([c for c in calls if 'web-evaluate' in c]) == 24
    assert all('web-prepare' not in c and 'web-repair' not in c for c in calls)
    final = calls[-1]
    start = final.index('--models') + 1
    assert final[start:final.index('--skip-tuide')] == aliases
    report = json.loads((root/'reevaluation-report.json').read_text())
    assert report['models'] == aliases and report['tables_regenerated']

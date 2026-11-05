import json

from PIL import Image
import pytest

from framediff.paper_suite import build, build_initial_seed_suite


def test_build_separates_main_and_ablation(tmp_path):
    rows=[]
    for method in ('initial','self-revision-1','screenshot-policy','abstract-policy','abstract-vlm-self-revision'):
        rows.append({'id':'page/repeat-1','page_id':'page','method':method,'dom_box_iou':.5,
                     'pixel_mae':.1,'pipeline_seconds':1.,'failed':False})
    evaluation=tmp_path/'qwen'/'evaluation';evaluation.mkdir(parents=True)
    with (evaluation/'metrics.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    build(tmp_path,['qwen'],'qwen',tmp_path/'paper')
    main=(tmp_path/'paper/main/qwen/paper-report.md').read_text()
    ablation=(tmp_path/'paper/ablation/paper-report.md').read_text()
    assert 'Initial VLM' in main and 'D2C Self-Revision (RGB)' in main and 'TUIDE w/o abstraction' not in main
    assert 'TUIDE w/o abstraction' in ablation and 'TUIDE w/o policy' in ablation and 'Initial VLM' not in ablation


def test_build_can_skip_api_expensive_ablation(tmp_path):
    rows=[{'id':'page/repeat-1','page_id':'page','method':method,'dom_box_iou':.5,
           'pixel_mae':.1,'pipeline_seconds':1.,'failed':False}
          for method in ('initial','self-revision-1','abstract-policy')]
    evaluation=tmp_path/'gpt4o'/'evaluation';evaluation.mkdir(parents=True)
    with (evaluation/'metrics.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    build(tmp_path,['gpt4o'],'gpt4o',tmp_path/'paper',include_ablation=False)
    index=(tmp_path/'paper/README.md').read_text()
    assert 'gpt4o' in index and 'Ablation' not in index


@pytest.mark.parametrize('name', ['qwen', 'ui2coden'])
def test_ui2coden_shares_initial_and_target(tmp_path, name):
    alias = tmp_path / name
    alias.mkdir()
    initial = tmp_path / 'initial.html'; initial.write_text('<html>initial</html>')
    target = tmp_path / 'target.png'; Image.new('RGB', (8, 8), 'white').save(target)
    prepared = {'id': 'page', 'screenshot': str(target), 'methods': {
                'initial': {'html': str(initial)}, 'design2code-self-revision-3': {'html': str(initial)}},
                'benchmark_protocol': 'local-reference-render-v1'}
    prep_dir = alias / 'prepare'; prep_dir.mkdir()
    (prep_dir / 'prepared.jsonl').write_text(json.dumps(prepared) + '\n')
    main_result = {**prepared, 'page_id': 'page', 'id': 'page/repeat-1'}
    main_repair = alias / 'repair'; main_repair.mkdir()
    (main_repair / 'results.jsonl').write_text(json.dumps(main_result) + '\n')
    evaluation = alias / 'evaluation'; evaluation.mkdir()
    def metric(method, score):
        return {'id': 'page/repeat-1', 'page_id': 'page', 'method': method,
                'benchmark_protocol': 'local-reference-render-v1', 'dom_box_iou': score,
                'official_clip': score, 'official_position': score, 'pixel_mae': 1-score, 'pipeline_seconds': 1, 'failed': False}
    (evaluation / 'metrics.jsonl').write_text(''.join(json.dumps(metric(m, v)) + '\n' for m, v in (
        ('initial', .3), ('design2code-self-revision-3', .4), ('abstract-policy', .5))))
    ui = alias / 'ui2coden'; ui.mkdir()
    (ui / 'results.jsonl').write_text(json.dumps({**prepared, 'methods': {'initial': {'html': str(initial)}}}) + '\n')
    ui_eval = ui / 'evaluation'; ui_eval.mkdir()
    (ui_eval / 'metrics.jsonl').write_text(json.dumps(metric('ui2coden-round-3', .6)) + '\n')
    build(tmp_path, [name], name, tmp_path / 'paper', False, True, 3, True, require_complete_metrics=True)
    table = (tmp_path / 'paper/main' / name / 'main-table.md').read_text()
    assert 'Position ↑ | CLIP ↑ | Box IoU ↑' in table
    assert 'Block ↑' not in table
    if name == 'ui2coden':
        assert 'UI2CodeN self-revision' in table and 'D2C Self-Revision (RGB)' not in table
    else:
        assert 'UI2CodeN polishing' in table and 'D2C Self-Revision (RGB)' in table
    scores = [json.loads(line) for line in (evaluation/'metrics.jsonl').read_text().splitlines()]
    scores[0].pop('official_clip')
    (evaluation/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores))
    with pytest.raises(ValueError, match='missing official_clip'):
        build(tmp_path, [name], name, tmp_path/'incomplete', False, True, 3, True, require_complete_metrics=True)
    (ui / 'results.jsonl').write_text(json.dumps({**prepared, 'screenshot': str(target),
                                                  'methods': {'initial': {'html': str(tmp_path / 'wrong.html')}}}) + '\n')
    (tmp_path / 'wrong.html').write_text('<html>wrong</html>')
    with pytest.raises(ValueError, match='initial HTML differs'):
        build(tmp_path, [name], name, tmp_path / 'bad-paper', False, True, 3, True, require_complete_metrics=True)


def test_initial_seed_suite_fixes_models_and_varies_drafts(tmp_path):
    alias = tmp_path / 'qwen'
    target = tmp_path / 'target.png'; Image.new('RGB', (8, 8), 'white').save(target)
    for seed, initial_score, repair_score in ((11, .2, .5), (22, .3, .7)):
        root = alias / 'seeds' / str(seed)
        html = root / 'initial.html'; root.mkdir(parents=True); html.write_text(f'<html>{seed}</html>')
        prepared = {'id': 'page', 'page_id': 'page', 'screenshot': str(target),
                    'benchmark_protocol': 'local-reference-render-v1',
                    'methods': {'initial': {'html': str(html)},
                                'design2code-self-revision-3': {'html': str(html)}}}
        prep = root / 'prepare'; prep.mkdir()
        (prep / 'prepared.jsonl').write_text(json.dumps(prepared) + '\n')
        (prep / 'config.json').write_text(json.dumps({'settings': {'seed': seed, 'initial_temperature': .7}}))
        repair = root / 'repair'; repair.mkdir()
        (repair / 'results.jsonl').write_text(json.dumps({**prepared, 'id': 'page/repeat-1'}) + '\n')
        (repair / 'config.json').write_text(json.dumps({'checkpoints': ['raw', 'abstract', 'detector']}))
        evaluation = root / 'evaluation'; evaluation.mkdir()
        def metric(method, score):
            return {'id': 'page/repeat-1', 'page_id': 'page', 'method': method,
                    'benchmark_protocol': 'local-reference-render-v1', 'dom_box_iou': score,
                    'pipeline_seconds': 1, 'failed': False}
        (evaluation / 'metrics.jsonl').write_text(''.join(json.dumps(metric(m, v)) + '\n' for m, v in (
            ('initial', initial_score), ('design2code-self-revision-3', initial_score + .1),
            ('abstract-policy', repair_score))))
    build_initial_seed_suite(tmp_path, ['qwen'], 'qwen', tmp_path / 'paper', ['11', '22'], False, False)
    summary = json.loads((tmp_path / 'paper/main/qwen/initial-seed-summary.json').read_text())
    assert summary['methods']['TUIDE']['metrics']['dom_box_iou']['mean'] == pytest.approx(.6)
    assert summary['methods']['TUIDE']['metrics']['dom_box_iou']['sd'] == pytest.approx(.141421356)
    assert '0.600 ± 0.141' in (tmp_path / 'paper/main/qwen/main-table.md').read_text()
    prepared_path = alias / 'seeds/22/prepare/prepared.jsonl'
    record = json.loads(prepared_path.read_text())
    record['methods']['initial']['failed'] = True
    prepared_path.write_text(json.dumps(record) + '\n')
    build_initial_seed_suite(tmp_path, ['qwen'], 'qwen', tmp_path / 'different-coverage', ['11', '22'], False, False)
    varying = json.loads((tmp_path / 'different-coverage/main/qwen/initial-seed-summary.json').read_text())
    assert varying['methods']['TUIDE']['n_pages_by_seed'] == {'11': 1, '22': 0}
    assert varying['methods']['Initial VLM']['metrics']['initial_success_rate']['mean'] == .5
    assert 'dom_box_iou' not in varying['methods']['TUIDE']['metrics']
    (alias / 'seeds/22/repair/config.json').write_text(json.dumps({'checkpoints': ['other']}))
    with pytest.raises(ValueError, match='checkpoint changed'):
        build_initial_seed_suite(tmp_path, ['qwen'], 'qwen', tmp_path / 'wrong-paper', ['11', '22'], False, False)


def test_select_second_round_from_three_round_evaluation(tmp_path):
    alias = tmp_path/'qwen'
    (alias/'evaluation').mkdir(parents=True)
    (alias/'prepare').mkdir()
    (alias/'prepare/prepared.jsonl').write_text(json.dumps({'methods': {
        'design2code-self-revision-2': {}, 'design2code-self-revision-3': {}}})+'\n')
    rows = [{'id':'page/repeat-1', 'page_id':'page', 'method':method,
             'dom_box_iou':score, 'pixel_mae':1-score, 'pipeline_seconds':1., 'failed':False}
            for method,score in [('initial',.1), ('abstract-policy',.8),
                                 ('design2code-self-revision-2',.6), ('design2code-self-revision-3',.9)]]
    (alias/'evaluation/metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    build(tmp_path,['qwen'],'qwen',tmp_path/'paper',include_ablation=False,revision_rounds=2)
    text=(tmp_path/'paper/main/qwen/paper-report.md').read_text()
    assert 'D2C Self-Revision (RGB)' in text
    quality_row = next(line for line in text.splitlines() if line.startswith('| D2C Self-Revision (RGB) |'))
    assert '0.6000' in quality_row and '0.9000' not in quality_row
    # The complete evaluation remains intact for separate N=3 reporting.
    assert len((alias/'evaluation/metrics.jsonl').read_text().splitlines()) == 4


@pytest.mark.parametrize('alias',['qwen3vl8b','ui2coden'])
def test_vlm_only_seed_suite_without_checkpoints(tmp_path,alias):
    from framediff.ir import write_json,write_jsonl
    root=tmp_path/alias/'seeds/42'
    root.mkdir(parents=True)
    target=root/'target.png';Image.new('RGB',(8,8),'white').save(target)
    initial=root/'initial.html';initial.write_text('<html><body>Initial</body></html>')
    rows=[];metrics=[];ui_metrics=[]
    for page,failed in [('good',False),('bad',True)]:
        methods={'initial':dict(html=str(initial),failed=failed),
                 'design2code-self-revision-2':dict(html=str(initial),failed=failed)}
        rows.append(dict(id=page,screenshot=str(target),methods=methods,benchmark_protocol='local-reference-render-v1'))
        for method,score in [('initial',.3),('design2code-self-revision-2',.4),('ui2coden-round-2',.6)]:
            metric=dict(id=page,method=method,failed=failed,benchmark_protocol='local-reference-render-v1',
                        **{k:None if failed else score for k in ('official_position','official_clip','dom_box_iou')})
            (ui_metrics if method.startswith('ui2coden') else metrics).append(metric)
    write_jsonl(root/'prepare/prepared.jsonl',rows)
    write_json(root/'prepare/config.json',dict(settings=dict(seed=42,initial_temperature=0)))
    write_jsonl(root/'evaluation/metrics.jsonl',metrics)
    write_jsonl(root/'ui2coden/results.jsonl',rows)
    write_jsonl(root/'ui2coden/evaluation/metrics.jsonl',ui_metrics)
    build_initial_seed_suite(tmp_path,[alias],alias,tmp_path/'paper',['42'],
        include_ablation=False,include_ui2coden=True,require_complete_metrics=True,
        revision_rounds=2,include_tuide=False)
    summary=json.loads((tmp_path/f'paper/main/{alias}/initial-seed-summary.json').read_text())
    assert not (root/'repair').exists()
    assert summary['fixed_checkpoint_hashes']=={}
    assert 'TUIDE' not in summary['methods']
    assert len(summary['methods'])==(2 if alias=='ui2coden' else 3)
    assert all(item['n_pages_by_seed']=={'42':1} for item in summary['methods'].values())
    assert summary['methods']['Initial VLM']['metrics']['initial_success_rate']['mean']==.5
    label='UI2CodeN self-revision' if alias=='ui2coden' else 'UI2CodeN polishing'
    assert summary['methods'][label]['metrics']['dom_box_iou']['mean']==.6

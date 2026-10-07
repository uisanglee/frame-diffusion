import json

from framediff.paper_suite import build


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

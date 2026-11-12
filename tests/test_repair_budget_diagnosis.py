from framediff.repair_budget_diagnosis import summarize


def sample(**kwargs):
    row=dict(dataset='webui',model='m',seed='42',budget=5,condition='detector',
             initial_failed=False,failed=False,position=.8,clip=.9,iou=.4,
             initial_iou=.3,delta_iou=.1,actions=4)
    row.update(kwargs)
    return row


def test_same_successful_initial_cohort_includes_failed_repair():
    result=summarize([sample(),sample(failed=True,iou=.3,delta_iou=0),
                      sample(initial_failed=True)])[0]
    assert result['attempted']==3
    assert result['eligible']==2
    assert result['failed']==1
    assert result['improved_percent']==50
    assert result['iou']==.35


def test_missing_metric_not_silently_removed():
    result=summarize([sample(),sample(position=None)])[0]
    assert result['missing']==1
    assert result['iou'] is None
    assert result['improved_percent'] is None


def test_budget_and_condition_are_separate():
    assert len(summarize([sample(),sample(budget=0),sample(condition='reference-dom')]))==3


def test_saved_baseline_reuses_valid_scores_and_rejects_errors(tmp_path):
    from framediff.repair_budget_diagnosis import saved_baseline
    from framediff.ir import write_jsonl
    html=tmp_path/'initial.html';html.write_text('<html>test</html>')
    image=tmp_path/'target.png';image.write_bytes(b'test')
    source=dict(id='page',viewport=[1280,720],screenshot=str(image),
                methods={'initial':dict(html=str(html),failed=False)})
    (tmp_path/'repair').mkdir();(tmp_path/'evaluation').mkdir()
    write_jsonl(tmp_path/'repair/results.jsonl',[dict(source,id='page/repeat-1')])
    metric=dict(id='page/repeat-1',method='initial',dom_box_iou=.3,official_position=.8,official_clip=.9)
    path=tmp_path/'evaluation/metrics.jsonl'
    write_jsonl(path,[metric])
    assert saved_baseline(tmp_path,[source])[0]['id']=='page'
    write_jsonl(path,[dict(metric,evaluation_error='render failed')])
    assert saved_baseline(tmp_path,[source]) is None

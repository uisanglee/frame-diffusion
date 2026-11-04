from framediff.repair_diagnostics import compare_three, summarize, outcomes


def box(x, label=1):
    return dict(box=[x,0,x+10,10],label=label)


def test_missing_and_spurious_detector_boxes_and_initial_displacement():
    ref=[box(0),box(30)]
    pairs=compare_three(ref,[box(0),box(70)],[box(0),box(100)])
    d=pairs['detector_vs_reference']
    assert (d['matched'],d['unmatched_predictions'],d['unmatched_targets'])==(1,1,1)
    assert d['precision']==d['recall']==.5
    assert pairs['initial_vs_reference']['unmatched_targets']==1
    assert pairs['initial_vs_detector']['unmatched_targets']==1
    # Displacement is only geometric nonmatching, never labeled semantic absence.
    assert 'missing_elements' not in pairs['initial_vs_reference']


def test_failed_initial_keeps_detector_evaluation_and_no_empty_dom_score():
    pairs=compare_three([box(0)],[],None)
    assert set(pairs)=={'detector_vs_reference'}
    assert pairs['detector_vs_reference']['recall']==0
    assert pairs['detector_vs_reference']['precision'] is None


def test_duplicate_predictions_are_not_multiple_true_positives():
    d=compare_three([box(0)],[box(0),box(0)],[box(0)])['detector_vs_reference']
    assert d['matched']==1 and d['unmatched_predictions']==1


def test_detector_targets_deduplicated_across_models_but_initials_retained():
    row=dict(target_key='same',comparisons=compare_three([box(0)],[box(0)],[box(0)]))
    summary=summarize([row,row])
    assert summary['detector_vs_reference']['pages']==1
    assert summary['initial_vs_reference']['pages']==2


def test_outcomes_keep_failed_repair_last_valid_score_and_missing_metrics():
    rows=[dict(id='p/repeat-1',method='initial',dom_box_iou=.4),
          dict(id='p/repeat-1',method='abstract-policy',dom_box_iou=.6,failed=True),
          dict(id='p/repeat-2',method='abstract-policy')]
    result=outcomes(rows)
    assert result[1]['improved'] is True and result[1]['failed'] is True
    assert result[2]['delta_box_iou'] is None


def test_postprocess_caches_targets_and_joins_metrics(tmp_path,monkeypatch):
    import io,json
    from types import SimpleNamespace
    from PIL import Image
    from framediff import visual_train,visual,html_bridge
    from framediff.ir import write_jsonl
    from framediff.repair_diagnostics import run
    buf=io.BytesIO();Image.new('RGB',(100,100),'white').save(buf,format='PNG')
    image=tmp_path/'target.png';image.write_bytes(buf.getvalue())
    html=tmp_path/'target.html';html.write_text('<html><body>Target</body></html>')
    checkpoint=tmp_path/'detector.pt';checkpoint.write_bytes(b'fake')
    seed=tmp_path/'runs/model/seeds/42';(seed/'prepare').mkdir(parents=True)
    source=dict(id='page/view',group='heldout',source_sha='source',viewport=[100,100],
                html=str(html),screenshot=str(image),target_image_source='reference-render',
                methods={'initial':dict(html=str(html),failed=False)})
    write_jsonl(seed/'prepare/prepared.jsonl',[source])
    (seed/'evaluation').mkdir()
    write_jsonl(seed/'evaluation/metrics.jsonl',[dict(id='page/view/repeat-1',method='initial',dom_box_iou=.4),
        dict(id='page/view/repeat-1',method='abstract-policy',dom_box_iou=.7)])
    class Browser:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def load(self,*a):pass
        @property
        def page(self):return self
        def screenshot(self,**kw):return buf.getvalue()
    monkeypatch.setattr(html_bridge,'HtmlBrowser',Browser)
    monkeypatch.setattr(visual,'detector_labels',lambda b:dict(elements=[box(0)],label_audit={}))
    monkeypatch.setattr(visual_train,'load_detector',lambda *a:(None,dict(config=dict(visual_geometry_contract=visual.VISUAL_GEOMETRY_CONTRACT))))
    calls=[]
    monkeypatch.setattr(visual_train,'detect',lambda *a:(calls.append(1) or [box(0)]))
    args=SimpleNamespace(root=str(tmp_path/'runs'),out=str(tmp_path/'out'),dataset='webui',
                         detector_checkpoint=str(checkpoint),device='cpu',threshold=.4)
    run(args);run(args)
    assert len(calls)==1
    report=json.loads((tmp_path/'out/summary.json').read_text())
    assert report['diagnostic_errors']==0
    assert report['detector']['recall_iou50']==1
    record=json.loads((tmp_path/'out/pages.jsonl').read_text())
    assert record['repair_outcomes'][1]['improved'] is True


def test_associations_exclude_failed_initial_and_preserve_method_groups():
    from framediff.repair_diagnostics import associations
    row=dict(model='vlm',initial_failed=False,errors={},
             comparisons=compare_three([box(0),box(30)],[box(0)],[box(0)]),
             repair_outcomes=outcomes([dict(id='p',method='initial',dom_box_iou=.2),
                                      dict(id='p',method='abstract-policy',dom_box_iou=.3)]))
    values=associations([row,{**row,'initial_failed':True}])
    assert len(values)==3
    assert all(v['n']==1 and v['improved_percent']==100 for v in values)
    assert values[0]['method']=='abstract-policy'

from types import SimpleNamespace

import pytest

from framediff.image_asset_oracle import paired_inputs, report
from framediff.ir import write_jsonl, read_json


def prepare_pair(root):
    row=dict(id='page/online-0',viewport=[320,240],declaration_state=['120px'],
             target_declaration_state=['200px'])
    for mode in ('blocked','images'):
        (root/mode).mkdir(parents=True)
        write_jsonl(root/mode/'policy-test.jsonl',[row])
    return row


def test_requires_identical_corruptions(tmp_path):
    row=prepare_pair(tmp_path)
    assert len(paired_inputs(tmp_path)['blocked'])==1
    row['declaration_state']=['130px']
    write_jsonl(tmp_path/'images/policy-test.jsonl',[row])
    with pytest.raises(ValueError,match='CSS edits differ'):
        paired_inputs(tmp_path)


def test_four_condition_report_preserves_failures(tmp_path):
    row=prepare_pair(tmp_path)
    out=tmp_path/'oracle-analysis';out.mkdir()
    detections=[]
    for mode in ('blocked','images'):
        work=out/mode/'repair';work.mkdir(parents=True)
        results=[]
        for method,score,failed in [('abstract-policy',.4,True),('abstract-oracle',.8,False)]:
            results.append(dict(id=row['id'],method=method,initial_box_iou=.4,final_box_iou=score,
                                failed=failed,scroll_guard_rolled_back=False,png=str(work/(method+'.png'))))
        write_jsonl(work/'results.jsonl',results)
        detections.append(dict(id=row['id'],condition=mode,precision_iou50=.8,
                               recall_iou50=.6,false_positives=1,false_negatives=2))
    write_jsonl(out/'detector-pages.jsonl',detections)
    report(SimpleNamespace(root=str(tmp_path)))
    summary=read_json(out/'repair-summary.json')
    assert len(summary)==4
    assert summary[0]['failures']==1 and summary[0]['improved_percent']==0
    assert summary[1]['improved_percent']==100
    assert (out/'index.html').is_file()


@pytest.mark.browser
def test_detector_audit_and_cache(tmp_path,monkeypatch):
    from framediff.image_asset_oracle import audit
    from framediff.html_bridge import HtmlBrowser
    from framediff.visual import detector_labels,VISUAL_GEOMETRY_CONTRACT
    import framediff.visual_train as training
    row=prepare_pair(tmp_path)
    target=tmp_path/'target.html';target.write_text('<html><body><button>Test</button></body></html>')
    png=tmp_path/'target.png'
    with HtmlBrowser() as browser:
        browser.load(target.read_text(),row['viewport'])
        capture,_=browser.stable_capture(timeout_ms=15000);png.write_bytes(capture)
        labels=detector_labels(browser)['elements']
    row.update(target_html=str(target),target_image=str(png),group='heldout')
    for mode in ('blocked','images'):
        write_jsonl(tmp_path/mode/'policy-test.jsonl',[row])
    checkpoint=tmp_path/'detector.pt';checkpoint.write_bytes(b'test')
    monkeypatch.setattr(training,'load_detector',lambda *a:(object(),{'config':{'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT}}))
    calls=[]
    def detect(*args):
        calls.append(1)
        return labels
    monkeypatch.setattr(training,'detect',detect)
    args=SimpleNamespace(root=str(tmp_path),detector_checkpoint=str(checkpoint),device='cpu',threshold=.4)
    audit(args);audit(args)
    assert len(calls)==2
    summary=read_json(tmp_path/'oracle-analysis/detector-summary.json')
    assert summary['blocked']['precision_iou50']==1
    assert summary['images']['recall_iou50']==1

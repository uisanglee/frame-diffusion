from types import SimpleNamespace

from PIL import Image
import pytest

from framediff.detector_support import Aggregate, compare
from framediff.ir import write_jsonl


def element(box, label=1, score=.9):
    return {'box': box, 'label': label, 'score': score}


def test_error_categories_and_counts():
    target = [element([0,0,10,10]), element([30,0,40,10],2), element([60,0,70,10])]
    pred = [target[0], target[0], element([30,0,40,10],1), element([65,0,75,10]), element([100,0,110,10])]
    detail = compare(pred,target)
    assert sorted(e['kind'] for e in detail['predictions']) == sorted(['true_positive','duplicate','class_confusion','localization','unmatched'])
    assert detail['missed_targets'] == [1,2]
    bucket = Aggregate()
    bucket.add({'id':'p/iPhone-13 Pro','page_id':'p'},pred,target,detail)
    report = bucket.report()
    assert report['precision_iou50'] == .2
    assert report['recall_iou50'] == 1/3
    assert report['by_device']['iPhone']['targets'] == 3
    assert report['by_class']['text']['false_positives'] == 4
    assert report['by_class']['image']['false_negatives'] == 1


@pytest.mark.parametrize('pred,target', [([],[]),([],[element([0,0,10,10])]),([element([0,0,10,10])],[])])
def test_empty_inputs(pred,target):
    detail=compare(pred,target)
    assert not detail['matches']
    assert len(detail['missed_targets']) == len(target)


def test_evaluation_single_inference_sweep_and_visualization(tmp_path,monkeypatch):
    from framediff import visual_train
    image=tmp_path/'image.png';Image.new('RGB',(120,100),'white').save(image)
    data=tmp_path/'data.jsonl'
    write_jsonl(data,[{'id':'p/default_1280-720','page_id':'p','view':'default_1280-720',
                     'group':'heldout','source_sha':'sha','image':str(image),'elements':[element([0,0,10,10])]}])
    net=SimpleNamespace(roi_heads=SimpleNamespace(score_thresh=.05))
    monkeypatch.setattr(visual_train,'load_detector',lambda *a:(net,{'training_groups':[],'training_hashes':[]}))
    calls=[]
    def detect(*args):
        calls.append(args)
        return [element([0,0,10,10],score=.8),element([50,50,70,70],score=.35)]
    monkeypatch.setattr(visual_train,'detect',detect)
    args=SimpleNamespace(device='cpu',cpu_threads=1,checkpoint='fake.pt',data=str(data),limit=0,
                         threshold=.4,thresholds=[.3,.4,.9],out=str(tmp_path/'report.json'),
                         visualize_dir=str(tmp_path/'viz'),visualize_limit=1)
    report=visual_train.evaluate_detector(args)
    assert len(calls)==1
    from framediff.visual import VISUAL_GEOMETRY_CONTRACT
    monkeypatch.setattr(visual_train,'load_detector',lambda *a:(net,{
        'training_groups':[],'training_hashes':[],
        'config':{'visual_geometry_contract':VISUAL_GEOMETRY_CONTRACT}}))
    with pytest.raises(ValueError,match='visual label versions differ'):
        visual_train.evaluate_detector(args)
    assert len(calls)==1
    assert report['threshold_sweep']['0.3']['precision_iou50']==.5
    assert report['threshold_sweep']['0.4']['precision_iou50']==1
    assert report['threshold_sweep']['0.9']['recall_iou50']==0
    assert report['by_device']['desktop']['n']==1
    assert (tmp_path/'viz/000000.png').is_file()
    assert (tmp_path/'viz/000000.json').is_file()
    monkeypatch.setattr(visual_train,'load_detector',lambda *a:(net,{'training_groups':['heldout'],'training_hashes':[]}))
    with pytest.raises(ValueError,match='leakage'):
        visual_train.evaluate_detector(args)
    assert len(calls)==1

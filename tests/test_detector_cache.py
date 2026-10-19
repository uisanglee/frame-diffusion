import io
from PIL import Image

from framediff.detector_support import TargetPredictionCache


def test_prediction_cache_batches_unique_misses_and_survives_restart(tmp_path,monkeypatch):
    import framediff.visual_train as train
    calls=[]
    def detect(net,images,threshold):
        calls.append(len(images))
        return [[dict(box=[0,0,1,1],label=4,score=.9)] for _ in images]
    monkeypatch.setattr(train,'detect_batch',detect)
    images=[]
    for color in ('red','blue'):
        buf=io.BytesIO();Image.new('RGB',(5,5),color).save(buf,format='PNG');images.append(buf.getvalue())
    cache=TargetPredictionCache(tmp_path,'checkpoint-a',.4)
    first=cache.predict(None,[images[0],images[1],images[0]])
    assert calls==[2] and first[0]==first[2]
    cache=TargetPredictionCache(tmp_path,'checkpoint-a',.4)
    assert cache.predict(None,images)==first[:2] and calls==[2]
    other=TargetPredictionCache(tmp_path,'checkpoint-b',.4)
    other.predict(None,images)
    assert calls==[2,2]
    for path in tmp_path.glob('*.json'):path.write_text('{bad')
    cache=TargetPredictionCache(tmp_path,'checkpoint-a',.4)
    assert cache.predict(None,images)==first[:2] and calls==[2,2,2]

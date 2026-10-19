"""Content-addressed frozen target predictions; infer only unique cache misses."""
import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path


class TargetPredictionCache:
    def __init__(self,folder,checkpoint_sha,threshold,capacity=256):
        from importlib.metadata import version
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.identity={'contract':'frozen-target-prediction-v1','checkpoint':checkpoint_sha,
                       'threshold':threshold,'torchvision':version('torchvision')}
        self.threshold=threshold;self.capacity=capacity;self.memory=OrderedDict()
        self.hits=0;self.misses=0

    def remember(self,key,items):
        self.memory[key]=items;self.memory.move_to_end(key)
        while len(self.memory)>self.capacity:self.memory.popitem(last=False)

    def read(self,key):
        from .visual import CLASSES
        if key in self.memory:
            self.memory.move_to_end(key);return self.memory[key]
        try:
            saved=json.loads((self.folder/f'{key}.json').read_text())
            if saved['key']!=key:return None
            items=saved['items']
            if not isinstance(items,list):return None
            for item in items:
                if (len(item['box'])!=4 or not all(math.isfinite(v) for v in item['box'])
                    or item['label'] not in range(1,len(CLASSES)) or not math.isfinite(item['score'])):return None
            self.remember(key,items);return items
        except (OSError,ValueError,KeyError,TypeError):return None

    def predict(self,net,images):
        from .visual_train import detector_input,detect_batch
        keys=[];missing={};results={}
        for image in images:
            data=image if isinstance(image,bytes) else Path(image).read_bytes()
            key=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()+data).hexdigest()
            keys.append(key)
            if key in results or key in missing:continue
            items=self.read(key)
            if items is None:missing[key]=data
            else:results[key]=items;self.hits+=1
        if missing:
            predictions=detect_batch(net,[detector_input(data) for data in missing.values()],self.threshold)
            for key,items in zip(missing,predictions):
                self.misses+=1;results[key]=items;self.remember(key,items)
                temporary=None
                try:
                    with tempfile.NamedTemporaryFile(mode='w',dir=self.folder,delete=False) as file:
                        temporary=Path(file.name);json.dump({'key':key,'items':items},file,allow_nan=False)
                    os.replace(temporary,self.folder/f'{key}.json')
                finally:
                    if temporary is not None:temporary.unlink(missing_ok=True)
        return [results[key] for key in keys]

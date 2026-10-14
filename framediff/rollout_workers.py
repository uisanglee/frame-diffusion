"""CPU browser workers with a single coordinator-owned batched model service.

Only CPU numpy payloads cross the queue. No checkpoint or CUDA context is loaded
in a worker. Requests are bounded by the number of running page jobs.
"""
import multiprocessing as mp
import queue
import time
from concurrent.futures import ProcessPoolExecutor

import torch


class RolloutCancelled(BaseException):
    pass


_requests=None
_cancel=None
_policies=None
_parser=None


def _request(kind,payload):
    if _cancel.is_set():raise RolloutCancelled()
    receive,send=mp.Pipe(duplex=False)
    try:
        _requests.put((kind,payload,send))
        while not receive.poll(.1):
            if _cancel.is_set():raise RolloutCancelled()
        ok,value=receive.recv()
        if not ok:raise RuntimeError(value)
        return value
    finally:
        receive.close();send.close()


class PolicyProxy:
    def __init__(self,name,cfg):
        from .tree_edits import EditTokenizer
        self.name=name;self.cfg=cfg;self.tokenizer=EditTokenizer(cfg.max_nodes,cfg.fields)
        self.placeholder=torch.zeros(1)

    def parameters(self):return iter([self.placeholder])
    def encode_pair(self,*args):return None  # encoded once, in the coordinator

    def predict(self,batch,target,current,states,encoded_pair=None,allowed_owners=None):
        return _request(self.name,{
            'batch':{k:v.numpy() if torch.is_tensor(v) else v for k,v in batch.items()},
            'target':target.numpy(),'current':current.numpy(),'states':states,
            'allowed_owners':allowed_owners})


class DetectorProxy:
    def detect_items(self,path,threshold):return _request('detector',(path,threshold))


def _initialize(requests,cancel,configs):
    global _requests,_cancel,_policies,_parser
    from .preparation_workers import _initialize_browser_worker
    torch.set_num_threads(1)
    _initialize_browser_worker()
    _requests=requests;_cancel=cancel
    _policies={name:PolicyProxy(name,cfg) for name,cfg in configs.items()}
    _parser=DetectorProxy()


def _page(task):
    from .fixed_test import evaluate_page
    from .html_bridge import HtmlBrowser
    if _cancel.is_set():raise RolloutCancelled()
    index,args,row,page,out=task
    with HtmlBrowser() as browser:
        result=evaluate_page(args,row,page,out,browser,_policies,_parser)
    return index,result


def merge_requests(payloads,device):
    """Pad variable owner counts exactly as tree_batch does for training."""
    n=max(p['batch']['mask'].shape[1] for p in payloads)
    batch={}
    for key in payloads[0]['batch']:
        if key=='explicit_owner_limits':
            batch[key]=[p['batch'][key][0] for p in payloads];continue
        values=[torch.as_tensor(p['batch'][key],device=device) for p in payloads]
        shape=list(values[0].shape);shape[0]=len(values);shape[1]=n
        if key=='relation':shape[2]=n
        merged=torch.zeros(shape,device=device,dtype=values[0].dtype)
        for i,value in enumerate(values):
            if key=='relation':merged[i,:value.shape[1],:value.shape[2]]=value[0]
            else:merged[i,:value.shape[1]]=value[0]
        batch[key]=merged
    target=torch.cat([torch.as_tensor(p['target'],device=device) for p in payloads])
    current=torch.cat([torch.as_tensor(p['current'],device=device) for p in payloads])
    return batch,target,current,[p['states'][0] for p in payloads]


def infer(kind,payloads,models,parser):
    if kind=='detector':
        from .visual_train import detector_input
        from .visual import CLASSES
        device=next(parser.parameters()).device
        with torch.inference_mode():predictions=parser([detector_input(p[0]).to(device) for p in payloads])
        return [[{'box':box,'label':int(label),'score':float(score)}
            for box,label,score in zip(pred['boxes'].cpu().tolist(),pred['labels'].cpu().tolist(),pred['scores'].cpu().tolist())
            if score>=payload[1] and 0<label<len(CLASSES)] for pred,payload in zip(predictions,payloads)]
    model=models[kind];device=next(model.parameters()).device
    batch,target,current,states=merge_requests(payloads,device)
    return model.predict_batch(batch,target,current,states,
                               allowed_owners=[p['allowed_owners'] for p in payloads])


def _serve(requests,models,parser,stats):
    kind=requests[0][0];payloads=[r[1] for r in requests]
    try:
        results=infer(kind,payloads,models,parser)
        replies=[(True,result) for result in results]
        stats['inference_batches']=stats.get('inference_batches',0)+1
        stats['inference_requests']=stats.get('inference_requests',0)+len(requests)
        stats['max_observed_batch']=max(stats.get('max_observed_batch',0),len(requests))
    except Exception as exc:
        # Isolate a bad page or oversized batch. Do not fail unrelated pages.
        if len(requests)>1:
            stats['batch_fallbacks']=stats.get('batch_fallbacks',0)+1
            for request in requests:_serve([request],models,parser,stats)
            return
        replies=[(False,f'{type(exc).__name__}: {exc}')]
    for (_,_,connection),reply in zip(requests,replies):
        try:connection.send(reply)
        finally:connection.close()


def parallel_pages(tasks,models,parser,workers,batch_size,wait_ms,stats):
    for model in models.values():
        if model.cfg.policy_head!='replacement' or not hasattr(model,'predict_batch'):
            raise ValueError('Parallel rollout requires a replacement policy; use --workers 1 for legacy models')
    context=mp.get_context('spawn');requests=context.Queue();cancel=context.Event()
    pool=ProcessPoolExecutor(workers,mp_context=context,initializer=_initialize,
        initargs=(requests,cancel,{name:model.cfg for name,model in models.items()}))
    tasks=iter(tasks);futures=set();waiting=[]
    try:
        for _ in range(workers):
            task=next(tasks,None)
            if task is not None:futures.add(pool.submit(_page,task))
        while futures:
            for future in list(futures):
                if future.done():
                    futures.remove(future)
                    yield future.result()
                    task=next(tasks,None)
                    if task is not None:futures.add(pool.submit(_page,task))
            try:waiting.append(requests.get(timeout=.01))
            except queue.Empty:continue
            deadline=time.monotonic()+wait_ms/1000
            while len(waiting)<batch_size:
                remaining=deadline-time.monotonic()
                if remaining<=0:break
                try:waiting.append(requests.get(timeout=remaining))
                except queue.Empty:break
            while waiting:
                kind=waiting[0][0];group=[r for r in waiting if r[0]==kind][:batch_size]
                selected={id(r) for r in group}
                waiting=[r for r in waiting if id(r) not in selected]
                _serve(group,models,parser,stats)
    finally:
        cancel.set()
        for future in futures:future.cancel()
        # Waiting RPCs poll cancellation, letting worker finally blocks close
        # contexts and Chromium rather than killing browser children abruptly.
        pool.shutdown(wait=True,cancel_futures=True)
        for request in waiting:request[2].close()
        while True:
            try:requests.get_nowait()[2].close()
            except queue.Empty:break
        requests.close();requests.join_thread()

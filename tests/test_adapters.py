import gzip
import json
from types import SimpleNamespace
import pytest
from framediff.adapters import fit_observation,import_webui,fit_frames
from framediff.ir import execute,read_jsonl
from framediff.vlm import json_answer

def observation():
    return {'viewport':[800,600],'nodes':[
        {'id':'a','parent':None,'role':'main','name':'content','box':[100,100,600,400]},
        {'id':'b','parent':'a','role':'button','name':'save','box':[150,200,100,40]}]}

def test_fit_preserves_geometry():
    tree,targets,fit=fit_observation(observation())
    assert fit['metrics']['box_iou']>.95
    assert [n['id'] for n in tree['nodes']]==['__viewport__','a','b']

def test_json_answer():
    assert json_answer('```json\n{"x":1}\n```')=={'x':1}
    with pytest.raises(ValueError):json_answer('Here is the answer: {"x":1}')

def test_webui_actual_schema(tmp_path):
    ax={'nodes':[{'nodeId':'1','role':{'value':'RootWebArea'},'backendDOMNodeId':7},
                 {'nodeId':'2','parentId':'1','ignored':True},
                 {'nodeId':'3','parentId':'2','role':{'value':'button'},'name':{'value':'save'},'backendDOMNodeId':8}]}
    bb={'7':{'x':0,'y':0,'width':800,'height':600},'8':{'x':100,'y':100,'width':100,'height':40}}
    for name,data in [('axtree',ax),('bb',bb)]:
        with gzip.open(tmp_path/f'default_800-600-{name}.json.gz','wt') as f:json.dump(data,f)
    (tmp_path/'default_800-600-url.txt').write_text('https://example.org/page')
    args=SimpleNamespace(root=tmp_path,out=tmp_path/'raw',limit=0,max_nodes=128,split='train',seed=42)
    import_webui(args)
    records=list(read_jsonl(args.out/'observations.jsonl'))
    assert records[0]['group']=='example.org'
    assert records[0]['nodes'][1]['parent']=='1'
    fit_frames(SimpleNamespace(input=args.out/'observations.jsonl',out=tmp_path/'fit',max_error=.15,seed=42))
    fitted=list(read_jsonl(tmp_path/'fit/train.jsonl'))
    assert len(fitted)==1 and fitted[0]['source']=='fitted_frame_surrogate'

@pytest.mark.browser
def test_absolute_parity():
    from framediff.browser import Browser
    tree,_,_=fit_observation(observation())
    tree['nodes'][0]['props']['padding']=3
    tree['nodes'][1]['props']['padding']=2
    with Browser() as browser:
        actual=browser.render(tree,[800,600]);proxy=execute(tree,[800,600])
    assert max(abs(actual[k][j]-proxy[k][j]) for k in actual for j in range(4))<.2

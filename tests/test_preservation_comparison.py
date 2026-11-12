import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('compare',Path(__file__).parents[1]/'scripts/compare_preservation_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_unpaired_weights_are_not_combined(tmp_path):
    roots={name:tmp_path/name for name in ('original','pixel','geometry')}
    for name,root in roots.items():
        folder=root/'evaluations/mild/budget-1';folder.mkdir(parents=True)
        (folder/'config.json').write_text(json.dumps({'inputs':{'a.pt':'different' if name=='geometry' else 'same','data.jsonl':'same'}}))
        (folder/'results.jsonl').write_text(json.dumps(dict(id='a',method='abstract-policy',repeat=1,initial_box_iou=.5,final_box_iou=.6,actions=1,failed=False))+'\n')
    module.compare(roots,tmp_path/'report')
    result=json.loads((tmp_path/'report/comparison.json').read_text())
    assert len(result['results'])==2
    assert any('fingerprints differ' in w for w in result['warnings'])

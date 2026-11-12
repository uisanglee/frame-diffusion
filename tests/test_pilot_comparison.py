import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('gallery',Path(__file__).parents[1]/'scripts/export_pilot_comparison.py')
gallery=importlib.util.module_from_spec(spec);spec.loader.exec_module(gallery)


def test_saved_pngs_and_failure_reporting(tmp_path):
    run=tmp_path/'run';base=run/'webui/llava/seeds/42'
    def jsonl(path,rows):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(json.dumps(r) for r in rows))
    target=tmp_path/'target.png';target.write_bytes(b'target')
    prepared=[dict(id='page/view',screenshot=str(target),methods={'initial':{'failed':False}}),
              dict(id='failed',screenshot=str(target),methods={'initial':{'failed':True,'error':'truncated'}})]
    jsonl(base/'prepare/prepared.jsonl',prepared)
    jsonl(base/'repair/results.jsonl',[dict(id='page/view/repeat-1',page_id='page/view',repeat=1,methods={'abstract-policy':{'failed':False}})])
    work=base/'evaluation/pages'/gallery.signature('page/view/repeat-1');work.mkdir(parents=True)
    (work/'initial.png').write_bytes(b'initial');(work/'abstract-policy.png').write_bytes(b'repaired')
    output=tmp_path/'gallery';rows=gallery.export(run,output)
    assert len(rows)==2 and not rows[0]['errors']
    assert rows[1]['images']['initial'] is None
    assert rows[1]['errors']
    assert len(list((output/'assets').iterdir()))==3
    text=(output/'index.html').read_text()
    assert 'page/view' in text and 'Target' in text and 'truncated' in text
    assert str(tmp_path) not in text  # portable images, no server absolute paths

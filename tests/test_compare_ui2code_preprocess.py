import importlib.util
import json
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('compare_preprocess',Path(__file__).parents[1]/'scripts/compare_ui2code_preprocess.py')
diag=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

@pytest.mark.browser
def test_comparison_preserves_source_and_reuses_diagnostic(tmp_path, monkeypatch):
    repo=tmp_path/'official'
    metrics=repo/'Design2Code/metrics';metrics.mkdir(parents=True)
    utils=repo/'Design2Code/data_utils';utils.mkdir()
    (utils/'dedup_post_gen.py').write_text('def check_repetitive_content(path):\n    pass\n')
    (metrics/'visual_score.py').write_text('''raise RuntimeError("CLIP must not load")
def truncate_repeated_html_elements(soup):
    return str(soup)
def make_html(path):
    pass
def pre_process(path):
    with open(path) as f: text=f.read()
    cut=text.index('<style')+7
    print('Repetitive content found, start at '+str(cut))
    with open(path,'w') as f: f.write(text[:cut])
def unrelated():
  if False:
  invalid indentation deliberately
''')
    run=tmp_path/'run'; folder=run/'design2code/model/seeds/42/ui2coden';folder.mkdir(parents=True)
    source=tmp_path/'original.html'
    text='<html><head><style>body{color:red}</style></head><body><h1>Keep me</h1></body></html>'
    source.write_text(text)
    manifest=folder/'results.jsonl'
    manifest.write_text(json.dumps(dict(id='p',viewport=[320,240],methods={
        'initial':dict(failed=False), 'ui2coden-round-1':dict(html=str(source),failed=False)}))+'\n')
    monkeypatch.setattr(sys,'argv',['compare','--run',str(run),'--official-repo',str(repo)])
    diag.main()
    out=run/'preprocess-diagnosis'
    row=json.loads((out/'summary.json').read_text())[0]
    assert row['cut_inside_style'] and row['suspected_content_loss']
    assert row['pixel_mae']>0 and row['before_dom']['visible_text']=='Keep me'
    assert source.read_text()==text
    assert (out/'index.html').is_file()
    diag.main()
    assert json.loads((out/'summary.json').read_text())[0]['reused']
    assert source.read_text()==text

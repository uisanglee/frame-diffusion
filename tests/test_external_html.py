from types import SimpleNamespace

from PIL import Image

from framediff.external_html import build
from framediff.ir import read_jsonl


def test_external_manifest_with_optional_reference(tmp_path):
    image=tmp_path/'target.png';Image.new('RGB',(320,240)).save(image)
    initial=tmp_path/'gpt.html';initial.write_text('<html><body>GPT</body></html>')
    reference=tmp_path/'reference.html';reference.write_text('<html><body>Target</body></html>')
    args=SimpleNamespace(screenshot=str(image),initial_html=str(initial),reference_html=str(reference),
        out=str(tmp_path/'manifest'),id='mine',source_model='gpt',resume=False)
    row=build(args);saved=list(read_jsonl(tmp_path/'manifest/manifest.jsonl'))[0]
    assert row==saved
    assert saved['external_source_model']=='gpt' and saved['viewport']==[320,240]
    args.resume=True;build(args)

import hashlib
from types import SimpleNamespace

import pytest

from framediff import visual_data
from framediff.ir import read_json, read_jsonl, write_jsonl


def inputs(tmp_path, clean_only):
    rows=[]
    for i,split in enumerate(('train','val','test')):
        html=tmp_path/f'{split}.html'
        html.write_text(f'<html><body><div style="width:{100+i}px;height:50px;background:red">{split}</div></body></html>')
        rows.append(dict(id=split,group=split,split=split,html=str(html),viewport=[400,200]))
    manifest=tmp_path/'manifest.jsonl';write_jsonl(manifest,rows)
    return SimpleNamespace(manifest=str(manifest),out=str(tmp_path/'rendered'),resume=True,
        trajectories=1,max_noise=1,seed=42,max_nodes=32,abstract_size=64,min_elements=1,
        max_source_mae=1.,clean_only=clean_only)


@pytest.mark.parametrize('clean_only',[True,False])
def test_interrupted_build_reuses_legacy_cache_with_parallel_resume(tmp_path,monkeypatch,clean_only):
    args=inputs(tmp_path,clean_only)
    original=visual_data._render_page
    def interrupt_after_first(task):
        if task[2]==1:raise KeyboardInterrupt()
        return original(task)
    monkeypatch.setattr(visual_data,'_render_page',interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):visual_data.build(args)
    root=tmp_path/'rendered'
    first=root/'pages'/hashlib.sha256(b'train').hexdigest()[:20]/'records.json'
    saved=first.read_bytes();mtime=first.stat().st_mtime_ns
    # Config was created without --workers, as in existing sequential runs.
    assert 'workers' not in read_json(root/'config.json')['settings']
    monkeypatch.setattr(visual_data,'_render_page',original)
    args.workers=2
    visual_data.build(args)
    assert first.read_bytes()==saved and first.stat().st_mtime_ns==mtime
    assert read_json(root/'progress.json')['cache_hits']==1
    before={p:p.read_bytes() for p in root.glob('*.jsonl')}
    # Cache-only resume must not even open Chromium.
    monkeypatch.setattr(visual_data,'HtmlBrowser',lambda:pytest.fail('cached page rerendered'))
    args.workers=1
    visual_data.build(args)
    assert read_json(root/'progress.json')['cache_hits']==3
    assert all(p.read_bytes()==value for p,value in before.items())


@pytest.mark.parametrize('clean_only',[True,False])
def test_missing_asset_and_truncated_record_are_rebuilt(tmp_path,clean_only):
    args=inputs(tmp_path,clean_only);args.workers=2
    visual_data.build(args)
    root=tmp_path/'rendered'
    pages={s:root/'pages'/hashlib.sha256(s.encode()).hexdigest()[:20] for s in ('train','val','test')}
    original=list(read_jsonl(root/'policy-train.jsonl'))
    (pages['train']/'clean.png').unlink()
    (pages['val']/'records.json').write_text('{')
    untouched=(pages['test']/'records.json').stat().st_mtime_ns
    args.workers=1
    visual_data.build(args)
    assert (pages['train']/'clean.png').is_file()
    assert read_json(pages['val']/'records.json')['policy']
    assert (pages['test']/'records.json').stat().st_mtime_ns==untouched
    assert list(read_jsonl(root/'policy-train.jsonl'))==original
    assert read_json(root/'progress.json')['cache_hits']==1
    args.seed=9
    with pytest.raises(ValueError,match='Resume configuration/input changed'):
        visual_data.build(args)

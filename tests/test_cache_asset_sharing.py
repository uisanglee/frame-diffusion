import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/share_cache_assets.py'
    spec=importlib.util.spec_from_file_location('sharing',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def fixture(root,content):
    root.mkdir(parents=True)
    image=root/'after.png';image.write_bytes(content)
    record=root/'record.json'
    record.write_text(json.dumps({'row':{},'page':{},'assets':{
        str(image):hashlib.sha256(content).hexdigest()}}))
    return image,record


def test_dedup_preserves_records_bytes_and_reuses_storage(tmp_path):
    a,ra=fixture(tmp_path/'normalized-pages',b'same')
    b,rb=fixture(tmp_path/'.online-target-cache',b'same')
    c,rc=fixture(tmp_path/'different-view',b'different')
    snapshots={p:p.read_bytes() for p in (a,b,c,ra,rb,rc)}
    roots=[a.parent,b.parent,c.parent];store=tmp_path/'store'
    report=module().run(roots,store)
    assert report['estimated_reclaimable_bytes']==4 and not store.exists()
    assert not os.path.samefile(a,b)
    report=module().run(roots,store,True)
    assert report['applied'] and os.path.samefile(a,b)
    assert not os.path.samefile(a,c)
    assert all(p.read_bytes()==value for p,value in snapshots.items())
    assert module().run(roots,store,True)['estimated_reclaimable_bytes']==0
    a.unlink()
    assert b.read_bytes()==b'same'


def test_bad_checksum_prevents_cleanup(tmp_path):
    a,_=fixture(tmp_path/'a',b'same');b,_=fixture(tmp_path/'b',b'same')
    b.write_bytes(b'changed')
    roots=[a.parent,b.parent]
    assert module().run(roots,tmp_path/'store')['errors']
    with pytest.raises(ValueError,match='audit'):
        module().run(roots,tmp_path/'store',True)
    assert not (tmp_path/'store').exists()

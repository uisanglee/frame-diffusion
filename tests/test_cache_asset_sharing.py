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


def test_dedup_includes_legacy_rendered_without_removing_paths(tmp_path):
    entry=tmp_path/'lazy-train-views'/('a'*64)
    rendered=entry/'rendered';rendered.mkdir(parents=True)
    a=rendered/'clean.png';b=rendered/'target.png'
    a.write_bytes(b'identical pixels');b.write_bytes(a.read_bytes())
    result=entry/'result.json'
    result.write_text(json.dumps({'contract':'lazy-webui-train-view-cache-v2-verified',
        'assets':{str(a):hashlib.sha256(a.read_bytes()).hexdigest()}}))
    original=result.read_bytes()
    tool=module()
    report=tool.run([tmp_path/'lazy-train-views'],tmp_path/'blobs',False,True)
    assert not report['errors'] and report['estimated_reclaimable_bytes']==len(a.read_bytes())
    tool.run([tmp_path/'lazy-train-views'],tmp_path/'blobs',True,True)
    assert a.is_file() and b.is_file() and os.path.samefile(a,b)
    assert result.read_bytes()==original


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


@pytest.mark.parametrize('contract', ['lazy-webui-train-view-cache-v1',
                                    'lazy-webui-train-view-cache-v2-verified'])
def test_committed_nested_staging_paths_are_hash_verified(tmp_path, contract):
    root=tmp_path/'lazy-train-views';key='a'*64
    entry=root/key;folder=entry/'labels/normalized-pages/train'/('b'*64)
    image,record=fixture(folder,b'same')
    stale=root/('.'+key+'.random_123')/image.relative_to(entry)
    value=json.loads(record.read_text());sha=next(iter(value['assets'].values()))
    value['assets']={str(stale):sha};record.write_text(json.dumps(value))
    (entry/'result.json').write_text(json.dumps({'contract':contract,
                                               'result':{},'assets':{str(image):sha}}))
    other,_=fixture(tmp_path/'normalized-pages',b'same')
    report=module().run([root,other.parent],tmp_path/'store',True)
    assert not report['errors'] and len(report['resolved_staging_paths'])==1
    assert os.path.samefile(image,other)
    assert json.loads(record.read_text())==value
    # Matching name alone must never excuse a content mismatch.
    value['assets'][str(stale)]='0'*64;record.write_text(json.dumps(value))
    assert module().run([root],tmp_path/'store')['errors']
    with pytest.raises(ValueError,match='audit'):
        module().run([root],tmp_path/'store',True)


def test_v1_without_top_level_hashes_uses_nested_verified_record(tmp_path):
    entry=tmp_path/'lazy'/('a'*64)
    image,record=fixture(entry/'labels',b'verified')
    value=json.loads(record.read_text())
    stale=entry.parent/('.'+entry.name+'.old')/'labels/after.png'
    value['assets']={str(stale):hashlib.sha256(b'verified').hexdigest()}
    record.write_text(json.dumps(value))
    (entry/'result.json').write_text(json.dumps({'contract':'lazy-webui-train-view-cache-v1',
                                               'result':{'target_image':str(image)}}))
    report=module().run([entry.parent],tmp_path/'store')
    assert not report['errors'] and len(report['resolved_staging_paths'])==1


def test_staging_mapping_requires_matching_completed_entry(tmp_path):
    root=tmp_path/'lazy-train-views';entry=root/('a'*64)
    image,record=fixture(entry/'labels',b'same')
    value=json.loads(record.read_text());sha=next(iter(value['assets'].values()))
    value['assets']={str(root/('.'+'c'*64+'.rand')/'labels/after.png'):sha}
    record.write_text(json.dumps(value))
    (entry/'result.json').write_text(json.dumps({'contract':'lazy-webui-train-view-cache-v2-verified'}))
    assert module().run([root],tmp_path/'store')['errors']


def test_diagnostic_cleanup_preserves_inputs_and_reports(tmp_path):
    root=tmp_path/'dataset'
    folder=root/'normalized-pages/train'/('a'*64);folder.mkdir(parents=True)
    keep=['normalized.html','initial.html','initial.png','after.png','report.json','record.json']
    for name in keep:(folder/name).write_text('{}' if name.endswith('.json') else 'input')
    (folder/'before.png').write_bytes(b'diagnostic')
    scroll=folder/'scroll-before';scroll.mkdir()
    (scroll/'0.png').write_bytes(b'zero')
    (scroll/'1.png').write_bytes(b'one')
    (root/'policy-val.jsonl').write_text(json.dumps({'target_image':str(scroll/'1.png')})+'\n')
    rejected=root/'normalized-pages/test'/('b'*64);rejected.mkdir(parents=True)
    (rejected/'report.json').write_text('{"accepted":false}')
    (rejected/'rejected.html').write_text('failed')
    before={name:(folder/name).read_bytes() for name in keep}
    audit=module().diagnostic_cleanup([root],min_age_hours=0)
    assert audit['candidate_count']==3 and audit['referenced_preserved']==1
    assert (folder/'before.png').exists()  # dry run
    result=module().diagnostic_cleanup([root],apply=True,min_age_hours=0)
    assert result['applied'] and not result['errors']
    assert not (folder/'before.png').exists() and not (scroll/'0.png').exists()
    assert not (rejected/'rejected.html').exists() and (rejected/'report.json').exists()
    assert (scroll/'1.png').exists()
    assert all((folder/name).read_bytes()==value for name,value in before.items())
    assert module().diagnostic_cleanup([root],min_age_hours=0)['candidate_count']==0


def test_cleanup_blocks_on_unreadable_metadata_and_honors_age(tmp_path):
    folder=tmp_path/'normalized-pages/val'/('a'*64);folder.mkdir(parents=True)
    image=folder/'before.png';image.write_bytes(b'png')
    assert module().diagnostic_cleanup([tmp_path])['recent_preserved']==1
    bad=tmp_path/'policy-train.jsonl';bad.write_text('broken json')
    report=module().diagnostic_cleanup([tmp_path],True,0)
    assert report['errors'] and not report['applied'] and image.exists()


def test_cleanup_counts_external_hardlinks_as_no_reclaim(tmp_path):
    root=tmp_path/'dataset';folder=root/'normalized-pages/train'/('a'*64);folder.mkdir(parents=True)
    image=folder/'before.png';image.write_bytes(b'png')
    os.link(image,tmp_path/'outside.png')
    report=module().diagnostic_cleanup([root],min_age_hours=0)
    assert report['candidate_count']==1 and report['estimated_reclaimable_bytes']==0


def test_fast_audit_never_hashes_images_and_reports_progress(tmp_path,monkeypatch,capsys):
    tool=module();folder=tmp_path/'normalized-pages/train'/('a'*64);folder.mkdir(parents=True)
    (folder/'before.png').write_bytes(b'large image')
    (folder/'after.png').write_bytes(b'training input')
    monkeypatch.setattr(tool,'digest',lambda p: (_ for _ in ()).throw(AssertionError('audit hashed an image')))
    result=tool.diagnostic_cleanup([tmp_path],min_age_hours=0,log_every=1)
    assert result['candidate_count']==1 and 'sha256' not in result['candidates'][0]
    output=capsys.readouterr().out
    assert all(f'"cleanup_stage": "{stage}"' in output for stage in ('discover','classify','references','complete'))


def test_fast_reference_scan_preserves_relative_and_staging_paths(tmp_path):
    entry=tmp_path/'lazy-train-views'/('a'*64)
    folder=entry/'labels/normalized-pages/train'/('b'*64);folder.mkdir(parents=True)
    for name in ('before.png','reloaded.png'):(folder/name).write_bytes(b'needed')
    stale=tmp_path/'previous-location'/('.'+entry.name+'.temp')/'labels/normalized-pages/train'/folder.name/'before.png'
    (folder/'input.json').write_text(json.dumps({'nested':[{'target':str(stale)}], 'other':'reloaded.png'}))
    result=module().diagnostic_cleanup([tmp_path],apply=True,min_age_hours=0,log_every=1)
    assert result['candidate_count']==0 and result['referenced_preserved']==2
    assert (folder/'before.png').is_file() and (folder/'reloaded.png').is_file()


def test_fast_apply_hashes_only_unreferenced_candidates(tmp_path,monkeypatch):
    tool=module();folder=tmp_path/'normalized-pages/test'/('a'*64);folder.mkdir(parents=True)
    image=folder/'before.png';image.write_bytes(b'diagnostic')
    training=folder/'after.png';training.write_bytes(b'train')
    calls=[];original=tool.digest
    def counted(path):calls.append(path);return original(path)
    monkeypatch.setattr(tool,'digest',counted)
    result=tool.diagnostic_cleanup([tmp_path],True,0,journal=tmp_path/'journal.json')
    assert result['applied'] and calls==[image] and training.read_bytes()==b'train'
    assert result['candidates'][0]['sha256']==hashlib.sha256(b'diagnostic').hexdigest()

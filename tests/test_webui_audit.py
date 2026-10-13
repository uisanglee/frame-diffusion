import io
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from framediff.ir import read_json, read_jsonl
from framediff.webui_visual import scan_sources, import_webui, _assign
from framediff.webui_audit import archive_inventory, remote_inventory, audit


def page(root, ident, view='default_10-10', html=None, size=(10,10), host='example.org'):
    folder=root/ident;folder.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',size).save(folder/f'{view}-screenshot.png')
    if html is not None:(folder/f'{view}-html.html').write_text(html)
    (folder/f'{view}-url.txt').write_text('https://'+host)
    return folder


def test_shared_scan_counts_units_and_precise_exclusions(tmp_path):
    root=tmp_path/'raw';out=tmp_path/'audit'
    page(root,'01',html='unique')
    page(root,'02',html='unique')
    page(root,'03')
    page(root,'04',html='mismatch',size=(11,10))
    folder=page(root,'05',html='broken');(folder/'default_10-10-screenshot.png').write_bytes(b'not PNG')
    page(root,'06',html='a',host='a.example')
    page(root,'06',view='mobile',html='b',host='b.example')
    candidates,_,_,summary=scan_sources(root,['default_10-10','mobile'],out)
    assert len(candidates)==1
    assert summary['screenshot_unique_pages']==6
    assert summary['html_paired_unique_pages']==5
    assert summary['pages_with_no_usable_view']==3
    assert summary['exclusions_by_unit']=={
        'page_view':{'missing_html':1,'viewport_mismatch':1,'unreadable_source':1},
        'unique_page':{'duplicate_html_page':1,'inconsistent_view_host':1}}
    records=list(read_jsonl(out/'source-exclusions.jsonl'))
    dup=next(r for r in records if r['reason']=='duplicate_html_page')
    assert dup['duplicate_of']=='01' and dup['matching_views']==['default_10-10']


def test_import_failure_still_saves_reports(tmp_path):
    root=tmp_path/'raw';page(root,'p',html='hello')
    args=SimpleNamespace(root=str(root),out=str(tmp_path/'out'),views=['default_10-10'],
                         train_count=8,val_count=1,test_count=1,seed=42,resume=True)
    with pytest.raises(ValueError,match='diagnostics:'):import_webui(args)
    assert read_json(tmp_path/'out/source-scan.json')['eligible_unique_pages']==1
    selection=read_json(tmp_path/'out/selection-audit.json')
    assert sum(selection['assigned'].values())==1
    assert sum(selection['missing'].values())==9
    assert (tmp_path/'out/selection-failure.json').exists()
    assert not (tmp_path/'out/manifest.jsonl').exists()


def test_assignment_records_domain_quota_discard():
    groups={'domain':[{'id':str(i)} for i in range(10)]};report={}
    with pytest.raises(ValueError):_assign(groups,{'train':8,'val':1,'test':1},42,report)
    assert report['eligible_unique_pages']==10
    assert len(report['unselected'])==2
    assert all(r['reason']=='domain_split_quota_exhausted' for r in report['unselected'])


def test_archive_listing_missing_and_wrong_size(tmp_path,monkeypatch):
    import framediff.webui_audit as module
    text='''Path = archive.zip
Type = zip

Path = test_split/1/iPhone-13 Pro-screenshot.png
Size = 4

Path = test_split/1/iPhone-13 Pro-html.html
Size = 9

Path = test_split/1/iPhone-13 Pro-url.txt
Size = 2

'''
    class Proc:
        def __enter__(self):self.stdout=io.StringIO(text);return self
        def __exit__(self,*args):pass
        def wait(self):return 0
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:Proc())
    target=tmp_path/'raw/test_split/1';target.mkdir(parents=True)
    (target/'iPhone-13 Pro-html.html').write_bytes(b'bad')
    (target/'iPhone-13 Pro-url.txt').write_bytes(b'ok')
    result=archive_inventory(tmp_path/'archive.zip',tmp_path/'raw',['iPhone-13 Pro'],tmp_path/'report','7z')
    assert result['checked_members']==3
    assert result['counts']=={'missing_extracted_file':1,'extracted_size_mismatch':1,'matched_size':1}
    assert result['extraction_status']=='mismatch'
    assert len(list(read_jsonl(tmp_path/'report/extraction-issues.jsonl')))==2


def test_remote_metadata_identifies_missing_truncated_and_matching(tmp_path,monkeypatch):
    class API:
        def list_repo_tree(self,*args,**kwargs):
            return [SimpleNamespace(path=f'a.zip.{i:03d}',size=10) for i in range(1,4)]
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(HfApi=API))
    (tmp_path/'a.zip.001').write_bytes(b'x'*10)
    (tmp_path/'a.zip.002').write_bytes(b'x'*3)
    result=remote_inventory('webui-70k',tmp_path)
    assert result['counts']=={'size_matches':1,'size_mismatch':1,'missing':1}


def test_audit_unknown_is_not_claimed_complete(tmp_path):
    root=tmp_path/'raw';page(root,'p',html='hello')
    result=audit(SimpleNamespace(root=str(root),out=str(tmp_path/'audit'),views=['default_10-10'],
                 size=10,seed=42,downloads=str(tmp_path/'downloads'),check_remote=False,test_archives=False))
    assert result['selection']['status']=='insufficient'
    for section in result['archives'].values():
        assert section['download_metadata']['status']=='not_checked'
        assert section['integrity']['status']=='not_tested'
        assert section['extraction']['status']=='unknown'


def test_failed_archive_listing_is_not_complete(tmp_path,monkeypatch):
    import framediff.webui_audit as module
    class Proc:
        def __enter__(self):self.stdout=io.StringIO('ERROR: missing volume\n');return self
        def __exit__(self,*args):pass
        def wait(self):return 2
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:Proc())
    result=archive_inventory(tmp_path/'a.zip.001',tmp_path/'raw',['mobile'],tmp_path/'report','7z')
    assert result['status']=='listing_failed'
    assert result['extraction_status']=='unknown'
    assert 'missing volume' in result['tail'][0]


def test_remote_failure_does_not_erase_source_audit(tmp_path,monkeypatch):
    import framediff.webui_audit as module
    def offline(*args):raise OSError('network unavailable')
    monkeypatch.setattr(module,'remote_inventory',offline)
    root=tmp_path/'raw';page(root,'p',html='hello')
    result=audit(SimpleNamespace(root=str(root),out=str(tmp_path/'audit'),views=['default_10-10'],
                 size=10,seed=42,downloads=str(tmp_path/'downloads'),check_remote=True,test_archives=False))
    assert result['source']['eligible_unique_pages']==1
    assert result['archives']['train']['download_metadata']['status']=='unavailable'
    assert (tmp_path/'audit/source-exclusions.jsonl').exists()

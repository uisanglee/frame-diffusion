import hashlib
from types import SimpleNamespace

import pytest

from framediff.candidate_diagnostics import diagnose
from framediff.html_bridge import HtmlBrowser
from framediff import css_owners
from framediff.ir import write_json,write_jsonl,read_json


@pytest.mark.browser
def test_missing_permission_is_traced_to_historical_skip_without_mutation():
    html='''<html data-tuide-inline-actions="targeted-v2"><body>
      <span data-fd-id="a" data-tuide-verify-id="2" style="width:100px">text</span>
      </body></html>'''
    tree={'nodes':[{'id':'root'},{'id':'a','visual_class':1}]}
    with HtmlBrowser() as b:
        b.load(html,[300,200]);before=b.page.content()
        parsed=css_owners.read(b,tree)
        seed=dict(current=tree,css_owners=parsed['owners'],target_declaration_state=parsed['state'])
        detail=diagnose(b,seed,{'normalized_elements':0,
            'preserved_elements':[{'id':'2','reason':'inline_or_table'}]})
        assert b.page.content()==before
    assert detail['fresh_editable_sites']==0
    assert detail['historical_skip_reasons']=={'inline_or_table':1}
    assert detail['field_reasons']=={'missing_editable_attribute':6}
    assert detail['elements'][0]['fields'][0]['inline_value']=='100px'
    assert detail['elements'][0]['fields'][0]['exclusion_evidence']=='inline_or_table'


@pytest.mark.browser
def test_distinguishes_disallowed_empty_invalid_and_stale_values():
    html='''<html data-tuide-inline-actions="targeted-v2"><body><div data-fd-id="a"
      data-tuide-editable-fields="width height margin-top margin-bottom"
      style="width:100px;height:calc(100% - 2px);margin-top:2px"></div></body></html>'''
    tree={'nodes':[{'id':'root'},{'id':'a','visual_class':4}]}
    with HtmlBrowser() as b:
        b.load(html,[300,200]);parsed=css_owners.read(b,tree)
        stored=[[['',''] for _ in range(6)] for _ in range(2)]
        detail=diagnose(b,dict(current=tree,css_owners=parsed['owners'],target_declaration_state=stored))
    fields={r['property']:r['reason'] for r in detail['elements'][0]['fields']}
    assert fields=={'width':'eligible','height':'unsupported_value','margin-top':'eligible',
                    'margin-bottom':'empty_parsed_value','margin-left':'field_not_allowed','margin-right':'field_not_allowed'}
    assert detail['stored_editable_sites']==0 and detail['fresh_editable_sites']==2
    assert detail['stored_state_matches_fresh'] is False
    assert detail['normalization']['report_available'] is False


@pytest.mark.browser
def test_audit_enriches_early_no_editable_exit(tmp_path):
    from framediff.candidate_audit import run
    root=tmp_path/'normalized';out=tmp_path/'audit'
    id='page';folder=root/'normalized-pages/train'/hashlib.sha256(id.encode()).hexdigest()
    folder.mkdir(parents=True)
    html=folder/'normalized.html'
    html.write_text('<html data-tuide-inline-actions="targeted-v2"><body><span data-fd-id="a" data-tuide-verify-id="2">X</span></body></html>')
    tree={'nodes':[{'id':'root'},{'id':'a','visual_class':1}]}
    with HtmlBrowser() as b:
        b.load(html.read_text(),[300,200]);parsed=css_owners.read(b,tree)
    seed=dict(id=id+'/seed',group='g',source_sha='sha',viewport=[300,200],current=tree,
              target_image='unused',target_elements=[],target_html=str(html),
              css_owners=parsed['owners'],target_declaration_state=parsed['state'])
    write_json(folder/'inspection-input.json',seed)
    write_json(folder/'report.json',{'normalized_elements':0,'preserved_elements':[{'id':'2','reason':'inline_or_table'}]})
    write_jsonl(root/'rejections-train.jsonl',[{'id':id,'stage':'visible_candidate_inspection'}])
    run(SimpleNamespace(normalized=str(root),out=str(out),split='train',limit=0,source=None,observation_size=64))
    summary=read_json(out/'summary.json')
    assert summary['reasons']=={'no_editable_css':1}
    assert summary['historical_skip_page_counts']=={'inline_or_table':1}
    assert summary['eligibility_diagnostic_errors']==0

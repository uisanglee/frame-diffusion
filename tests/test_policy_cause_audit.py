import json
from framediff.policy_cause_audit import select_pages, teacher_check, legality, difference, trace_audit
from framediff.tree_edits import FIELDS, EditTokenizer


def state(height):
    return [[["",""] for _ in FIELDS],[["100px",""] if f=='width' else [height,""] if f=='height' else ['0px',''] for f in FIELDS]]


def test_teacher_and_numeric_roundtrip():
    row=dict(declaration_state=state('120px'),target_declaration_state=state('80px'),replacement_edit=[1,'height','80px',''])
    r=teacher_check(row)
    assert r['teacher_improves'] and r['full_symbolic_recovery'] and r['token_roundtrip']
    row['replacement_edit']=[1,'height','18px','']
    assert not teacher_check(row)['teacher_improves']


def test_legality_reports_size_floor():
    tok=EditTokenizer(2)
    assert legality(tok,[1,'height','80px',''],state('120px'),{1:{'height':20}})['allowed']
    assert not legality(tok,[1,'height','18px',''],state('120px'),{1:{'height':20}})['allowed']


def test_page_selection_is_unique_and_stable():
    rows=[dict(id=f'{p}/v{v}',page_id=p) for p in ['a','b','c'] for v in range(3)]
    assert select_pages(rows,2,42)==select_pages(list(reversed(rows)),2,42)
    assert len({r['page_id'] for r in select_pages(rows,2,42)})==2


def test_missing_boxes_not_silent_success():
    r=difference({'a':{'box':[0,0,10,10]}},{'b':{'box':[0,0,10,10]}})
    assert r['missing']==['a'] and r['added']==['b']


def test_trace_repeat_separate_from_geometry(tmp_path):
    p=tmp_path/'abstract-policy-r1-trace.json'
    p.write_text(json.dumps(dict(history=[dict(step=i,action=[1,'height','18px',''],boxes={'a':[0,0,10,18]}) for i in range(3)])))
    r=trace_audit(tmp_path)[0]
    assert r['repeated_action_steps']==[1,2]
    assert r['unchanged_observation_steps']==[1,2]


def test_browser_teacher_and_execution(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff import css_owners
    from framediff.policy_cause_audit import browser_teacher, execute_audit
    tree={'nodes':[{'id':'root'},{'id':'a'}]}
    target='<html><body><div data-fd-id="a" style="height:80px;width:100px;background:red">Text</div></body></html>'
    current=target.replace('height:80px','height:120px')
    path=tmp_path/'target.html';path.write_text(target)
    row=dict(current=tree,viewport=[300,200],current_html_text=current,target_html=str(path))
    with HtmlBrowser() as browser:
        browser.load(target,row['viewport']);parsed=css_owners.read(browser,tree)
        row['target_declaration_state']=parsed['state'];row['css_owners']=parsed['owners']
        browser.load(current,row['viewport']);row['declaration_state']=css_owners.read(browser,tree)['state']
        result=browser_teacher(browser,row,tmp_path/'teacher')
        assert result['full_browser_recovery']
        assert result['pixel_mae']==0
        result=execute_audit(browser,row,[1,'height','18px',''],tmp_path/'execution')
        assert result['prediction_applied']
        assert result['geometry']['changed'][0]['after'][3]==18

from framediff.vlm_first_edit_audit import candidate_edits,first_decrease,context
from framediff.tree_edits import EditTokenizer,FIELDS
from framediff.policy_cause_audit import legality


def test_first_drop_is_not_assumed_first_action():
    assert first_decrease([.1,.2,.2,.15,.3])==3
    assert first_decrease([.1,.1,.2]) is None
    assert first_decrease([.1,.1-1e-10]) is None


def test_search_respects_limits_priority_and_budget():
    state=[[['',''] for _ in FIELDS],[[('80px' if f=='height' else '100px' if f=='width' else '0px'),'important'] for f in FIELDS]]
    tok=EditTokenizer(2);limits={1:{'height':60,'width':90}}
    reference={'nodes':[{'box':[0,0,10,18]},{'box':[0,0,120,90]}]}
    edits=candidate_edits(state,[],1,reference,limits,tok,maximum=10)
    assert len(edits)==10
    for e in edits:
        assert legality(tok,e,state,limits)['allowed']
        assert e[3]=='important'
        assert e[2]!=state[e[0]][FIELDS.index(e[1])][0]


def test_browser_replay_and_scoring_preserve_edit_ids(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff import css_owners
    html='<html><body><div style="height:80px;width:100px;background:red">Text</div></body></html>'
    from framediff.vlm_first_edit_audit import audit_browsers
    with audit_browsers() as (browser,scorer):
        dom=browser.snapshot(html,[300,200],tmp_path/'initial.png',16)
        tree,_,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        source={'current':tree,'viewport':[300,200]}
        row=context(browser,dom['html'],source)
        owner=next(i for i,o in enumerate(row['css_owners']) if o.get('id')=='fd-1')
        css_owners.execute(browser,row['css_owners'],[owner,'height','18px',''])
        edited=browser.page.content()
        scorer.snapshot(edited,[300,200],tmp_path/'after.png',2048)
        after=context(browser,edited,source)
        assert after['css_owners']==row['css_owners']
        assert after['current_boxes']['fd-1'][3]==18
        assert row['current_boxes']['fd-1'][3]==80

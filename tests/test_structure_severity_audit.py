from framediff.structure_severity_audit import scaled_edits, build_page


def test_magnitude_and_sites_are_paired_and_respect_floor():
    sites=[{'edit':[1,'width','100px','important'],'direction':-1},
           {'edit':[2,'margin-left','0px',''],'direction':1}]
    assert scaled_edits(sites,0,[1000,800],{})==[]
    mild=scaled_edits(sites,.1,[1000,800],{})
    strong=scaled_edits(sites,1,[1000,800],{1:{'width':30}})
    assert mild[0][2]=='87.0551px'
    assert strong==[[1,'width','30px','important'],[2,'margin-left','250px','']]
    assert [(e[0],e[1],e[3]) for e in mild]==[(e[0],e[1],e[3]) for e in strong]


def test_browser_structure_and_restoration(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    html='<html><body><div style="width:100px;height:80px;margin-left:10px;background:red">Text</div></body></html>'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[300,200],tmp_path/'original.png',16)
        tree,_,_=dom_tree(dom);annotate(browser,tree,visual_boxes=True)
        target=tmp_path/'original.html';target.write_text(dom['html'])
        source=dict(id='page/view/corruption',split='test',group='test-page',current=tree,
                    viewport=[300,200],target_html=str(target),page_id='page')
        result=build_page(browser,source,tmp_path/'prepared',3,42)
        assert len(result['rows'])==4
        assert all(r['structure_preserved'] and r['teacher_restore_pixel_mae']==0 for r in result['measurements'])
        assert result['measurements'][0]['initial_pixel_mae']==0
        assert result['measurements'][-1]['initial_pixel_mae']>0
        assert result['measurements'][-1]['changed_declarations']==3
        assert result['page']['id']=='page/view'
        from framediff.image_asset_oracle import reference_labels
        assert reference_labels(browser,result['rows'][0])['elements']
        from framediff.fixed_test import boxes
        from pathlib import Path
        browser.load(Path(result['rows'][0]['current_html']).read_text(), [300,200])
        assert boxes(browser,result['rows'][0]['current'],[300,200]) == result['page']['evaluation_target_boxes']


def test_report_includes_failures_and_actual_severity(tmp_path):
    from argparse import Namespace
    from framediff.structure_severity_audit import report, LEVELS
    from framediff.ir import write_json, write_jsonl
    write_json(tmp_path/'inputs/coverage.json', {'selected':1,'paired_pages':1,'failures':[]})
    write_jsonl(tmp_path/'inputs/severity.jsonl', [dict(id='page/view',level=level,
        changed_declarations=2,initial_pixel_mae=value,initial_abstraction_mae=value,
        initial_box_iou=1-value) for level,value in LEVELS.items()])
    write_jsonl(tmp_path/'evaluations/strong/budget-20/results.jsonl', [dict(
        page_id='page/view',method='abstract-oracle',failed=True,
        initial_box_iou=.2,final_box_iou=.2)])
    report(Namespace(out=str(tmp_path)))
    import json
    result=json.loads((tmp_path/'summary.json').read_text())['results'][0]
    assert result['failed']==1 and result['improved']==0 and result['n']==1

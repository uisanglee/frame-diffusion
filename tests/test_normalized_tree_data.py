from types import SimpleNamespace

from framediff.html_bridge import HtmlBrowser
from framediff.ir import read_jsonl, write_jsonl
from framediff.plans import dom_tree
from framediff.tree_normalize import prepare, CONTRACT
from framediff.tree_online import OnlineSampler, target_pool
from framediff.tree_freeze import freeze
from framediff.visual import annotate


def test_normalize_then_corrupt_and_resume(tmp_path, monkeypatch):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'normalized'
    with HtmlBrowser() as browser:
        for split in ('train','val','test'):
            html=f'''<html><head><style>body{{margin:0}}
            .row{{display:flex;gap:20px;width:500px}}
            .card{{width:300px;max-width:200px;height:60px;max-height:50px;background:red}}
            .row .card{{width:250px!important}}</style></head><body>
            <div class="row"><div class="card">{split}</div><div class="card">B</div></div></body></html>'''
            dom=browser.snapshot(html,[600,200],max_nodes=16)
            tree,boxes,_=dom_tree(dom);annotate(browser,tree)
            path=source/f'{split}.html';path.write_text(dom['html'])
            row=dict(id=split+'/clean',group=split,split=split,source_sha=split,
                     current_html=str(path),current=tree,current_boxes=boxes,viewport=[600,200])
            write_jsonl(source/f'policy-{split}.jsonl',[row])
            write_jsonl(source/f'pages-{split}.jsonl',[{**row,'id':split,'html':str(path)}])
    args=SimpleNamespace(rendered=str(source),out=str(out),resume=True,seed=42,
                         max_css_owners=64,observation_size=96,max_pixel_mae=.01,max_box_error=1.)
    prepare(args)
    row=list(read_jsonl(out/'policy-train.jsonl'))[0]
    assert row['normalization_contract']==CONTRACT
    assert row['symbolic_distance']>0
    with HtmlBrowser() as browser:
        from pathlib import Path
        browser.load(Path(row['target_html']).read_text(),row['viewport'])
        assert browser.page.locator('style:not([data-framediff-static])').count()==1
        assert browser.page.locator('.row > .card').count()==2
        assert browser.page.locator('.row').evaluate('e=>getComputedStyle(e).display')=='flex'
        assert browser.page.locator('.card').first.evaluate('e=>e.style.width')=='200px'
        assert browser.page.locator('.card').first.evaluate('e=>getComputedStyle(e).maxWidth')=='none'
        assert browser.page.locator('.card').first.evaluate('e=>getComputedStyle(e).maxHeight')=='none'
        browser.page.locator('.card').first.evaluate("e=>e.style.setProperty('width','300px','important')")
        assert browser.page.locator('.card').first.bounding_box()['width']==300
        assert browser.page.locator('.card').nth(1).bounding_box()['x']==320
    pool=target_pool([row],cache_dir=out/'.online-target-cache',observation_size=96)
    sampler=OnlineSampler(pool,'abstract',max_noise=2)
    try:
        generated=sampler.generate(5)
        assert generated['target_html']==row['target_html']
        assert generated['explicit_size_limits']
        assert generated['online']['abstraction_max_difference']>0
    finally:sampler.close()
    import torch
    from framediff.css_owners import CONTRACT as OWNER_CONTRACT
    from framediff.tree_policy import TreePolicy,TreeConfig,batch_inputs
    torch.set_num_threads(1)
    net=TreePolicy(TreeConfig(size=96,hidden=32,layers=1,heads=4,max_nodes=64,
                             stylesheets=True,action_contract=OWNER_CONTRACT),pretrained=False)
    loss,_=net.loss(*batch_inputs(net,[generated],'cpu'),[generated['replacement_edit']])
    assert torch.isfinite(loss)
    loss.backward()
    assert net.output.weight.grad is not None
    monkeypatch.setattr('framediff.tree_normalize.convert',lambda *a,**k: (_ for _ in ()).throw(AssertionError('renormalized')))
    prepare(args)
    assert list(read_jsonl(out/'policy-train.jsonl'))==[row]
    frozen=tmp_path/'fixed'
    freeze(SimpleNamespace(rendered=str(out),out=str(frozen),samples_per_page=1,
                           max_noise=2,seed=90210,observation_size=96,resume=True))
    for split in ('train','val','test'):
        fixed=list(read_jsonl(frozen/f'policy-{split}.jsonl'))[0]
        assert '/normalized-pages/' in fixed['target_html']
        assert fixed['explicit_size_limits']

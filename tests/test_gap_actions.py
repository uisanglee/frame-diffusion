import random

import pytest

from framediff.tree_edits import FIELDS, EditTokenizer, value_valid, value_prefix, repair_path
from framediff.tree_online import sample_mutation


@pytest.mark.parametrize('field', ['row-gap', 'column-gap'])
def test_gap_grammar_and_corruption(field):
    assert value_valid('normal', field)
    assert value_valid('10px', field) and value_valid('5%', field)
    assert not value_valid('-1px', field) and not value_valid('auto', field)
    assert not value_prefix('-', field)
    tok=EditTokenizer(4)
    edit=[1,field,'10px','important'];prefix=[]
    for token in tok.encode(edit,2):
        assert token in tok.allowed(prefix,2)
        prefix.append(token)
    assert tok.decode(prefix,2)==edit
    state=[[['',''] for _ in FIELDS] for _ in range(2)]
    state[1][FIELDS.index(field)]=['10px','important']
    for i in range(100):
        mutation=sample_mutation(state,[400,300],random.Random(i))
        assert mutation[:2]==[1,field] and mutation[3]=='important'
        assert value_valid(mutation[2],field) and mutation[2]!='10px'


def test_gap_diffusion_grid_is_nonnegative():
    from framediff.css_diffusion import numeric_slots, slot_edit, grid
    state=[[['',''] for _ in FIELDS] for _ in range(2)]
    for field in ('row-gap','column-gap'):state[1][FIELDS.index(field)]=['16px','']
    slots,labels,_=numeric_slots(state,FIELDS,129)
    assert len(slots)==2
    for slot,label in zip(slots,labels):
        edit=slot_edit(slot,label,FIELDS,129)
        assert edit[2]=='16px' and grid(edit[1],'px',129).min()==0


@pytest.mark.browser
def test_gap_shorthand_repair_preserves_grid_and_parent(tmp_path):
    from framediff import css_owners
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    html='''<html><head><style>
      .parent {display:grid;width:420px;grid-template-columns:1fr 1fr;gap:10px 20px!important}
      .child {height:30px;background:red}
      </style></head><body><div class="parent" id="p">
      <div class="child" id="a"></div><div class="child" id="b"></div>
      <div class="child" id="c"></div></div></body></html>'''
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[600,300],max_nodes=16)
        tree,_,_=dom_tree(dom);target=css_owners.read(browser,tree)
        owner=next(i for i,o in enumerate(target['owners']) if o.get('selector')=='.parent')
        assert target['state'][owner][FIELDS.index('row-gap')]==['10px','important']
        assert target['state'][owner][FIELDS.index('column-gap')]==['20px','important']
        before=browser.page.locator('#c').bounding_box()
        css_owners.execute(browser,target['owners'],[owner,'row-gap','50px','important'])
        assert browser.page.locator('#c').bounding_box()['y']==pytest.approx(before['y']+40)
        # Export and reload must preserve the unedited half of the shorthand.
        browser.load(browser.page.content(),[600,300])
        state=css_owners.read(browser,tree)
        assert state['state'][owner][FIELDS.index('column-gap')]==['20px','important']
        path=repair_path(state['state'],target['state'])
        assert path==[[owner,'row-gap','10px','important']]
        css_owners.execute(browser,state['owners'],path[0])
        assert browser.page.locator('#c').bounding_box()['y']==pytest.approx(before['y'])
        assert browser.page.locator('#c').evaluate('e=>e.parentElement.id')=='p'
        assert browser.page.locator('#p').evaluate('e=>getComputedStyle(e).display')=='grid'
        assert css_owners.read(browser,tree)['state']==target['state']


@pytest.mark.browser
def test_online_gap_only_page_has_gap_repair_labels(tmp_path):
    from framediff import css_owners
    from framediff.html_bridge import HtmlBrowser
    from framediff.plans import dom_tree
    from framediff.visual import annotate, elements
    from framediff.tree_online import OnlineSampler
    html='''<html><head><style>.p {display:flex;gap:10px}</style></head>
    <body><div class="p"><button>A</button><button>B</button></div></body></html>'''
    path=tmp_path/'clean.html';image=tmp_path/'clean.png'
    with HtmlBrowser() as browser:
        dom=browser.snapshot(html,[400,300],image,max_nodes=16)
        tree,boxes,_=dom_tree(dom);annotate(browser,tree)
        parsed=css_owners.read(browser,tree);path.write_text(dom['html'])
    target=dict(id='gap',group='train',split='train',source_sha='gap',current=tree,
        viewport=[400,300],target_html=str(path),target_image=str(image),
        target_elements=elements(tree,boxes,[400,300]),css_owners=parsed['owners'])
    sampler=OnlineSampler([target],'abstract',seed=7,max_noise=1,attempts=16)
    try:
        row=sampler.generate(0)
        assert row['replacement_edit'][1:] == ['column-gap','10px','']
        assert row['replacement_edit'] in repair_path(row['declaration_state'],parsed['state'])
    finally:sampler.close()
    import torch
    from framediff.tree_policy import TreeConfig, TreePolicy, batch_inputs
    torch.set_num_threads(2)
    cfg=TreeConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=32,
                   stylesheets=True,action_contract=css_owners.CONTRACT,existing_values_only=True)
    policy=TreePolicy(cfg,False)
    loss,_=policy.loss(*batch_inputs(policy,[row],'cpu'),[row['replacement_edit']])
    loss.backward()
    assert torch.isfinite(loss) and policy.output.weight.grad.abs().sum()>0

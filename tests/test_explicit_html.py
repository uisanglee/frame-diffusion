import pytest
import json
from pathlib import Path
from types import SimpleNamespace

from framediff.explicit_html import convert
from framediff.html_bridge import HtmlBrowser


@pytest.mark.browser
@pytest.mark.parametrize('mode',['boxes'])
def test_padding_removed_with_border_children_and_direct_text_preserved(tmp_path,mode):
    html='''<html><body style="margin:0"><div id="p" style="position:absolute;left:30px;top:20px;
    width:250px;height:120px;padding:20px 30px;border:5px solid red;border-radius:8px;
    font:16px/24px Arial">Hello<div id="child" style="width:30px;height:20px;background:blue"></div></div></body></html>'''
    with HtmlBrowser() as browser:
        browser.load(html,[600,300])
        before=browser.page.locator('#child').bounding_box()
        text_before=browser.page.locator('#p').evaluate('''e=>{const r=document.createRange();
          r.selectNodeContents(e.firstChild);const b=r.getBoundingClientRect();return [b.x,b.y];}''')
        report=convert(browser,html,[600,300],tmp_path,mode=mode)
        assert report['accepted'],report
        assert browser.page.locator('#child').bounding_box()==before
        style=browser.page.locator('#p').evaluate('''e=>{const s=getComputedStyle(e);
          return [s.paddingTop,s.paddingLeft,s.borderTopWidth,s.borderTopColor,s.borderRadius]}''')
        assert style==['0px','0px','5px','rgb(255, 0, 0)','8px']
        text_after=browser.page.locator('tuide-text').evaluate('''e=>{const r=document.createRange();
          r.selectNodeContents(e);const b=r.getBoundingClientRect();return [b.x,b.y];}''')
        assert text_after==text_before
        browser.page.locator('#p').evaluate("e=>e.style.width='20px'")
        assert browser.page.locator('#p').bounding_box()['width']==20


@pytest.mark.browser
def test_bakes_cascade_constraints_and_decouples_siblings(tmp_path):
    html='''<html><head><style>
    body{margin:0}.row{display:flex;gap:20px;width:500px}
    .card{width:300px;height:60px;max-width:200px;background:red}
    .row .card{width:250px!important}.card.b{background:blue}
    </style></head><body><div class="row"><div class="card a">A</div><div class="card b">B</div></div></body></html>'''
    with HtmlBrowser() as browser:
        report=convert(browser,html,[600,200],tmp_path)
        assert report['accepted'],report
        assert browser.page.locator('style:not([data-framediff-static])').count()==0
        def boxes():
            return browser.page.locator('.card').evaluate_all('els=>els.map(e=>{const r=e.getBoundingClientRect();return [r.x,r.width]})')
        assert boxes()==[[0,200],[220,200]]
        browser.page.locator('.a').evaluate("e=>e.style.width='300px'")
        assert boxes()==[[0,300],[220,200]]
        browser.page.locator('.b').evaluate("e=>e.style.left='320px'")
        assert boxes()==[[0,300],[320,200]]
        # New inline styles must remain legal input to the existing CSS owner reader.
        from framediff.css_owners import read
        browser.page.locator('.card').evaluate_all("els=>els.forEach((e,i)=>e.setAttribute('data-fd-id','n'+i))")
        parsed=read(browser,{'nodes':[{'id':'root'},{'id':'n0'},{'id':'n1'}]})
        assert len(parsed['owners'])==3


@pytest.mark.browser
def test_rejects_generated_content_instead_of_silently_losing_it(tmp_path):
    html='<html><head><style>div:before{content:"icon"}</style></head><body><div>Text</div></body></html>'
    with HtmlBrowser() as browser,pytest.raises(Exception,match='pseudo-element'):
        convert(browser,html,[400,200],tmp_path)
    assert not (tmp_path/'normalized.html').exists()


@pytest.mark.browser
def test_border_floor_is_enforced_by_both_edit_executors(tmp_path):
    from framediff.plans import dom_tree
    from framediff.tree_edits import execute
    from framediff import css_owners
    html='<html><body style="margin:0"><div id="a" style="width:100px;height:100px;border:5px solid red"></div></body></html>'
    with HtmlBrowser() as browser:
        assert convert(browser,html,[400,300],tmp_path,mode='boxes')['accepted']
        snapshot=browser.snapshot(browser.page.content(),[400,300],max_nodes=16)
        tree,_,_=dom_tree(snapshot);id=browser.page.locator('#a').get_attribute('data-fd-id')
        index=next(i for i,n in enumerate(tree['nodes']) if n['id']==id)
        owners=css_owners.read(browser,tree)['owners']
        for perform in (lambda value:execute(browser,tree,[index,'width',value,'']),
                        lambda value:css_owners.execute(browser,owners,[index,'width',value,''])):
            for value in ('5px','auto','1%',''):
                with pytest.raises(ValueError,match='Explicit'):perform(value)
            perform('10px')
            assert browser.page.locator('#a').bounding_box()['width']==10
        with pytest.raises(ValueError,match='Explicit'):
            execute(browser,tree,[index,'height','9px',''])
        execute(browser,tree,[index,'height','10px',''])
        assert browser.page.locator('#a').bounding_box()['height']==10
        html=browser.page.content()
        for perform,delta in ((browser.edit_property,-1),(browser.edit_visual_action,-.0125)):
            with pytest.raises(Exception,match='Explicit'):
                perform(html,[400,300],id,'width',delta)


def test_diffusion_slot_grid_respects_border_minimum():
    from framediff.css_diffusion import numeric_slots,slot_edit
    from framediff.tree_edits import FIELDS
    state=[[['',''] for _ in FIELDS],[['20px',''],['30px',''],*[[ '', '' ] for _ in FIELDS[2:]]]]
    slots,labels,_=numeric_slots(state,FIELDS,129,{1:{'width':10.,'height':12.}})
    for slot in slots:
        for i in range(129):
            assert float(slot_edit(slot,i,FIELDS,129)[2][:-2])>=slot['minimum']
    assert len(labels)==2


@pytest.mark.browser
def test_parent_preserving_dataset_and_rollout(tmp_path):
    import torch
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.visual_data import build
    from framediff.tree_policy import TreeConfig,TreePolicy
    from framediff.visual_experiment import rollout
    sources=[]
    with HtmlBrowser() as browser:
        for i,split in enumerate(('train','val','test')):
            folder=tmp_path/split
            html=f'<html><body style="margin:0"><div id="p" style="width:{180+i}px;height:150px"><div id="c" style="width:50px;height:40px;background:red;border:2px solid black"></div></div></body></html>'
            assert convert(browser,html,[400,300],folder,mode='boxes')['accepted']
            sources.append(dict(id=split,group=split,split=split,html=str(folder/'normalized.html'),viewport=[400,300]))
    manifest=tmp_path/'manifest.jsonl';write_jsonl(manifest,sources)
    rendered=tmp_path/'rendered'
    build(SimpleNamespace(manifest=str(manifest),out=str(rendered),resume=False,trajectories=1,
                          max_noise=1,max_nodes=16,abstract_size=32,seed=42))
    row=next(read_jsonl(rendered/'policy-train.jsonl'))
    assert 'hierarchy' not in row
    torch.set_num_threads(2)
    net=TreePolicy(TreeConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=16),False)
    # Deterministic valid edit; integration test is not a learned-policy quality test.
    index=next(i for i,n in enumerate(row['current']['nodes']) if i and n['id']!='__viewport__')
    net.predict=lambda *args:[index,'width','200px','important']
    with HtmlBrowser() as browser:
        html,stats,_=rollout(browser,Path(row['current_html']).read_text(),row['current'],row['viewport'],
                            row['target_image'],net,steps=1,oracle_elements=row['target_elements'])
        assert not stats['failed']
        assert 'hierarchy_restoration' not in stats
        assert 'data-tuide-flat-box' not in html
        assert browser.page.locator('#c').evaluate('e=>e.parentElement.id')=='p'


def test_size_constraints_prune_tokens_before_execution():
    from framediff.tree_edits import EditTokenizer,bounded_px_prefix
    tok=EditTokenizer(4);limits={1:{'width':10.,'height':12.}}
    def allowed(tokens):return [tok.tokens[i] for i in tok.allowed([tok.ids[t] for t in tokens],3,size_limits=limits)]
    assert allowed(['N1','width'])==['SET']
    assert 'a' not in allowed(['N1','width','SET'])
    assert 'p' not in allowed(['N1','width','SET','5'])
    assert '.' not in allowed(['N1','width','SET','5'])
    assert 'p' in allowed(['N1','width','SET','1','0'])
    assert bounded_px_prefix('10p',10) and not bounded_px_prefix('9.8',10)
    for value in ('10px','10.5px','100px'):
        prefix=[]
        for token in tok.encode([1,'width',value,''],3):
            assert token in tok.allowed(prefix,3,size_limits=limits)
            prefix.append(token)


def test_symbolic_validation_enforces_execution_size_floor(monkeypatch):
    import contextlib
    import copy
    import framediff.tree_policy as policy
    from framediff.tree_edits import FIELDS
    current=[[['',''] for _ in FIELDS] for _ in range(2)]
    current[1][0]=['30px',''];target=copy.deepcopy(current);target[1][0]=['20px','']
    row={'current':{'nodes':[{'id':'root'},{'id':'a'}]},'declaration_state':current,
         'explicit_size_limits':{'a':{'width':10,'height':10}}}
    class Net:
        def predict(self,*args):return [1,'width','5px','']
    monkeypatch.setattr(policy,'batch_inputs',lambda *args:(None,None,None,None))
    result=policy.symbolic_policy_metrics(Net(),[row],[target],'cpu',False,contextlib.nullcontext)
    assert result['invalid_action_rate']==1
    assert result['symbolic_no_change_action_rate']==0


def test_greedy_decoder_cannot_finish_below_border_floor(monkeypatch):
    import torch
    from framediff.tree_policy import TreeConfig,TreePolicy
    from framediff.tree_edits import FIELDS
    torch.set_num_threads(2)
    net=TreePolicy(TreeConfig(size=32,hidden=16,layers=1,heads=2,token_grid=2,max_nodes=4),False).eval()
    tok=net.tokenizer
    monkeypatch.setattr(net,'memory',lambda *args:(None,None))
    def logits(memory,padding,tokens):
        p=[tok.tokens[i] for i in tokens[0].tolist()[1:]]
        value=''.join(p[3:])
        preferred=('N1' if not p else 'width' if len(p)==1 else 'SET' if len(p)==2
                   else 'EOS' if p[-1]=='NORMAL' else '5' if value==''
                   else 'p' if value in ('5','50') else 'x' if value=='50p' else 'NORMAL')
        scores=torch.full((1,1,len(tok.tokens)),-100.)
        scores[0,0,tok.ids['0']]=10.
        scores[0,0,tok.ids[preferred]]=100.
        return scores
    monkeypatch.setattr(net,'logits',logits)
    state=[[['',''] for _ in FIELDS] for _ in range(2)]
    image=torch.zeros(1,4,32,32)
    action=net.predict({'explicit_owner_limits':[{1:{'width':10.,'height':10.}}]},image,image,[state])
    assert action==[1,'width','50px','']


def test_nested_same_class_boxes_have_distinct_boundary_features():
    import torch
    from framediff.visual import semantic_masks
    parent={'box':[0,0,300,200],'label':4}
    a={'box':[20,20,80,80],'label':4};b={'box':[160,100,270,180],'label':4}
    x=semantic_masks([parent,a],[320,240],384);y=semantic_masks([parent,b],[320,240],384)
    assert x.shape==(8,384,384)
    assert torch.equal(x[:4],y[:4])  # Original representation loses the change.
    assert not torch.equal(x[4:],y[4:])
    assert torch.equal(x,semantic_masks([a,parent],[320,240],384))
    assert x.isfinite().all() and x.min()>=0 and x.max()<=1


@pytest.mark.browser
def test_button_text_stays_inside_control_and_no_restore_command(tmp_path):
    from framediff.plans import dom_tree
    from framediff.visual import annotate
    from framediff.cli import main
    html='<html><body style="margin:0"><button id="b" style="width:140px;height:60px;padding:10px;border:1px solid black"><span id="text">Hello</span></button></body></html>'
    with HtmlBrowser() as browser:
        assert convert(browser,html,[400,300],tmp_path,mode='boxes')['accepted']
        assert browser.page.locator('#text').evaluate('e=>e.parentElement.id')=='b'
        snap=browser.snapshot(browser.page.content(),[400,300],max_nodes=16)
        tree,_,_=dom_tree(snap);annotate(browser,tree)
        assert [n['visual_class'] for n in tree['nodes']].count(3)==1
        assert 1 not in [n['visual_class'] for n in tree['nodes']]
        with pytest.raises(ValueError,match='separation was removed'):
            convert(browser,html,[400,300],tmp_path/'flat',mode='flat')
        with pytest.raises(ValueError,match='no longer supported'):
            browser.load('<html data-tuide-flat><body></body></html>',[400,300])
    with pytest.raises(SystemExit):main(['visual-restore-hierarchy','--help'])


def test_reject_old_abstract_checkpoint(tmp_path):
    import torch
    from framediff.tree_policy import KIND
    from framediff.visual import load_policy
    path=tmp_path/'old.pt'
    torch.save({'kind':KIND,'config':{'mode':'abstract'},'model':{}},path)
    with pytest.raises(ValueError,match='occupancy-only'):load_policy(path,'cpu')

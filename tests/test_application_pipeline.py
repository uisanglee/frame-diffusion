from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup
from PIL import Image

from framediff.app_browser import AppBrowser, freeze_application
from framediff.application_repair import restore_application
from framediff.web_experiment import prepare


APP = '''<!DOCTYPE html><html><head><style>body{margin:0}button{width:100px;height:40px}</style></head>
<body><div id="root"></div><script>
let count=0;
function render(){document.getElementById('root').innerHTML='<button id="buy">Buy</button><span id="count">'+count+'</span>';
document.getElementById('buy').onclick=()=>{count++;render();};}render();
</script></body></html>'''


@pytest.mark.browser
def test_prepare_executes_only_ui2_application_and_keeps_original_source(tmp_path,monkeypatch):
    Image.new('RGB',(320,240),'white').save(tmp_path/'a.png')
    (tmp_path/'a.html').write_text('<html><body>Reference</body></html>')
    prompts=[]
    def generate(args,prompt,*unused):
        prompts.append(prompt)
        if args.task=='revise-html':
            assert 'function render()' in Path(args.current).read_text()
        return APP,{}
    monkeypatch.setattr('framediff.vlm.generate',generate)
    args=SimpleNamespace(root=str(tmp_path),manifest=None,out=str(tmp_path/'out'),rounds=1,
        max_nodes=128,max_pixels=1048576,max_new_tokens=4096,limit=0,seed=42,resume=False,
        backend='openai-compatible',model='zai-org/UI2Code_N',revision='main',endpoint='http://unused',
        api_key_env='TEST_KEY',four_bit=False,initial_mode='text-augmented',
        revision_protocol='design2code',repair_conditioning='visual',vlm_retries=0)
    row=prepare(args)[0]
    assert not row['errors']
    for method in row['methods'].values():
        assert not method['failed']
        assert Path(method['application_html']).read_text()==APP
        assert '<script>' not in Path(method['html']).read_text()
        assert method['application_capture']['pixel_mae']<=.001
    assert all('disables JavaScript' not in p and 'Do not use React' not in p for p in prompts)
    assert all('no external URLs and no JavaScript' not in p for p in prompts)


@pytest.mark.browser
def test_policy_edit_roundtrip_keeps_application_behavior(tmp_path):
    capture=freeze_application(APP,[320,240],tmp_path/'initial')
    initial={'html':capture['static_html'],'application_html':capture['source_html']}
    soup=BeautifulSoup(Path(initial['html']).read_text(),'html.parser')
    soup.select_one('#buy')['style']='width: 180px !important; margin-top: 8px !important;'
    restored=restore_application(initial,str(soup),[320,240],tmp_path/'repaired')
    assert restored['application_patch_verification']['accepted']
    source=Path(restored['application_html']).read_text()
    assert 'function render()' in source
    with AppBrowser() as browser:
        browser.load(source,[320,240])
        browser.page.locator('#buy').click()
        browser.page.wait_for_function("document.querySelector('#count').textContent==='1' && getComputedStyle(document.querySelector('#buy')).width==='180px'")
        assert browser.page.evaluate('window.__tuideApplicationEdits.errors')==[]


@pytest.mark.browser
def test_policy_edit_roundtrip_refuses_nonlayout_change(tmp_path):
    capture=freeze_application(APP,[320,240],tmp_path/'initial')
    initial={'html':capture['static_html'],'application_html':capture['source_html']}
    with pytest.raises(ValueError,match='Unsupported application'):
        restore_application(initial,Path(initial['html']).read_text().replace('Buy','Different'),[320,240],tmp_path/'bad')


@pytest.mark.browser
@pytest.mark.parametrize('invalid_edit',[False,True])
def test_visual_evaluation_delivers_live_application_or_retains_initial(tmp_path,monkeypatch,invalid_edit):
    import torch
    from framediff import visual_experiment as experiment
    from framediff.cli import main
    from framediff.ir import read_jsonl,write_jsonl
    from framediff.visual import CONTRACT,ACTION_CONTRACT

    capture=freeze_application(APP,[320,240],tmp_path/'initial')
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.cfg=SimpleNamespace(mode='abstract',semantic=True)
    monkeypatch.setattr(experiment,'load_policy',lambda *a:(Policy(),{}))
    monkeypatch.setattr(experiment,'load_detector',lambda *a:(None,{}))
    def rollout(browser,html,*args,**kwargs):
        soup=BeautifulSoup(html,'html.parser')
        if invalid_edit:soup.select_one('#buy').string='Wrong content'
        else:soup.select_one('#buy')['style']='width:180px!important'
        return str(soup),dict(failed=False,error=None,seconds=.01,history=[]),None
    monkeypatch.setattr(experiment,'rollout',rollout)
    for name in ('policy.pt','detector.pt'):(tmp_path/name).write_text('fixture')
    data=tmp_path/'prepared.jsonl'
    write_jsonl(data,[{'id':'page','group':'test','current':{},'viewport':[320,240],
        'screenshot':capture['screenshot'],'tagged_html':capture['static_html'],
        'visual_contract':CONTRACT,'visual_action_contract':ACTION_CONTRACT,
        'methods':{'initial':{'html':capture['static_html'],'application_html':capture['source_html'],
            'application_capture':capture,'failed':False,'seconds':1.,'vlm_calls':1}}}])
    main(['visual-evaluate','--data',str(data),'--abstract-only',
        '--abstract-checkpoint',str(tmp_path/'policy.pt'),'--detector-checkpoint',str(tmp_path/'detector.pt'),
        '--out',str(tmp_path/'out'),'--device','cpu','--warmup','0','--repeats','1'])
    method=list(read_jsonl(tmp_path/'out/results.jsonl'))[0]['methods']['abstract-policy']
    assert method['failed'] is invalid_edit
    if invalid_edit:
        assert method['application_html']==capture['source_html']
        assert Path(method['html']).read_text()==Path(capture['static_html']).read_text()
    else:
        assert method['application_patch_verification']['accepted']
        assert 'function render()' in Path(method['application_html']).read_text()
        assert method['repair_seconds']>.01 and method['browser_executions']==4

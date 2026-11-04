import base64
import io
import pytest
from bs4 import BeautifulSoup
from PIL import Image
from framediff.webui_placeholders import convert_html,embed_placeholders,CONTRACT,PROMPT


def uri(w,h):
    buffer=io.BytesIO();Image.new('RGB',(w,h),'white').save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def test_ordinary_html_unchanged():
    text='<html><body><img src="rick.jpg"></body></html>'
    assert embed_placeholders(text)==text
    assert all(f'rick{i}.jpg' in PROMPT for i in range(1,6))


@pytest.mark.browser
def test_intrinsic_size_cycle_and_geometry(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    original='<html><body>'+''.join(f'<img src="{uri(60+i,30+i)}">' for i in range(7))+'</body></html>'
    with HtmlBrowser() as browser:
        result,mapping=convert_html(original,'',tmp_path,browser,[800,600])
        assert CONTRACT in result
        soup=BeautifulSoup(result,'html.parser')
        assert [x['src'] for x in soup.find_all('img')]==['rick1.jpg','rick2.jpg','rick3.jpg','rick4.jpg','rick5.jpg','rick1.jpg','rick2.jpg']
        assert [m['width'] for m in mapping]==list(range(60,67))
        assert browser.page.evaluate('()=>document.images[0].naturalWidth')==60
        assert browser.page.evaluate('()=>getComputedStyle(document.images[0]).width')=='60px'


@pytest.mark.browser
def test_responsive_picture_and_css_background(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    original=f'''<html><body><picture><source srcset="{uri(200,100)}"><img src="{uri(40,40)}" style="width:100px;height:auto"></picture>
    <div style="width:50px;height:50px;background-image:url('{uri(20,10)}')"></div></body></html>'''
    with HtmlBrowser() as browser:
        result,mapping=convert_html(original,'',tmp_path,browser,[800,600])
        assert mapping[0]['width']==200
        assert browser.page.evaluate('()=>document.images[0].getBoundingClientRect().height')==50
        assert 'data:image/svg+xml' in result


def test_polishing_prompt_assets():
    from baselines.ui2coden.runner import polishing_prompt
    assert PROMPT in polishing_prompt('<img src="rick2.jpg">',[800,600],'layout')
    assert PROMPT not in polishing_prompt('<img src="rick.jpg">',[800,600],'layout')


@pytest.mark.browser
def test_input_image_uses_intrinsic_not_css_size(tmp_path):
    from framediff.html_bridge import HtmlBrowser
    original=f'<html><body><input type="image" src="{uri(120,60)}" style="width:80px;height:auto"></body></html>'
    with HtmlBrowser() as browser:
        result,mapping=convert_html(original,'',tmp_path,browser,[800,600])
        assert (mapping[0]['width'],mapping[0]['height'])==(120,60)
        assert 'rick1.jpg' in result
        assert browser.page.locator('input').bounding_box()['height']==40


@pytest.mark.browser
def test_only_active_image_sources_downloaded(tmp_path,monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    calls=[]
    def fetch(url,*args):
        calls.append(url)
        assert url in ('https://example.test/small.png','https://example.test/background.png')
        return dict(ok=True,uri=uri(100,50))
    monkeypatch.setattr('framediff.image_asset_pilot.fetch_image',fetch)
    original='''<html><head><style>
      .unused {background-image:url(unused.png)}
      @media(min-width:1200px) {body {background-image:url(wide.png)}}
      body {background-image:url(background.png)}
      </style></head><body><picture>
      <source media="(min-width:1200px)" srcset="missing-wide.png">
      <img src="missing-fallback.png" srcset="small.png 1x, missing-retina.png 2x">
      </picture></body></html>'''
    with HtmlBrowser() as browser:
        _,mapping=convert_html(original,'https://example.test/',tmp_path,browser,[800,600])
    assert len(calls)==2 and mapping[0]['width']==100


@pytest.mark.browser
def test_required_image_failure_still_rejects(tmp_path,monkeypatch):
    from framediff.html_bridge import HtmlBrowser
    monkeypatch.setattr('framediff.image_asset_pilot.fetch_image',lambda *a:dict(ok=False,error='404'))
    with HtmlBrowser() as browser,pytest.raises(ValueError,match='Image unavailable'):
        convert_html('<html><body><img src="missing.png"></body></html>',
                     'https://example.test/',tmp_path,browser,[800,600])


@pytest.mark.browser
def test_prepared_source_reaches_detector_and_policy(tmp_path):
    from types import SimpleNamespace
    from framediff.webui_placeholders import prepare
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.visual_data import _render_page
    raw=tmp_path/'raw.html';raw.write_text(f'<html><body><img src="{uri(100,60)}"><p>Title</p></body></html>')
    manifest=tmp_path/'input.jsonl'
    write_jsonl(manifest,[dict(id='page/default',page_id='page',group='heldout',split='train',
                             source_sha='old',html=str(raw),viewport=[320,240])])
    out=tmp_path/'source';args=SimpleNamespace(manifest=str(manifest),out=str(out))
    prepare(args);prepare(args)
    source=list(read_jsonl(out/'manifest.jsonl'))[0]
    settings=SimpleNamespace(lazy_train=True,seed=42,max_nodes=512,abstract_size=384,min_elements=1,
                             max_source_mae=1.,resume=True,clean_only=True)
    lazy,_,_=_render_page((settings,tmp_path/'rendered',0,source))
    assert lazy['policy'][0]['lazy_source']['html']==source['html']
    settings.lazy_train=False
    materialized,error,_=_render_page((settings,tmp_path/'rendered',0,source))
    assert error is None
    assert materialized['detection'][0]['elements']
    policy=materialized['policy'][0]
    assert CONTRACT in __import__('pathlib').Path(policy['current_html']).read_text()
    assert materialized['detection'][0]['target_image']==policy['target_image']


@pytest.mark.browser
def test_color_script_prepares_train_before_training(tmp_path,monkeypatch):
    import os
    import subprocess
    from pathlib import Path
    from framediff.ir import write_jsonl,read_jsonl
    from framediff.tree_online import target_pool
    source=[]
    for split in ('train','val','test'):
        html=tmp_path/f'{split}.html'
        html.write_text(f'''<html><body style="margin:0"><img src="{uri(100,50)}">
          <h1>{split}</h1><div style="width:150px;height:60px;background:red">One</div>
          <div style="width:180px;height:60px;background:blue">Two</div></body></html>''')
        source.append(dict(id=split,page_id=split,group=split,split=split,html=str(html),viewport=[400,300]))
    manifest=tmp_path/'source.jsonl';write_jsonl(manifest,source)
    output=tmp_path/'prepared'
    env={**os.environ,'SOURCE_MANIFEST':str(manifest),'COLOR_DATA':str(output),
         'DATA_SIZE':'3','PREPARE_WORKERS':'1'}
    root=Path(__file__).resolve().parents[1]
    for _ in range(2):
        result=subprocess.run(['bash','scripts/prepare_webui_color_images.sh'],cwd=root,env=env,
                              capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
    detector=list(read_jsonl(output/'3/rendered/detector-train.jsonl'))
    train=list(read_jsonl(output/'3/css/policy-train.jsonl'))
    assert len(detector)==len(train)==1
    assert detector[0]['elements'] and Path(detector[0]['image']).is_file()
    assert not detector[0].get('lazy_contract') and not train[0].get('lazy_contract')
    assert train[0]['clean_target_only'] and train[0]['visible_candidates']
    assert CONTRACT in Path(train[0]['target_html']).read_text()
    assert Path(train[0]['target_image']).is_file()
    assert not list(output.glob('master/labels/normalized-pages/train/*/initial.html'))
    def unexpected(*a,**kw):raise AssertionError('Repeated candidate inspection')
    monkeypatch.setattr('framediff.tree_online.inspect_candidates',unexpected)
    diagnostics=[]
    pool=target_pool(train,output/'3/css/pages-train.jsonl',cache_dir=tmp_path/'new-run-cache',
                     diagnostics=diagnostics)
    assert pool and diagnostics[0]['cache_hit']
    for split in ('val','test'):
        row=list(read_jsonl(output/f'3/css/policy-{split}.jsonl'))[0]
        assert row['replacement_edit'] is not None and Path(row['current_image']).is_file()

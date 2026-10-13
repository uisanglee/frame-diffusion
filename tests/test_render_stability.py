import copy
import io

import pytest
from PIL import Image

from framediff.html_bridge import HtmlBrowser
from framediff.explicit_html import compare_renders


def png(color='white'):
    buf=io.BytesIO();Image.new('RGB',(100,100),color).save(buf,format='PNG')
    return buf.getvalue()


def test_non_target_geometry_is_diagnostic_but_local_pixels_are_protected():
    before=[dict(id='a',fd_id=None,box=[0,0,10,10],visible_clip=[0,0,10,10])]
    after=copy.deepcopy(before);after[0]['box'][3]=18
    report=compare_renders(before,after,png(),png(),geometry_target_ids={'target'})
    assert report['accepted'] and report['non_target_geometry_changes']
    image=Image.new('RGB',(100,100),'white')
    image.paste('black',(0,0,10,10));buf=io.BytesIO();image.save(buf,format='PNG')
    report=compare_renders(before,after,png(),buf.getvalue(),geometry_target_ids={'target'})
    assert not report['accepted'] and report['element_failures']
    before[0]['fd_id']='target';after[0]['fd_id']='target'
    assert not compare_renders(before,after,png(),png(),geometry_target_ids={'target'})['accepted']


@pytest.mark.browser
def test_stable_capture_waits_for_three_samples():
    with HtmlBrowser() as browser:
        browser.load('<div data-fd-id="a">Hello</div>',[200,100])
        image,stats=browser.stable_capture({'a'})
        assert image and stats['consecutive_stable']==3 and stats['resources_ready']


@pytest.mark.browser
def test_unstable_capture_is_not_accepted(monkeypatch):
    with HtmlBrowser() as browser:
        browser.load('<div data-fd-id="a">Hello</div>',[200,100])
        shots=iter([png('red'),png('blue')]*100)
        monkeypatch.setattr(browser.page,'screenshot',lambda **kw:next(shots))
        with pytest.raises(ValueError,match='pixels/target boxes keep changing'):
            browser.stable_capture({'a'},timeout_ms=200,interval_ms=30)

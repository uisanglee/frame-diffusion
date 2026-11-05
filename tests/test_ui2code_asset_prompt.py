from framediff.ui2code_asset_prompt import image_asset_instruction, WEBUI_CONTRACT


def test_design2code_rules_preserve_application_rendering():
    prompt = image_asset_instruction('<img src="rick.jpg">')
    assert 'Design2Code-Hard' in prompt and 'rick.jpg' in prompt
    assert 'rick1.jpg' not in prompt
    assert 'React, JSX, Babel, and Tailwind are allowed' in prompt
    assert 'Do not generate new images' in prompt
    assert 'base64/data-URI image payloads' in prompt
    assert 'Do not copy library contents or asset binaries' in prompt


def test_webui_contract_and_existing_names_select_five_colors():
    for html, contract in [('', WEBUI_CONTRACT), (f'<html data-tuide-image-contract="{WEBUI_CONTRACT}">', None),
                           ('<img src="rick2.jpg">', None)]:
        prompt = image_asset_instruction(html, contract)
        for pair in ['red=rick1.jpg', 'green=rick2.jpg', 'yellow=rick3.jpg', 'blue=rick4.jpg', 'black=rick5.jpg']:
            assert pair in prompt
        assert 'Design2Code' not in prompt


def test_legacy_webui_does_not_invent_design2code_assets():
    prompt = image_asset_instruction(image_contract='webui-no-benchmark-placeholder-v1')
    assert 'rick.jpg' not in prompt
    assert 'do not invent new image filenames' in prompt


def test_polishing_keeps_binary_assets_out_of_model_input(tmp_path, monkeypatch):
    import base64
    from baselines.ui2coden.runner import polishing_prompt
    from framediff.local_assets import ORIGIN, read_asset
    monkeypatch.setenv('TUIDE_LOCAL_ASSET_CACHE', str(tmp_path))
    resource = b'a' * 100_000
    encoded = base64.b64encode(resource).decode()
    html = '<style>@font-face{font-family:existing;src:url(data:font/woff2;base64,' + encoded + ')}</style><img src="rick2.jpg">'
    prompt = polishing_prompt(html)
    assert encoded not in prompt
    assert len(prompt) < 5000
    assert 'red=rick1.jpg' in prompt
    current = prompt.split('Current HTML:\n', 1)[1]
    url = current.split('url(', 1)[1].split(')', 1)[0]
    assert url.startswith(ORIGIN + '/')
    assert read_asset(url)['body'] == resource

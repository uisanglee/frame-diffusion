import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


spec = importlib.util.spec_from_file_location('target_provenance',
    Path(__file__).parents[1] / 'scripts/audit_target_provenance.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def dump(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + ('\n' if jsonl else ''))


def picture(path, color='white', size=(10, 10)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path)
    return path


@pytest.fixture
def chain(tmp_path):
    original = tmp_path / 'raw/page/view-html.html'
    color = tmp_path / 'data/master/source/pages/color/target.html'
    tagged = tmp_path / 'data/master/rendered/pages/tagged/target.html'
    normalized = tmp_path / 'data/master/labels/normalized-pages/test/normalized/normalized.html'
    for path in (original, color, tagged, normalized):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<html><head><link rel="stylesheet" href="https://example.test/site.css">'
                        '<style>@import "font.css";p{color:red}</style></head>'
                        '<body><p style="margin:0">Example</p></body></html>')
    picture(original.with_name('view-screenshot.png'), 'red')
    for path in (color.with_suffix('.png'), tagged.with_suffix('.png'), normalized.parent / 'after.png'):
        picture(path)
    evaluation = picture(tmp_path / 'cache/A_rendering.png')
    source = {'id': 'page/view', 'page_id': 'page', 'view': 'view',
              'html': str(color), 'original_html': str(original), 'screenshot': str(color.with_suffix('.png')),
              'source_sha': audit.sha256(color), 'raw_input_sha': audit.sha256(original)}
    dump(tmp_path / 'data/master/source/manifest.jsonl', source, jsonl=True)
    dump(tagged.parent / 'records.json', {'page': {'screenshot': str(tagged.with_suffix('.png')),
                                                 'source_sha': audit.sha256(color)}})
    dump(normalized.parent / 'record.json', {'page': {'original_html': str(tagged)},
         'source_sha': audit.sha256(tagged), 'assets': {str(normalized): audit.sha256(normalized)}})
    dump(normalized.parent / 'normalization.json', {'source_sha': audit.sha256(tagged), 'sha': audit.sha256(normalized)})
    dump(normalized.parent / 'report.json', {'accepted': True, 'pixel_mae': 0.0})
    row = {'id': 'page/view', 'group': 'webui-page:page', 'viewport': [10, 10],
           'html': str(normalized), 'screenshot': str(evaluation), 'provided_screenshot': str(normalized.parent / 'after.png')}
    manifest = tmp_path / 'run/webui/model-a/seeds/42/prepare/prepared.jsonl'
    dump(manifest, row, jsonl=True)
    return tmp_path, row, manifest, normalized


def test_resolves_saved_chain_and_preserves_missing_causal_evidence(chain):
    root, row, manifest, normalized = chain
    report = audit.audit(root / 'run', root / 'data', root / 'out', 10)
    target = report['targets'][0]
    assert all(info['available'] for info in target['images'].values())
    assert target['images']['original']['path'].endswith('view-screenshot.png')
    assert target['comparisons']['original_to_evaluation']['rgb_mae'] > 0
    assert target['comparisons']['tagged_to_normalized']['exact_rgb_pixels']
    assert target['normalization']['hash_checks']['normalized_html']['matches'] is True
    assert all(check['matches'] is True for check in target['source_hash_checks'].values())
    assert target['tagged_record_path'].endswith('/records.json')
    assert target['html']['original']['stylesheet_links'][0]['href'] == 'https://example.test/site.css'
    assert target['html']['original']['css_imports']
    assert target['html']['original']['inline_style_count'] == 1
    assert any('No CSS ablation' in text for text in target['findings'])
    assert (root / 'out/index.html').is_file()
    assert (root / 'out/images/001-original.png').is_file()


def test_deduplicates_methods_but_retains_changed_target_variant(chain):
    root, row, manifest, _ = chain
    dump(root / 'run/webui/model-b/seeds/42/prepare/prepared.jsonl', row, jsonl=True)
    other = dict(row, screenshot=str(picture(root / 'cache/changed.png', 'blue')))
    dump(root / 'run/webui/model-c/seeds/42/prepare/prepared.jsonl', other, jsonl=True)
    report = audit.audit(root / 'run', root / 'data', root / 'out', 1)
    assert report['selected_page_views'] == 1
    assert len(report['targets']) == 2
    assert len(report['targets'][0]['origins']) == 2
    assert report['multiple_target_variants'] == [[0, 1]]


def test_missing_files_and_hash_mismatch_are_visible(chain):
    root, _, _, normalized = chain
    (root / 'raw/page/view-screenshot.png').unlink()
    normalized.write_text('<html><body>Changed after QA</body></html>')
    target = audit.audit(root / 'run', root / 'data', root / 'out', 1)['targets'][0]
    assert not target['images']['original']['available']
    assert target['comparisons']['original_to_evaluation']['reason'] == 'missing_image'
    assert target['normalization']['hash_checks']['normalized_html']['matches'] is False
    assert any('Missing original image' in issue for issue in target['issues'])
    assert any('hash disagrees' in finding for finding in target['findings'])


def test_source_join_rejects_stale_color_hash(chain):
    root, _, _, _ = chain
    color = root / 'data/master/source/pages/color/target.html'
    color.write_text('<html><body>A newer preparation</body></html>')
    target = audit.audit(root / 'run', root / 'data', root / 'out', 1)['targets'][0]
    assert target['source_hash_checks']['color_to_tagged_source']['matches'] is False
    assert any('different preprocessing versions' in text for text in target['findings'])


def test_image_comparison_never_resizes(tmp_path):
    a = picture(tmp_path / 'a.png')
    b = picture(tmp_path / 'b.png', size=(20, 10))
    comparison = audit.compare_images(a, b)
    assert comparison['status'] == 'unavailable'
    assert comparison['reason'] == 'dimension_mismatch'
    assert 'rgb_mae' not in comparison
    assert audit.compare_images(a, None)['reason'] == 'missing_image'
    c = picture(tmp_path / 'c.png', 'black')
    comparison = audit.compare_images(a, c)
    assert comparison['rgb_mae'] == 1.0
    assert comparison['changed_pixel_fraction'] == 1.0


def test_embedded_css_is_not_labeled_external(tmp_path):
    path = tmp_path / 'data-css.html'
    path.write_text('<link rel="stylesheet" href="data:text/css,p%7Bcolor:red%7D">')
    inventory = audit.css_inventory(path)
    assert inventory['stylesheet_links'][0]['resource_kind'] == 'embedded_data_uri'

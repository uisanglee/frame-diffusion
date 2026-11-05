#!/usr/bin/env python3
"""Read saved polishing outputs; diagnose upstream preprocessing on copies only."""
import argparse
import contextlib
import difflib
import hashlib
import html as html_lib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bs4 import BeautifulSoup, NavigableString, Comment
from framediff.html_bridge import HtmlBrowser, embed_placeholder


def load_preprocess(repo):
    """Execute the installed upstream functions unchanged, without loading CLIP.

    Extract top-level definitions separately: some upstream versions have an
    unrelated indentation error in visual_eval_v3_multi (handled by our scorer).
    """
    metrics = repo/'Design2Code/metrics/visual_score.py'
    dedup = repo/'Design2Code/data_utils/dedup_post_gen.py'
    source = metrics.read_text()
    spec = importlib.util.spec_from_file_location('_diagnostic_dedup', dedup)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ns = dict(re=re, BeautifulSoup=BeautifulSoup, NavigableString=NavigableString,
              Comment=Comment, check_repetitive_content=module.check_repetitive_content)
    for name in ('truncate_repeated_html_elements', 'make_html', 'pre_process'):
        match = re.search(r'^def '+name+r'\(.*?(?=^def |\Z)', source, re.M | re.S)
        if not match:
            raise ValueError(f'Unsupported upstream version: missing {name}')
        exec(compile(match[0], str(metrics), 'exec'), ns)
    digest = hashlib.sha256(source.encode()+dedup.read_bytes()).hexdigest()
    return ns['pre_process'], digest


def stats(text):
    soup = BeautifulSoup(text, 'html.parser')
    css = '\n'.join(t.get_text() for t in soup.find_all('style'))
    for t in soup(['style', 'script', 'head']):
        t.decompose()
    return dict(html_chars=len(text), css_chars=len(css),
                elements=len(soup.find_all(True)), text=soup.get_text(' ', strip=True)), css


def inspect(browser, content, viewport, path):
    browser.reset_context()
    browser.load(content, viewport)
    browser.page.screenshot(path=str(path), animations='disabled')
    return browser.page.evaluate('''()=>({visible_text:document.body?.innerText||'',
      visible_elements:[...document.querySelectorAll('body *')].filter(e=>{
        const r=e.getBoundingClientRect(),s=getComputedStyle(e);
        return !['SCRIPT','STYLE'].includes(e.tagName)&&r.width>0&&r.height>0&&r.bottom>0&&r.right>0&&r.top<innerHeight&&r.left<innerWidth&&s.visibility!=='hidden'&&s.display!=='none';
      }).length})''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--official-repo', type=Path, default=ROOT/'external/design2code-official')
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.out or run/'preprocess-diagnosis').resolve()
    out.mkdir(parents=True, exist_ok=True)
    preprocess, upstream_sha = load_preprocess(args.official_repo.resolve())
    implementation = hashlib.sha256(Path(__file__).read_bytes()+b''.join(
        (ROOT/'framediff'/p).read_bytes() for p in ('browser.py','html_bridge.py','local_assets.py'))).hexdigest()
    manifests = sorted(run.glob('*/*/seeds/*/ui2coden/results.jsonl'))
    if not manifests:
        raise SystemExit('No */*/seeds/*/ui2coden/results.jsonl found under '+str(run))
    summaries = []
    with HtmlBrowser() as browser:
        for manifest in manifests:
            for line in manifest.read_text().splitlines():
                if not line.strip(): continue
                record = json.loads(line)
                for method, data in record['methods'].items():
                    if not method.startswith('ui2coden-round-'): continue
                    if record['methods'].get('initial', {}).get('failed'): continue
                    identity = dict(manifest=str(manifest), id=record['id'], method=method,
                                    failed=data.get('failed'), error=data.get('error'), source=data['html'])
                    print(json.dumps(dict(identity, status='started')), flush=True)
                    try:
                        original = Path(data['html']).read_text()
                        placeholder = Path(record['html']).parent/'rick.jpg' if record.get('html') else None
                        content = embed_placeholder(original, placeholder if placeholder and placeholder.is_file() else None)
                        viewport = record['viewport']
                        key = hashlib.sha256((content+json.dumps(viewport)+upstream_sha+implementation).encode()).hexdigest()
                        folder = out/key[:24]
                        folder.mkdir(exist_ok=True)
                        result_path = folder/'report.json'
                        if result_path.exists() and all((folder/p).exists() for p in ('before.png','after.png','before.html','after.html','css.diff','html.diff')):
                            report = json.loads(result_path.read_text())
                            reused = True
                        else:
                            (folder/'source.html').write_text(original)
                            (folder/'before.html').write_text(content)
                            after = folder/'after.html'
                            after.write_text(content)
                            before_dom = inspect(browser, content, viewport, folder/'before.png')
                            log = io.StringIO()
                            start = time.perf_counter()
                            print('Official preprocessing: '+str(after), flush=True)
                            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                                preprocess(str(after))
                            seconds = time.perf_counter()-start
                            (folder/'preprocess.log').write_text(log.getvalue())
                            processed = after.read_text()
                            after_dom = inspect(browser, processed, viewport, folder/'after.png')
                            before_stats, before_css = stats(content)
                            after_stats, after_css = stats(processed)
                            for name, a, b in [('html', content, processed), ('css', before_css, after_css),
                                               ('text', before_stats['text'], after_stats['text'])]:
                                # No expensive fuzzy diff on huge single-line CSS.
                                (folder/(name+'.diff')).write_text(''.join(difflib.unified_diff(
                                    a.splitlines(True), b.splitlines(True), fromfile='before', tofile='after')))
                            from PIL import Image, ImageChops, ImageStat
                            with Image.open(folder/'before.png') as a, Image.open(folder/'after.png') as b:
                                diff = ImageChops.difference(a.convert('RGB'), b.convert('RGB'))
                                mae = sum(ImageStat.Stat(diff).mean)/(3*255)
                                diff.save(folder/'difference.png')
                            # Literal-prefix truncation is checked before BeautifulSoup rewrites it.
                            retained = None
                            warnings = re.findall(r'start at (\d+)', log.getvalue())
                            if warnings: retained = int(warnings[-1])
                            report = dict(key=key, upstream_sha=upstream_sha, viewport=viewport,
                                preprocess_seconds=seconds, before=before_stats, after=after_stats,
                                before_dom=before_dom, after_dom=after_dom, pixel_mae=mae,
                                suspected_content_loss=after_stats['elements']<before_stats['elements'] or len(after_dom['visible_text'])<len(before_dom['visible_text']),
                                repetition_cut_position=retained,
                                cut_inside_style=(content.lower().rfind('<style',0,retained)>content.lower().rfind('</style',0,retained)) if retained is not None else None)
                            result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
                            reused = False
                        row = dict(identity, folder=folder.name, reused=reused, **report)
                    except Exception as exc:
                        row = dict(identity, diagnostic_error=str(exc))
                    summaries.append(row)
                    (out/'summary.json').write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
                    print(json.dumps({k:row.get(k) for k in ('id','method','failed','reused','preprocess_seconds','pixel_mae','cut_inside_style','suspected_content_loss','diagnostic_error')}, ensure_ascii=False), flush=True)
    cards = []
    for row in summaries:
        title = html_lib.escape(f"{Path(row['manifest']).relative_to(run)} | {row['id']} | {row['method']} | failed={row['failed']}")
        if 'diagnostic_error' in row:
            cards.append('<h3>'+title+'</h3><pre>'+html_lib.escape(row['diagnostic_error'])+'</pre>'); continue
        f = row['folder']
        details = {k:row.get(k) for k in ('preprocess_seconds','pixel_mae','cut_inside_style','suspected_content_loss')}
        cards.append(f'<h3>{title}</h3><pre>{html_lib.escape(json.dumps(details))}</pre><p><a href="{f}/report.json">Report</a> | <a href="{f}/html.diff">HTML diff</a> | <a href="{f}/css.diff">CSS diff</a> | <a href="{f}/text.diff">Text diff</a> | <a href="{f}/preprocess.log">Warnings</a></p><div class="pair"><figure>Before<img src="{f}/before.png"></figure><figure>After<img src="{f}/after.png"></figure></div>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Preprocessing comparison</title><style>body{font-family:sans-serif}.pair{display:flex}figure{width:48%;margin:1%}img{width:100%}pre{white-space:pre-wrap}</style><h1>Saved UI2Code output / official preprocessing</h1>'+''.join(cards))
    print('Comparison: '+str(out/'index.html'), flush=True)


if __name__ == '__main__': main()

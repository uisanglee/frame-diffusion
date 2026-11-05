"""Transfer static policy edits back to a live app, verifying the rendered result."""
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .html_bridge import HtmlBrowser


def restore_application(initial, repaired_html, viewport, out, browser_host=None):
    from .app_browser import freeze_application
    from .app_edits import apply_application_edits
    started = time.perf_counter()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    source = Path(initial['application_html']).read_text()
    before = Path(initial['html']).read_text()
    application, edits = apply_application_edits(source, before, repaired_html)
    (out/'edits.json').write_text(json.dumps(edits, indent=2))
    capture = freeze_application(application, viewport, out/'application',browser_host=browser_host)
    # Verify the executed patch also reproduces the intended policy result, not
    # just that freezing the patched application itself is lossless.
    with (nullcontext(browser_host) if browser_host is not None else HtmlBrowser()) as browser:
        expected = browser.snapshot(repaired_html, viewport, out/'expected.png',2048)
        actual = browser.snapshot(Path(capture['static_html']).read_text(),viewport,out/'actual.png',2048)
        a = np.asarray(Image.open(out/'expected.png').convert('RGB'),dtype=float)
        b = np.asarray(Image.open(out/'actual.png').convert('RGB'),dtype=float)
        mae = float(np.abs(a-b).mean()/255.)
        # Source-preserving patches must keep element traversal and geometry.
        x,y=expected['nodes'],actual['nodes']
        box_error = (max((abs(p-q) for u,v in zip(x,y) for p,q in zip(u['box'],v['box'])),default=0.)
                     if len(x)==len(y) and all(u['role']==v['role'] for u,v in zip(x,y)) else float('inf'))
    report={'pixel_mae':mae,'max_box_error_px':box_error if np.isfinite(box_error) else None,
            'accepted':mae<=.001 and box_error<=1.,'edits':len(edits),
            'seconds':time.perf_counter()-started}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    if not report['accepted']:
        raise ValueError(f'Application patch does not preserve policy result: {report}')
    return {'html':Path(capture['static_html']).read_text(),
            'application_html':capture['source_html'],'application_capture':capture,
            'application_patch_verification':report,
            'application_patch_seconds':report['seconds'],
            'application_patch_browser_executions':capture.get('browser_executions',0)+2,
            'application_patch_browser_screenshots':capture.get('browser_screenshots',0)+2}

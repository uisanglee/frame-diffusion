"""Bring-your-own screenshot/initial-HTML manifest for model-agnostic repair."""
from pathlib import Path

from PIL import Image

from .html_bridge import html_answer
from .ir import write_jsonl
from .web_experiment import digest, guard_run


def add_parser(sub):
    p=sub.add_parser('web-external-manifest',help='Build one evaluation item from an externally generated initial HTML')
    p.add_argument('--screenshot',required=True);p.add_argument('--initial-html',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reference-html');p.add_argument('--id',default='external-page')
    p.add_argument('--source-model',default='external-vlm');p.add_argument('--resume',action='store_true')


def build(args):
    screenshot=Path(args.screenshot).resolve();initial=Path(args.initial_html).resolve()
    reference=Path(args.reference_html).resolve() if args.reference_html else None
    for path in (screenshot,initial,*([reference] if reference else [])):
        if not path.is_file():raise FileNotFoundError(path)
    html_answer(initial.read_text())
    if reference:html_answer(reference.read_text())
    with Image.open(screenshot) as image:
        viewport=list(image.size)
        if min(viewport)<1:raise ValueError('Invalid screenshot dimensions')
    out=Path(args.out).resolve()
    config={'kind':'external-html-manifest-v1','id':args.id,'source_model':args.source_model,
            'screenshot':digest(screenshot),'initial_html':digest(initial),
            'reference_html':digest(reference) if reference else None}
    guard_run(out,config,args.resume)
    row={'id':args.id,'group':'external/'+digest(initial)[:16],'dataset':'external-html',
         'screenshot':str(screenshot),'initial_html':str(initial),'html':str(reference) if reference else None,
         'external_source_model':args.source_model,'external_initial_html':True,'viewport':viewport}
    write_jsonl(out/'manifest.jsonl',[row])
    print(out/'manifest.jsonl')
    return row

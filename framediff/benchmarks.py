"""Real-page benchmark preparation and screenshot-to-LayoutIR batch inference."""
from __future__ import annotations
import copy
import json
import time
import traceback
from pathlib import Path
from PIL import Image
from .adapters import fit_observation
from .data import make_record
from .ir import read_jsonl,validate,write_json,write_jsonl


def discover_design2code(root):
    """Return screenshot/HTML pairs from the official xx.png + xx.html layout."""
    root=Path(root);pairs=[]
    for image in sorted(root.rglob('*.png')):
        html=image.with_suffix('.html')
        if html.exists():
            rel=image.relative_to(root).with_suffix('')
            pairs.append((str(rel).replace('/','__'),image,html))
    return pairs


def combine_data(args):
    records=[];seen=set();sources=[]
    for path in args.input:
        chunk=list(read_jsonl(path));sources.append({'path':str(path),'records':len(chunk)})
        for record in chunk:
            key=(record.get('source','unknown'),record['id'])
            if key in seen:raise ValueError(f'Duplicate source/id pair: {key}')
            if args.split and record.get('split')!=args.split:
                raise ValueError(f'{record["id"]} has split={record.get("split")}, expected {args.split}')
            seen.add(key);records.append(record)
    if not records:raise ValueError('No records to combine')
    write_jsonl(args.out,records)
    write_json(Path(args.out).with_suffix(Path(args.out).suffix+'.manifest.json'),{'sources':sources,'records':len(records)})
    print(f'Combined {len(records)} records into {args.out}')


def prepare_design2code(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    pairs=discover_design2code(args.root)
    if args.limit:pairs=pairs[:args.limit]
    if not pairs:raise ValueError('No matching .png/.html pairs found')
    manifest=[];oracle=[];report=[]
    from .browser import Browser
    with Browser() as browser:
        for sample_id,image_path,html_path in pairs:
            try:
                with Image.open(image_path) as image:viewport=list(image.size)
                if min(viewport)<=0:raise ValueError('Invalid screenshot size')
                observation=browser.extract_html(html_path.read_text(errors='replace'),viewport,max_nodes=args.max_nodes,overflow='salience')
                if not observation['nodes']:raise ValueError('No visible DOM nodes')
                target={n['id']:n['box'] for n in observation['nodes']}
                root_ids=[observation['nodes'][0]['id']]
                item={'id':sample_id,'group':sample_id,'split':'test','dataset':args.dataset,
                      'screenshot':str(image_path.resolve()),'html':str(html_path.resolve()),
                      'reference_observations':[{'viewport':viewport,'target':target}],
                      'reference_root_ids':root_ids}
                manifest.append(item)
                fitted,original,fidelity=fit_observation(observation)
                accepted=1-fidelity['metrics']['box_iou']<=args.max_fit_error
                if accepted:
                    record=make_record(fitted,sample_id,sample_id,args.seed+len(manifest),
                                       viewports=[viewport],mode='geometry')
                    actual={'viewport':viewport,'target':original}
                    record.update(source=f'{args.dataset}_oracle_surrogate',split='test',
                                  observations=[actual],evaluation_observations=[actual],
                                  evaluation_matching='identity',fit=fidelity,
                                  reference_root_ids=root_ids)
                    oracle.append(record)
                report.append({'id':sample_id,'accepted':True,'oracle_accepted':accepted,
                               'visible_nodes':len(observation['nodes']),'total_visible_nodes':observation['total_visible_nodes'],
                               'selection':observation['selection'],'viewport':viewport,'fit':fidelity})
                print(f'{sample_id}: nodes={len(observation["nodes"])} fit_iou={fidelity["metrics"]["box_iou"]:.4f}',flush=True)
            except Exception as error:
                report.append({'id':sample_id,'accepted':False,'reason':str(error)})
                print(f'{sample_id}: rejected: {error}',flush=True)
                if args.fail_fast:raise
    write_jsonl(out/'manifest.jsonl',manifest)
    write_jsonl(out/'oracle-test.jsonl',oracle)
    write_json(out/'prepare-report.json',{'dataset':args.dataset,'discovered':len(pairs),
               'prepared':len(manifest),'oracle_prepared':len(oracle),'records':report,
               'note':'Reference DOM boxes are FrameDiff geometry ground truth; this is not the official Design2Code visual metric.'})
    print(f'Prepared {len(manifest)}/{len(pairs)} pages; {len(oracle)} oracle records')


def _vlm_call(args,task,image,current,runtime):
    from . import vlm
    call=copy.copy(args);call.task=task;call.image=[str(image)];call.current=str(current) if current else None;call.frames=None
    prompt=vlm.prompt_for(call);images=vlm.images_for(call)
    if task=='extract-frames':
        with Image.open(image) as source:prompt+=f' Original screenshot dimensions are {list(source.size)}; scale coordinates back to these dimensions.'
    started=time.perf_counter();answer,meta=vlm.generate(call,prompt,images,runtime)
    meta={**meta,'wall_seconds':time.perf_counter()-started}
    result=vlm.json_answer(answer)
    if task=='generate-ir':
        result=validate(result)
        if len(result['nodes'])>args.max_nodes:raise ValueError(f'Generated IR exceeds max_nodes={args.max_nodes}')
    else:
        with Image.open(image) as source:expected_viewport=list(source.size)
        expected={n['id'] for n in validate(json.loads(Path(current).read_text()))['nodes']}
        if result.get('viewport')!=expected_viewport or set(result.get('target',{}))!=expected:
            raise ValueError('Invalid VLM viewport/identity coverage')
        for box in result['target'].values():
            if len(box)!=4 or any(not isinstance(v,(float,int)) for v in box) or min(box[2:])<=0:
                raise ValueError('Invalid predicted bounding box')
    return result,meta,prompt,answer


def run_design2code_vlm(args):
    """Generate initial IR and identity-aligned target frames, once per screenshot."""
    items=list(read_jsonl(args.manifest))
    if args.limit:items=items[:args.limit]
    if not items:raise ValueError('Empty benchmark manifest')
    out=Path(args.out);(out/'initial').mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    from . import vlm
    runtime=vlm.load_qwen_runtime(args) if args.backend=='qwen' else None
    records=[];report=[]
    for item in items:
        sample_id=item['id'];initial_path=out/'initial'/f'{sample_id}.json';frames_path=out/'frames'/f'{sample_id}.json'
        try:
            if args.resume and initial_path.exists():
                initial=json.loads(initial_path.read_text());initial_meta=json.loads(initial_path.with_suffix('.meta.json').read_text())
            else:
                initial,initial_meta,prompt,raw=_vlm_call(args,'generate-ir',item['screenshot'],None,runtime)
                write_json(initial_path,initial);write_json(initial_path.with_suffix('.meta.json'),{**initial_meta,'prompt':prompt})
                initial_path.with_suffix('.raw.txt').write_text(raw)
            if args.resume and frames_path.exists():
                frames=json.loads(frames_path.read_text());frames_meta=json.loads(frames_path.with_suffix('.meta.json').read_text())
            else:
                frames,frames_meta,prompt,raw=_vlm_call(args,'extract-frames',item['screenshot'],initial_path,runtime)
                write_json(frames_path,frames);write_json(frames_path.with_suffix('.meta.json'),{**frames_meta,'prompt':prompt})
                frames_path.with_suffix('.raw.txt').write_text(raw)
            record={'id':sample_id,'group':item.get('group',sample_id),'split':'test','source':item.get('dataset','design2code')+'_vlm',
                    'target_kind':'predicted_frames','current':validate(initial),'observations':[frames],
                    'reference_observations':item['reference_observations'],'reference_root_ids':item.get('reference_root_ids',[]),
                    'evaluation_matching':'hungarian_geometry','screenshot':item['screenshot'],'html':item['html'],
                    'upstream_seconds':initial_meta.get('wall_seconds',0)+frames_meta.get('wall_seconds',0)}
            records.append(record);report.append({'id':sample_id,'accepted':True,'nodes':len(initial['nodes'])})
            write_jsonl(out/'test.jsonl',records)
            print(f'{sample_id}: generated {len(initial["nodes"])} nodes',flush=True)
        except Exception as error:
            report.append({'id':sample_id,'accepted':False,'reason':str(error),'traceback':traceback.format_exc()})
            write_json(out/'vlm-report.json',{'processed':len(report),'accepted':len(records),'records':report})
            print(f'{sample_id}: failed: {error}',flush=True)
            if args.fail_fast:raise
    write_jsonl(out/'test.jsonl',records)
    write_json(out/'vlm-report.json',{'processed':len(report),'accepted':len(records),'records':report,
               'note':'Evaluation uses identity-free Hungarian geometry matching against browser-extracted reference DOM boxes.'})
    print(f'Generated benchmark records for {len(records)}/{len(items)} pages')

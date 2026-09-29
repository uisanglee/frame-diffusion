import argparse
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description='FrameDiff executable layout repair toolkit')
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--out',required=True);g.add_argument('--count',type=int,default=1000);g.add_argument('--seed',type=int,default=42)
    t=sub.add_parser('train')
    t.add_argument('--train',required=True);t.add_argument('--val',required=True);t.add_argument('--out',required=True)
    for name,default in [('steps',10000),('batch-size',16),('accumulation',1),('hidden',256),('layers',4),('heads',8),('max-nodes',128),('max-noise',5),('seed',42),('eval-every',500),('log-every',50),('val-samples',128),('cpu-threads',4)]:t.add_argument('--'+name,type=int,default=default)
    t.add_argument('--lr',type=float,default=3e-4);t.add_argument('--dropout',type=float,default=.1)
    t.add_argument('--device',default='auto');t.add_argument('--bf16',action='store_true');t.add_argument('--no-tree-bias',action='store_true')
    t.add_argument('--fixed-pairs',action='store_true');t.add_argument('--resume');t.add_argument('--objective',choices=['edits','boxes'],default='edits')
    e=sub.add_parser('evaluate');e.add_argument('--data',required=True);e.add_argument('--out',required=True);e.add_argument('--checkpoint')
    e.add_argument('--methods',default='none,coordinate,copy-boxes');e.add_argument('--device',default='auto')
    for name,default in [('steps',10),('beam',2),('topk',8),('budget',500),('limit',0),('seed',42),('cpu-threads',4)]:e.add_argument('--'+name,type=int,default=default)
    e.add_argument('--browser-verify',action='store_true');e.add_argument('--screenshots',action='store_true');e.add_argument('--allow-train-overlap',action='store_true')
    r=sub.add_parser('render');r.add_argument('--ir',required=True);r.add_argument('--out',required=True);r.add_argument('--width',type=int,default=1024);r.add_argument('--height',type=int,default=768);r.add_argument('--screenshot',action='store_true')
    x=sub.add_parser('extract-html');x.add_argument('--html',required=True);x.add_argument('--out',required=True);x.add_argument('--width',type=int,default=1024);x.add_argument('--height',type=int,default=768);x.add_argument('--max-nodes',type=int,default=128)
    w=sub.add_parser('import-webui');w.add_argument('--root',required=True);w.add_argument('--out',required=True);w.add_argument('--max-nodes',type=int,default=128);w.add_argument('--limit',type=int,default=0);w.add_argument('--split',choices=['train','val','test']);w.add_argument('--seed',type=int,default=42)
    f=sub.add_parser('fit-frames');f.add_argument('--input',required=True);f.add_argument('--out',required=True);f.add_argument('--max-error',type=float,default=.15);f.add_argument('--seed',type=int,default=42)
    v=sub.add_parser('vlm');v.add_argument('--backend',choices=['qwen','openai-compatible'],default='qwen');v.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');v.add_argument('--revision',default='main');v.add_argument('--endpoint',default='http://localhost:8000/v1/chat/completions');v.add_argument('--api-key-env',default='VLM_API_KEY')
    v.add_argument('--task',choices=['generate-ir','revise-ir','generate-html','revise-html','extract-frames'],required=True);v.add_argument('--image',action='append',default=[]);v.add_argument('--current');v.add_argument('--frames');v.add_argument('--out',required=True)
    v.add_argument('--four-bit',action='store_true');v.add_argument('--max-new-tokens',type=int,default=4096);v.add_argument('--max-pixels',type=int,default=1048576);v.add_argument('--seed',type=int,default=42)
    pair=sub.add_parser('make-pair');pair.add_argument('--current',required=True);pair.add_argument('--target',required=True);pair.add_argument('--out',required=True);pair.add_argument('--id',required=True);pair.add_argument('--group',required=True);pair.add_argument('--clean');pair.add_argument('--ground-truth');pair.add_argument('--target-kind',choices=['oracle_frames','predicted_frames'],required=True)
    a=sub.add_parser('suite');a.add_argument('--config',required=True)
    export=sub.add_parser('export-example');export.add_argument('--data',required=True);export.add_argument('--out',required=True);export.add_argument('--index',type=int,default=0)
    infer=sub.add_parser('repair');infer.add_argument('--ir',required=True);infer.add_argument('--target',required=True);infer.add_argument('--checkpoint',required=True);infer.add_argument('--out',required=True);infer.add_argument('--device',default='auto')
    for name,default in [('steps',10),('beam',2),('topk',8),('budget',160)]:infer.add_argument('--'+name,type=int,default=default)
    args=p.parse_args(argv)
    for key in ('steps','batch_size','accumulation','hidden','layers','heads','max_nodes','max_noise','eval_every','log_every','val_samples','beam','topk','budget','cpu_threads'):
        if hasattr(args,key) and getattr(args,key)<1:p.error(key+' must be positive')
    if args.command=='generate':
        from .data import generate
        print(generate(args.out,args.count,args.seed))
    elif args.command=='train':
        from .train import train
        train(args)
    elif args.command=='evaluate':
        from .evaluate import evaluate
        evaluate(args)
    elif args.command=='render':
        from .ir import read_json,compile_html,write_json,execute
        tree=read_json(args.ir);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);v=(args.width,args.height)
        (out/'layout.html').write_text(compile_html(tree,v));write_json(out/'frames.json',execute(tree,v))
        if args.screenshot:
            from .browser import Browser
            with Browser() as b:write_json(out/'browser-frames.json',b.render(tree,v,out/'layout.png'))
    elif args.command=='extract-html':
        from .browser import Browser
        from .ir import write_json
        with Browser() as b:write_json(args.out,b.extract_html(Path(args.html).read_text(),(args.width,args.height),max_nodes=args.max_nodes))
    elif args.command in ('import-webui','fit-frames'):
        from .adapters import import_webui,fit_frames
        (import_webui if args.command=='import-webui' else fit_frames)(args)
    elif args.command=='vlm':
        from .vlm import run
        run(args)
    elif args.command=='make-pair':
        from .ir import read_json,write_jsonl,validate
        from .data import group_split,differences
        current=validate(read_json(args.current));target=read_json(args.target)
        observations=target if isinstance(target,list) else [target]
        for obs in observations:
            if 'viewport' not in obs or 'target' not in obs:raise ValueError('Expected viewport + target boxes in target file')
            if set(obs['target'])!={n['id'] for n in current['nodes']}:raise ValueError('Target/current IDs must match; explicitly match predictions first')
        record={'id':args.id,'group':args.group,'split':group_split(args.group),'current':current,'observations':observations,'target_kind':args.target_kind,'source':'external'}
        if args.ground_truth:
            truth=read_json(args.ground_truth)
            record['evaluation_observations']=truth if isinstance(truth,list) else [truth]
        if args.target_kind=='predicted_frames' and not args.ground_truth:
            p.error('predicted_frames requires --ground-truth for independent evaluation')
        if args.clean:record['clean']=validate(read_json(args.clean));differences(current,record['clean'])
        write_jsonl(args.out,[record])
    elif args.command=='suite':
        from .suite import run_suite
        run_suite(args.config)
    elif args.command=='export-example':
        from .ir import read_jsonl,write_json,compile_html
        record=list(read_jsonl(args.data))[args.index];out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        write_json(out/'current.json',record['current']);write_json(out/'target.json',record['observations'])
        write_json(out/'record.json',record)
        if 'clean' in record:
            write_json(out/'clean.json',record['clean'])
            (out/'reference.html').write_text(compile_html(record['clean'],record['observations'][0]['viewport']))
        (out/'current.html').write_text(compile_html(record['current'],record['observations'][0]['viewport']))
    elif args.command=='repair':
        from .ir import read_json,write_json,compile_html,validate
        from .model import load_model
        from .search import repair
        from .train import select_device
        model,ck=load_model(args.checkpoint,select_device(args.device))
        if ck['objective']!='edits':raise ValueError('Repair requires an edit checkpoint')
        target=read_json(args.target);obs=target if isinstance(target,list) else [target]
        result,trace=repair(validate(read_json(args.ir)),obs,model=model,steps=args.steps,beam=args.beam,proposals_per_state=args.topk,budget=args.budget)
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        write_json(out/'repaired.json',result);write_json(out/'trace.json',trace)
        for i,o in enumerate(obs):(out/f'repaired-{i}.html').write_text(compile_html(result,o['viewport']))

if __name__=='__main__':main()

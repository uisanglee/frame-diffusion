"""Explicit CLI for image-based experiments; legacy plans are never consulted."""


def add_parsers(sub):
    p=sub.add_parser('visual-subset-numeric',help='Filter cached trajectories without rendering or changing source assets')
    p.add_argument('--rendered',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('visual-tree-prepare',
        help='Parse cached current/target HTML and construct exact CSS declaration replacement paths; no rerendering')
    p.add_argument('--rendered',required=True);p.add_argument('--out',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--stylesheets',action='store_true',help='Address inline declarations and embedded stylesheet rules')
    p.add_argument('--max-css-owners',type=int,default=512,help='Reject pages exceeding owner capacity in stylesheet mode')
    p=sub.add_parser('visual-tree-freeze',help='Render fixed inline+stylesheet val/test corruption, keeping target assets')
    p.add_argument('--rendered',required=True);p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--seed',type=int,default=90210)
    p.add_argument('--max-noise',type=int,default=4);p.add_argument('--samples-per-page',type=int,default=1)
    p=sub.add_parser('visual-import-webui',help='Build domain-disjoint WebUI manifests and optional native AX detector labels')
    p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--view',default='default_1280-720')
    p.add_argument('--train-count',type=int,default=600);p.add_argument('--val-count',type=int,default=200)
    p.add_argument('--test-count',type=int,default=200);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true')
    p=sub.add_parser('visual-generate',help='Generate styled procedural HTML for browser-based visual training')
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('visual-build-data',help='Render HTML and verified corruption trajectories for parser and policies')
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--resume',action='store_true')
    for k,v in [('max-nodes',128),('trajectories',3),('max-noise',4),('abstract-size',384),('seed',42),('min-elements',1)]:p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--max-source-mae',type=float,default=1.,help='Reject rerenders too different from optional source screenshot; 1 disables')
    for name in ('visual-train-detector','visual-train-policy','visual-tree-train'):
        p=sub.add_parser(name)
        for k in ('train','val','out'):p.add_argument('--'+k,required=True)
        p.add_argument('--device',default='auto');p.add_argument('--resume');p.add_argument('--init-checkpoint')
        p.add_argument('--no-pretrained',action='store_true',help='Offline tests/ablation only; default downloads pretrained ImageNet backbone')
        p.add_argument('--bf16',action='store_true',help='Policy training only; detector uses float32')
        p.add_argument('--lr',type=float,default=1e-4)
        if name!='visual-tree-train':p.add_argument('--prediction-probability',type=float,default=0.)
        p.add_argument('--early-stop-patience',type=int,default=10,
                       help='Stop after this many consecutive validations without improvement; 0 disables')
        p.add_argument('--early-stop-min-delta',type=float,default=0.,
                       help='Minimum absolute validation-loss decrease to reset patience')
        for k,v in [('steps',10000),('batch-size',2),('accumulation',4),('seed',42),('cpu-threads',4),
                    ('eval-every',500),('log-every',50),('val-samples',32)]:p.add_argument('--'+k,type=int,default=v)
        if name.endswith('detector'):
            for k,v in [('min-size',800),('max-size',1600),('max-detections',256)]:p.add_argument('--'+k,type=int,default=v)
            p.add_argument('--metric-threshold',type=float,default=.4,
                           help='Confidence threshold for checkpoint-time IoU50 validation metrics')
        else:
            p.add_argument('--mode',choices=['screenshot','abstract'],required=True)
            if name=='visual-tree-train':
                p.add_argument('--stylesheets',action='store_true',help='Train CSS owner decoder; requires v2 prepared labels')
                p.add_argument('--predicted-targets',action='store_true',help='Use frozen detector cache rather than DOM target masks')
                p.add_argument('--online-corruption',action='store_true',help='Generate fresh CPU-browser corruption/path states during training only')
                p.add_argument('--online-targets',help='Training pages manifest containing clean tagged HTML; not initial_html')
                for key,default in [('online-workers',2),('online-prefetch',4),('online-max-noise',4),('online-attempts',8)]:
                    p.add_argument('--'+key,type=int,default=default)
                p.add_argument('--online-timeout',type=float,default=180.)
            else:
                p.add_argument('--policy-head',choices=['flat','hierarchical','autoregressive'],default='flat')
                p.add_argument('--numeric-only',action='store_true')
                p.add_argument('--balanced-policy',action='store_true',help='Legacy baseline only')
                p.add_argument('--decoding',choices=['joint','aggregate'],default='joint')
            for k,v in [('size',384),('hidden',128),('layers',3),('heads',4),('max-nodes',128),('token-grid',12)]:p.add_argument('--'+k,type=int,default=v)
    p=sub.add_parser('visual-cache-targets',help='Run frozen detector once per target for policy fine-tuning')
    p.add_argument('--reuse-cache',help='Reuse image-hash cache after verifying detector/threshold/size identity')
    for k in ('data','checkpoint','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='auto');p.add_argument('--threshold',type=float,default=.4)
    p.add_argument('--size',type=int,default=384);p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--resume',action='store_true')
    p=sub.add_parser('visual-evaluate-detector')
    for k in ('data','checkpoint','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='auto');p.add_argument('--threshold',type=float,default=.4)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--cpu-threads',type=int,default=4)
    p=sub.add_parser('visual-evaluate',help='Paired actual screenshot vs one-shot abstraction greedy rollouts')
    p.add_argument('--decoding',choices=['joint','aggregate'],help='Override flat decoding for a paired ablation')
    for k in ('data','raw-checkpoint','abstract-checkpoint','detector-checkpoint','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='auto');p.add_argument('--threshold',type=float,default=.4)
    p.add_argument('--goal-threshold',type=float,default=-1,
                   help='RGB MAE termination threshold; -1 disables for fixed-budget comparisons')
    p.add_argument('--abstract-goal-threshold',type=float,default=-1,
                   help='Semantic mask MAE termination threshold; calibrate on validation only; -1 disables')
    p.add_argument('--time-budget',type=float,default=0.,help='Per-method seconds including target preprocessing; soft operation-boundary limit')
    p.add_argument('--oracle-ablation',action='store_true',help='Controlled visual-build-data pages ONLY')
    p.add_argument('--resume',action='store_true')
    for k,v in [('steps',20),('repeats',3),('warmup',1),('limit',0),('seed',42),('cpu-threads',4)]:p.add_argument('--'+k,type=int,default=v)
    p=sub.add_parser('visual-figures',help='Generate paper-ready training and denoising figures')
    p.add_argument('--run-root',required=True,help='Root containing detector and policy stage directories')
    p.add_argument('--out',required=True)
    p.add_argument('--evaluation',help='Optional visual-evaluate output containing per-step traces')


def run(args):
    if args.command=='visual-tree-freeze':
        from .tree_freeze import freeze
        return freeze(args)
    if args.command=='visual-subset-numeric':
        from .visual_numeric import prepare
        return prepare(args)
    if args.command=='visual-tree-prepare':
        from .tree_edits import prepare
        return prepare(args)
    if args.command=='visual-tree-train':
        from .tree_policy import train
        return train(args)
    if getattr(args,'threshold',.4)<0 or getattr(args,'threshold',.4)>1:raise ValueError('threshold must be in [0,1]')
    if getattr(args,'metric_threshold',.4)<0 or getattr(args,'metric_threshold',.4)>1:raise ValueError('metric-threshold must be in [0,1]')
    for key in ('goal_threshold','abstract_goal_threshold'):
        value=getattr(args,key,-1)
        if value!=-1 and not 0<=value<=1:raise ValueError(key+' must be -1 or in [0,1]')
    if getattr(args,'limit',0)<0:raise ValueError('limit must be nonnegative')
    if getattr(args,'warmup',0)<0:raise ValueError('warmup must be nonnegative')
    if not 0<=getattr(args,'max_source_mae',1.)<=1:raise ValueError('max-source-mae must be in [0,1]')
    for key in ('size','abstract_size','trajectories','min_elements'):
        if hasattr(args,key) and getattr(args,key)<1:raise ValueError(key+' must be positive')
    if args.command=='visual-import-webui':
        from .webui_visual import import_webui
        return import_webui(args)
    if args.command=='visual-generate':
        from .visual_data import generate
        return generate(args)
    if args.command=='visual-build-data':
        from .visual_data import build
        return build(args)
    if args.command.startswith('visual-train-'):
        from .visual_train import train
        return train(args)
    if args.command=='visual-cache-targets':
        from .visual_train import cache_targets
        return cache_targets(args)
    if args.command=='visual-evaluate-detector':
        from .visual_train import evaluate_detector
        return evaluate_detector(args)
    if args.command=='visual-evaluate':
        from .visual_experiment import evaluate
        return evaluate(args)
    if args.command=='visual-figures':
        from .visual_figures import generate
        return generate(args)
    raise ValueError('Unknown visual command')

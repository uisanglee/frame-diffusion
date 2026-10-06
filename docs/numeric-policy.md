# CSS declaration-tree policy (current implementation)

For the new inline + embedded stylesheet policy, use [stylesheet-policy.md](stylesheet-policy.md).
The commands below preserve the inline-only v1 experiment.

This is the current policy-only workflow. Old detector checkpoints and cached
HTML/PNG observations are reusable. Old delta-policy checkpoints are NOT
compatible with this replacement decoder: train the policy in a new run.

## Four separate responsibilities

| Layer | Implementation | What it does NOT decide |
|---|---|---|
| Grammar legality | tree_edits.EditTokenizer / validate_edit | Whether the rendered result improves |
| Teacher construction | tree_edits.repair_path | Pixel/box gain or reversed corruption history |
| Learning | tree_policy.TreePolicy.loss | Set probability, reward weighting, STOP |
| Execution evaluation | visual_experiment.rollout + web-evaluate | Exact equality with a teacher action |

Reference inspected: [revalo/tree-diffusion at e8b29f2](https://github.com/revalo/tree-diffusion/tree/e8b29f27d6bd5e9fbf4679da49603df946b95eb1).
In td/samplers/mutator.py, a Mutation replaces source text; reverse stores the
original substring. find_path compares programs, not screenshots. Training in
scripts/train.py uses masked ordinary next-token cross-entropy. Constrained
decoding masks grammar-invalid continuations. Image goal checking is separate.

This implementation transfers those distinctions to a RESTRICTED fixed-DOM CSS
tree. It is not a byte-for-byte reproduction of TinySVG or full HTML synthesis.

## Scope and action

Editable declarations: width, height, margin-left, margin-right, margin-top,
margin-bottom. Padding, gap, flex, DOM insertion/removal, text and image content
are not edited. They remain in the page and participate in browser reflow.

An edit is:
  node position → property → SET value + priority / REMOVE → EOS

Examples: replace width:213px!important with width:50%; delete an inline
height override to restore stylesheet/auto behavior; replace margin-left:-12px.
EOS ends one edit, NOT the rollout. No learned STOP token exists.
A mutation edits exactly ONE declaration. Unlike the retired policy, this
restricts subtree size but does not bound a numeric delta to 5% of the viewport.

The finite CSS grammar supports decimal lengths, percentages, common units,
auto, CSS-wide keywords, and intrinsic size keywords. It does not support
arbitrary calc()/var() replacement expressions. Unsupported TARGET replacements
and differences outside the editable tree are rejected with explicit reports,
not approximated to pixels. Unchanged arbitrary CSS stays in the source.
Overlapping inline logical-size/margin and all declarations are rejected because
their ordering can affect the physical-property cascade outside this grammar.

CSSOM parses cached inline styles, including margin shorthands and priorities.
Teacher generation diffs actual current and target declarations. Repeated
corruption of one declaration is one difference, not several inverse actions.
A shuffled sequence of direct replacements must reach the exact target
declaration state. Offline mode labels cached states; online mode samples fresh
corruption and an intermediate reverse-path state each time.

Loss is ordinary teacher-forced token CE, averaged over non-padding output
tokens. No gain-weighted CE, log-sum-exp positive set, STOP objective, or
best-action accuracy is used. Validation CE diagnoses imitation, NOT successful
page repair. A different predicted edit can produce an equally correct image.

## Image conditioning

- Screenshot baseline: target RGB + current rendered RGB + absolute difference
  = 9 channels. Current screenshot is captured after each edit.
- Abstract policy: target/current four-class occupancy masks + difference
  = 12 channels. Target detector runs once per rollout; current masks come from
  current DOM geometry after layout/reflow. No current RGB capture is required.

Both encode their paired observation each step and decode a replacement token
sequence conditioned on current DOM geometry and current CSS declarations.
Target HTML/declarations and target boxes are teacher/evaluation information,
never inputs to rollout decisions. The optional abstract-oracle experiment is
explicitly privileged. Browser layout is still required; abstraction does not
mean CSS reflow is free.

## Online corruption (default in training scripts)

The CLI opts in with --online-corruption. GPU 1/3 and full training scripts
enable it by default. ONLINE_CORRUPTION=0 selects the offline baseline; use a
different output directory for that comparison.

CPU worker processes each own Chromium. They choose a training page, load its
clean tagged HTML, randomly mutate 1..ONLINE_MAX_NOISE declarations, build a
declaration-repair path, and execute a randomly chosen prefix of that path.
The next edit is the teacher. No image-distance ranking or learned-policy
rollout chooses the teacher. Existing and absent declarations are sampled, and
existing declarations can also be removed, so repairs include SET and REMOVE.
Source pages dominated by stylesheet rules may still yield mostly REMOVE labels.

Target screenshots and frozen detector predictions are reused. Workers never
load a detector/GPU model. Abstract mode queries current DOM layout without RGB
capture. Screenshot mode captures the same sampled state as PNG bytes in memory.
No transient image corpus is written to disk. Small per-worker caches retain
clean HTML/declaration states. New current states still require browser work.

An ordered bounded queue overlaps generation with GPU training. Seeds depend on
sample index, not worker completion order. Checkpoints save CONSUMED sample count;
resume regenerates from that index, discarding unconsumed prefetch. Equal seeds,
target pools and noise settings give the two modes the same intended states.
Browser failures/platform/font differences can still affect exact pairing.

Validation and test stay fixed: only TRAIN states are generated online.
Target manifests are checked against training identities. Clean HTML, target
image and predicted-target file hashes are checked on resume.

| Shell variable | Default | CLI |
|---|---:|---|
| ONLINE_CORRUPTION | 1 | --online-corruption |
| ONLINE_WORKERS | 2 | --online-workers |
| ONLINE_PREFETCH | 4 | --online-prefetch |
| ONLINE_MAX_NOISE | 4 | --online-max-noise |
| ONLINE_TIMEOUT | 180 seconds | --online-timeout (GPU wrappers) |

Two GPU jobs start four workers total by default. Start with one worker per job
if host RAM is limited. Samples retry at most --online-attempts (default 8), then
fail explicitly. Shutdown cancels queued work and closes worker browsers after
an in-progress browser call finishes/times out.

Logs include online_samples, online_retries, data_wait_s, producer_seconds,
online_teacher_remove_rate and online_mean_remaining_edits. Timing/rate counters
restart on resume; online_samples is the persisted global cursor. Producer time
is summed worker effort, not wall time. More workers do not guarantee a speedup.

Use --init-checkpoint and a NEW output directory to warm-start from an existing
declaration-tree policy. Do not --resume an offline run as online: the sampling
contract changed. Detector checkpoints and cached assets remain compatible.
Worker count, queue size and timeout may be tuned on resume; noise settings,
data and model settings must stay fixed.

## Reuse existing data and detector

From repository root, using the existing Python environment:

~~~bash
# One-time NEW label view; defaults to the existing v5 cached corpus.
# Accepts either the corpus root or its rendered/ subdirectory.
CACHE_GPU=4 bash scripts/prepare_numeric_policy_tree.sh
~~~

Defaults:
- Source: data/webui-10k-v5-improvement-distribution
- Frozen detector: runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt
- New label view: data/webui-css-tree-v1
- Predicted target cache: runs/css-tree-v1-shared

Supply these four paths as positional arguments if yours differ.
No dataset deletion, detector training, HTML screenshot recapture, or blanket
rerender occurs during preparation. Online TRAINING renders fresh current states.
Chromium is used only as an inert CSS parser during relabeling.
The frozen detector cache is created once for this view and reused on reruns.
Parsed-page caches avoid reparsing unchanged HTML. Missing source files cause a
clear error; the script does not silently switch to another corpus.

Inspect prepare-report.json and rejections-*.jsonl before full training. Each
split must remain nonempty. Rejected pages can reduce test count; report the
actual count. Test initial_html is selected from a retained repairable state.

If data/webui-css-tree-v1 and runs/css-tree-v1-shared are already prepared, SKIP
preparation. Online training obtains clean HTML from pages-train.jsonl and does
not require rewriting existing labels or detector caches.

## Two GPUs

~~~bash
mkdir -p log
ONLINE_WORKERS=2 ONLINE_PREFETCH=4 nohup bash scripts/train_numeric_policy_gpu1.sh > log/css-tree-online-screenshot.log 2>&1 &
ONLINE_WORKERS=2 ONLINE_PREFETCH=4 nohup bash scripts/train_numeric_policy_gpu3.sh > log/css-tree-online-abstract.log 2>&1 &
~~~

GPU 1: screenshot baseline. GPU 3: abstract policy.
Both default to 30,000 stage-1 / 15,000 stage-2 maximum steps with early stopping.
Stage 1 uses exact DOM target abstractions; stage 2 uses frozen detector outputs
for the abstract model. Screenshot training keeps RGB for both stages.
Each wrapper accepts the same four optional paths as documented in its usage.
Defaults now write runs/css-tree-online-v1-screenshot and
runs/css-tree-online-v1-abstract. For an offline ablation set ONLINE_CORRUPTION=0
AND supply a different fourth positional argument for the output directory.

Existing last.pt resumes only matching data/configuration. Stage 2 initializes
from the new stage-1 best.pt. Do not point these runs to old delta-policy output
directories. Detector training is never called by these scripts.

## Evaluate the same held-out controlled pages

~~~bash
CUDA_VISIBLE_DEVICES=1 \
PAGE_MANIFEST=data/webui-css-tree-v1/pages-test.jsonl \
RAW_CHECKPOINT=runs/css-tree-online-v1-screenshot/replacement/raw-stage2/best.pt \
ABSTRACT_CHECKPOINT=runs/css-tree-online-v1-abstract/replacement/abstract-stage2/best.pt \
DETECTOR_CHECKPOINT=runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
PAGE_LIMIT=0 REPEATS=3 REPAIR_STEPS=20 ORACLE_ABLATION=1 \
bash scripts/run_visual_web.sh data/raw/webui/test runs/eval-css-tree-online-v1 runs/css-tree-v1-shared
~~~

Use a new evaluation directory. Final geometry/pixel metrics and per-step
diagnostic traces measure execution outcomes. No exact teacher-action success
rate is reported for the replacement model. Geometry IoU is not the official
Design2Code metric. Count independent pages separately from repeat trials.

Fixed step/time budgets terminate rollout by default. Optional RGB/mask MAE
goal thresholds are external and must be calibrated separately on validation
data. They do not rank candidates or supervise teacher selection.

## Important differences and limits

- TinySVG can replace grammar subtrees; we currently replace six CSS declaration
  types on a fixed DOM. Missing/wrong elements from a VLM are outside this scope.
- Online training also samples fresh corruption and path intermediates, but its
  targets are fixed real HTML, not newly generated TinySVG programs. Our numeric
  CSS sampler is not the reference's full grammar/random-program mixture.
- Offline cached-data adaptation uses existing corrupted states and samples a first edge
  of their exact declaration-repair path. It does not fabricate reverse-path
  screenshots or claim the same training distribution.
  Reusing an old distance-filtered corpus also preserves that selection bias;
  relabeling alone cannot change the original corruption distribution.
- New visual-build-data samples executable numeric CSS corruption without a
  target-distance acceptance filter. It writes unlabeled observations; run
  visual-tree-prepare to obtain replacement supervision.
- train_visual.sh / finetune_visual_webui.sh use the same replacement teacher
  and trainer for newly built corpora. Use fresh output/data-view directories.
- Historical flat/hierarchical baseline code and old checkpoint readers remain
  for reproducibility. Gain/set-loss relabel commands and v7 preparation script
  are removed; old autoregressive delta training fails with a migration message.
- Existing corruption distribution may contain mostly inline overrides whose
  repair is REMOVE. The preparation report is not evidence of generalization;
  evaluate real VLM initial HTML separately from same-DOM controlled corruption.

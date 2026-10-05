# Numeric-only policy recovery

The detector is independent and remains frozen. The v6 policy uses 12 semantic
mask channels for TUIDE and 9 original screenshot RGB channels for its baseline.
The current policy version uses
`best-improving-reverse-v1` labels, so policy data must be rebuilt with the current
`visual-build-data` command.  Old v3 PNG/HTML assets may be archived, but their
last-mutation labels must not be used to train the autoregressive policy.  After
the v4 corpus exists, policy-only training never retrains the detector.  The fixed
detector runs once per target to cache predicted abstractions.

## Actions and comparison

Only width, height, margin-left, margin-right, margin-top and margin-bottom edits
are selectable by the autoregressive policy. Each numeric change is one of the ten viewport-normalized
deltas, bounded by 5%. Existing CSS flex/padding/gap is preserved, not edited.
This is size/margin repair, **not general structural reconstruction**.

Four comparisons can use the same numeric dataset:

1. Flat head + joint argmax: existing STOP-versus-individual-edit baseline.
2. The exact same flat checkpoint + aggregate decoding: compare STOP probability
   against the sum of all edit probabilities, then select the best edit.
3. Hierarchical head: P(operation) P(node|edit) P(property|node,edit)
   P(delta|property,node,edit). Greedy decoding follows that hierarchy. Training
   uses separately normalized operation, node, property, and delta losses.
   There is no forced minimum edit count.
4. Autoregressive head (default): jointly encodes
   `[target,current,abs(target-current)]`, points to a current LayoutIR
   node, and conditions property on that node and delta on both previous tokens.
   Its objective is the mean masked NLL of node, property and delta tokens. It has
   no learned STOP head or clean/STOP training examples. Rollout ends when the
   step/time budget is exhausted. Optional external image goal tests are configured
   separately with `--goal-threshold` (RGB) and `--abstract-goal-threshold` (masks).
   Both default to -1 (disabled) for equal-step comparisons; calibrate thresholds
   using validation pages only, never assume their numeric values are equivalent.

Screenshot feedback uses the original rendered RGB images (9 channels with pair
and difference), not color-coded box previews. Abstract feedback uses four
occupancy planes: text, image, control, painted-region (12 channels with pair and
difference). Box edges use fractional pixel coverage to retain tiny elements;
different classes can overlap. Masks use [0,1] values with no ImageNet RGB
normalization. RGB previews are for inspection only.

The detector processes the target once per rollout. Current masks come from the
current DOM geometry; browser layout/reflow is still required, but screenshot
capture is skipped. The paired encoder processes both masks at each step: only
the target abstraction is cached, not its jointly conditioned encoder features.
Timing reports distinguish pair encoding, policy decoding, DOM queries, layout,
and screenshot capture, and include actual model parameter counts.

Policy labels are not simply the inverse of the most recent corruption.  During
data construction, all legal reverse-path edits accumulated so far are tried in
the browser.  The label is the edit whose observed whole-page reflow most reduces
normalized target box distance.  Candidate rendering is an offline supervision
cost and is not part of inference.

For the autoregressive head, balanced-v1 samples the six edit properties uniformly
instead of in corpus-frequency proportion. Legacy flat/hierarchical baselines keep
their earlier balanced STOP/EDIT sampling and calibration solely for ablation.

The legacy 81-slot indexing is retained for checkpoint/action interoperability;
the 21 categorical slots are masked out, leaving 60 legal candidate slots per
eligible node. Width/height positivity and DOM legality masks still apply.
Old checkpoints remain loadable; new policy training starts with an ImageNet
image backbone and new policy weights, not the collapsed old policy. Detector
weights are independent and unchanged. Do not resume old policy output folders.

## Train (from the repository root)

If you already have v4 best-reverse data, reuse it directly: the numeric subset
command derives target elements from cached clean DOM boxes without a browser.
Cached detector prediction JSONs can also be reused by `visual-cache-targets`.
The mask representation change alone requires no HTML rerendering.

If you only have older last-mutation supervision, first create v4 policy supervision
in a new directory. This rerenders candidate
edits for labels but does not require retraining an existing detector:

```bash
python -m framediff visual-build-data \
  --manifest data/webui-10k-v3-nospacing-fresh-webui/source/manifest.jsonl \
  --out data/webui-10k-v4-autoregressive/rendered \
  --abstract-size 384 --trajectories 3 --max-noise 4 --resume
```

Then train the new policies with the frozen detector:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 \
BATCH_SIZE=2 ACCUMULATION=4 VAL_SAMPLES=2000 EVAL_EVERY=500 \
POLICY_STAGE1_STEPS=30000 POLICY_STAGE2_STEPS=15000 \
bash scripts/train_numeric_policy.sh \
  data/webui-10k-v4-autoregressive/rendered \
  runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
  runs/numeric-policy-v6-semantic
```

Use the actual existing detector path if different. The default now trains only
the autoregressive head in screenshot/abstract modalities and two stages.
`POLICY_HEADS="flat hierarchical autoregressive"` runs all architecture ablations.
Stage 1 uses stored ground-truth abstractions; stage 2 uses frozen detector
predictions for target abstractions. Screenshot stages use stored RGB images.
`POLICY_MODES=abstract` or `POLICY_MODES=screenshot` trains only one modality.
`PREPARE_DATA=0` may be used by parallel lanes after one process has completed
subset/cache preparation. Identical arguments resume each stage from last.pt. Changed settings or source
manifests require a new output directory. No source assets are regenerated.
Set `POLICY_RUN_DIR` to write new balanced checkpoints separately while reusing
the third argument's existing subset and predicted-target cache.
New training records an explicit observation contract and checkpoint kind
`visual-policy-v6-semantic`. Use a new policy output directory. Existing detectors
remain compatible; old policy checkpoints remain loadable for legacy evaluation
but cannot resume the new training contract.

The subset retains clean samples and complete numeric-only trajectory prefixes.
It drops categorical teacher edits **and all their descendants**, including ones
with numeric labels. Missing predecessor rows are conservatively excluded.
`subset/subset-report.json` records counts; `subset/pages-test.jsonl` selects a
remaining numeric-corrupted initial HTML per test page. Thus the filtered test
set is a new controlled evaluation, not directly comparable to old mixed-action
results. It is not a screenshot-to-HTML end-to-end benchmark.

## Paired held-out evaluation

```bash
CUDA_VISIBLE_DEVICES=1 \
PAGE_MANIFEST=runs/numeric-policy-v6-semantic/subset/pages-test.jsonl \
RAW_CHECKPOINT=runs/numeric-policy-v6-semantic/autoregressive/raw-stage2/best.pt \
ABSTRACT_CHECKPOINT=runs/numeric-policy-v6-semantic/autoregressive/abstract-stage2/best.pt \
DETECTOR_CHECKPOINT=runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
PAGE_LIMIT=0 REPEATS=3 REPAIR_STEPS=20 ORACLE_ABLATION=1 \
bash scripts/run_visual_web.sh data/raw/webui/test \
  runs/eval-numeric-policy-v6 runs/numeric-policy-v6-semantic
```

For comparison 1 use both `flat/...` checkpoints, `--decoding joint`, and a new
`--out .../eval-flat-joint`. For comparison 2 reuse those same checkpoints with
`--decoding aggregate` and `--out .../eval-flat-aggregate`. Never select a decoding
threshold on test results. Keep pages, steps, seeds and repeat counts identical.

For the autoregressive policy, monitor node accuracy,
`property_accuracy_given_correct_node`, `delta_accuracy_given_correct_node_property`,
per-token losses, goal MAE, and final rollout improvement. STOP-related metrics
apply only to legacy ablations. This change does not guarantee recovery:
corruption coverage, ambiguous reverse labels and real-image distribution shift
remain empirical issues.

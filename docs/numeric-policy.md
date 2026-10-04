# Numeric-only policy recovery

This workflow leaves the existing detector checkpoint, rendered PNGs, HTML files,
and original manifests untouched. It does not call a browser or train a detector.
The fixed detector runs inference once per target to cache predicted abstractions
in the new output directory. Repeating the command reuses that cache.

## Actions and comparison

Only width, height, margin-left, margin-right, margin-top, margin-bottom and STOP
are selectable. Each numeric change is one of the existing ten viewport-normalized
deltas, bounded by 5%. Existing CSS flex/padding/gap is preserved, not edited.
This is size/margin repair, **not general structural reconstruction**.

Three comparisons use the same numeric dataset:

1. Flat head + joint argmax: existing STOP-versus-individual-edit baseline.
2. The exact same flat checkpoint + aggregate decoding: compare STOP probability
   against the sum of all edit probabilities, then select the best edit.
3. Hierarchical head: P(operation) P(node|edit) P(property|node,edit)
   P(delta|property,node,edit). Greedy decoding follows that hierarchy. Training
   minimizes joint negative log likelihood (sum of conditional cross-entropies
   for a single teacher edit). There is no forced minimum edit count.

The legacy 81-slot indexing is retained for checkpoint/action interoperability;
the 21 categorical slots are masked out, leaving 60 legal candidate slots per
eligible node. Width/height positivity and DOM legality masks still apply.
Old checkpoints remain loadable; new policy training starts with an ImageNet
image backbone and new policy weights, not the collapsed old policy. Detector
weights are independent and unchanged. Do not resume old policy output folders.

## Train (from the repository root)

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 \
BATCH_SIZE=2 ACCUMULATION=4 VAL_SAMPLES=2000 EVAL_EVERY=500 \
POLICY_STAGE1_STEPS=30000 POLICY_STAGE2_STEPS=15000 \
bash scripts/train_numeric_policy.sh \
  data/webui-10k-v3-nospacing-fresh-webui/rendered \
  runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
  runs/numeric-policy-v1
```

Use the actual existing detector path if different. Default trains both flat and
hierarchical heads, each in screenshot/abstract modalities and two stages (eight
stages total). `POLICY_HEADS=hierarchical` trains only the new architecture.
Stage 1 uses stored ground-truth abstractions; stage 2 uses frozen detector
predictions for target abstractions. Screenshot stages use stored RGB images.
Identical arguments resume each stage from last.pt. Changed settings or source
manifests require a new output directory. No source assets are regenerated.

The subset retains clean samples and complete numeric-only trajectory prefixes.
It drops categorical teacher edits **and all their descendants**, including ones
with numeric labels. Missing predecessor rows are conservatively excluded.
`subset/subset-report.json` records counts; `subset/pages-test.jsonl` selects a
remaining numeric-corrupted initial HTML per test page. Thus the filtered test
set is a new controlled evaluation, not directly comparable to old mixed-action
results. It is not a screenshot-to-HTML end-to-end benchmark.

## Paired held-out evaluation

```bash
CUDA_VISIBLE_DEVICES=1 python -m framediff visual-evaluate \
  --data runs/numeric-policy-v1/subset/pages-test.jsonl \
  --raw-checkpoint runs/numeric-policy-v1/hierarchical/raw-stage2/best.pt \
  --abstract-checkpoint runs/numeric-policy-v1/hierarchical/abstract-stage2/best.pt \
  --detector-checkpoint runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
  --out runs/numeric-policy-v1/eval-hierarchical \
  --device cuda --steps 20 --repeats 3 --oracle-ablation
```

For comparison 1 use both `flat/...` checkpoints, `--decoding joint`, and a new
`--out .../eval-flat-joint`. For comparison 2 reuse those same checkpoints with
`--decoding aggregate` and `--out .../eval-flat-aggregate`. Never select a decoding
threshold on test results. Keep pages, steps, seeds and repeat counts identical.

Monitor `false_stop_rate`, `predicted_stop_rate`, `non_stop_accuracy`,
`edit_only_joint_accuracy`, per-property accuracy and final rollout improvement.
Low edit-only accuracy means separating STOP alone is insufficient. This change
does not guarantee recovery: corruption coverage, ambiguous inverse labels and
real-image distribution shift remain empirical issues. Stop accuracy alone and
declining training loss are not evidence of successful layout repair.

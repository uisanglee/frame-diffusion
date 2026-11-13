# Mixed geometry corruption

`visual-tree-train --online-corruption` and the official training scripts default to
`--corruption-policy mixed`. Use a **new training output**, not an old checkpoint
resume: the new model learns STOP (initial EOS) on clean samples. Vocabulary/tensor
sizes do not change; historical checkpoints retain their old decoding grammar.

Five requested profiles cycle deterministically with the consumed sample index:
clean, mild, medium, strong, multi. Each has a nominal 20% share before generation
failures/timeouts. Page selection remains uniform over pages, then over views.
Workers and resume do not change the index-seeded stream. Actual accepted shares
are reported, not assumed to be exactly 20% after skipped samples.

- Clean: no changed declarations; teacher is STOP.
- Mild/medium/strong: one existing width/height/margin declaration is changed by
  a nominal 1–4%, 4–12%, or 12–30% of its viewport axis, with either sign. Percentage
  values retain their unit. Sizes remain positive; existing explicit-size bounds,
  priority preservation, and visible-change probes still apply.
- Multi: up to four distinct CSS owners (bounded by `--online-max-noise`), nominal
  4–20% changes. At least two owners and two visible target elements must change.
  Insufficient editable owners/effects cause a reported generation failure, not a
  sample falsely labelled multi. Mixed mode requires max-noise >= 2.

Mixed samples retain the entire resulting repair path instead of randomly undoing
part of the corruption before supervision. The next teacher action still comes
from the exact original CSS values. Geometry is measured **after** all mutations.
Requested strong is not a guarantee of strong visual damage; clipping, cascade and
layout dependencies can suppress or amplify changes. No unbounded severity search
or RGB-based severity judgement is used.

## Measured geometry

Same-DOM box identities are used; synthetic viewport/root is excluded. Evaluate
positive-area target boxes intersecting the viewport. For each matched element:
position error is mean absolute top-left x/y error; size error is mean absolute
width/height error. Log means in pixels and normalized by viewport width/height.
Also log maximum normalized axis error, matched/missing element counts, changed
DOM elements (>1 px on any axis), changed declarations and changed owners.
Missing/non-finite boxes are reported as unmeasurable, never zero-error success.
Measured bins: clean if every axis error <=1 px; otherwise mild if max normalized
axis error <=.04, medium <=.12, strong above .12. These are diagnostic bins, not a
perceptual correctness definition. Fonts/colors are not compared as pixels.

## Frozen validation/test and caches

The CSS-first final publisher creates five fixed samples per held-out view, one
per profile. `visual-tree-freeze` defaults to five samples; mixed mode requires a
multiple of five. Original page split is preserved. If a view cannot supply all
conditions, its rejection is reported (there is no silent fallback to weak noise).
Check coverage and rejection counts before comparing versions.

Existing normalized targets, image/CSS download and inspection caches are reused.
The held-out corruption cache key includes the mixture version/settings. Old
single-noise results are not reused as mixed examples. Previous published manifests
are backed up by the final publisher; target assets are not rebuilt by refresh.
Train manifests continue to store clean targets; mixed samples are generated online.

If CSS-first preprocessing is unfinished, run its existing wrapper. If already
normalized, refresh only the fixed corruption/publication step:

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
mkdir -p log
nohup setsid env PYTHONUNBUFFERED=1 PREPARE_WORKERS=4 \
  bash scripts/refresh_webui_mixed_corruption.sh \
  >> log/webui-mixed-corruption.log 2>&1 < /dev/null &
echo $! > log/webui-mixed-corruption.pid
```

Logs:
- `data/new-css-v1/7000/css/corruption-val.jsonl`, `corruption-test.jsonl`: each sample.
- `data/new-css-v1/7000/css/freeze-report.json`: held-out geometry/profile summaries.
- `data/new-css-v1/split-report.json`: coverage and held-out rejections.
- Policy stage output `corruption-train.jsonl`: each consumed sample, step/index and geometry.
- Policy stage output `train.jsonl`: interval geometry/profile summary and ordinary learning metrics.
- Policy stage output `corruption-validation.json`: fixed validation geometry summary.
- Validation CE is also grouped by requested profile/action kind for stylesheet policies.

After preparing the mixed data, start a new Stage1, for example:

```bash
nohup setsid env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  POLICY_STAGES=1 CORRUPTION_POLICY=mixed \
  bash scripts/train_tuide_scale.sh s \
    data/new-css-v1/7000/css \
    runs/webui-css-mixed-v1/7000/detector/best.pt \
    runs/webui-css-mixed-v1/7000/shared \
    runs/webui-css-mixed-v1/7000/tuide-s-abstract \
  >> log/webui-css-mixed-s-stage1.log 2>&1 < /dev/null &
```

Stage1 does not require the detector path to exist. Stage2 requires the appropriate
trained detector and rebuilt target-cache manifests covering the new fixed samples;
target image predictions can be reused when image/checkpoint identities match.
Do not claim this distribution change alone fixes cross-DOM VLM generalization.

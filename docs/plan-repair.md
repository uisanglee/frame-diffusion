# Fixed visual plan → iterative layout repair

The research scope is element position, size, alignment and spatial relationships.
`web-prepare --repair-conditioning plan` sends three images to the frozen VLM:
the target screenshot, the initial HTML's full screenshot, and its named-frame map.
The DOM IDs/names/ancestry accompany the images. The VLM returns a single fixed
JSON plan of normalized intervals and pairwise constraints. It does not return
target bounding boxes. `repair-plan.prompt.json`, raw output, and `plan.json` are
saved for inspection. The default new script uses zero retries: one planner call.
Opting into retries increases the recorded call count.

Each edit updates CSS, measures all tracked DOM boxes in Chromium, creates a
cheap frame raster, and recomputes plan violations. Candidate selection never
reads the target screenshot or reference HTML. Initial and final full screenshots
are outside the iteration. Browser layout is **not eliminated**. A persistent
Chromium process loads candidate documents; `browser_executions` counts those
loads, separately from screenshots. Frame raster pixels are an audit output:
the model consumes measured geometry and encoded constraint states, not pixels.
The frame and its geometric representation describe the same state.
Intermediate frame rasters are capped at approximately 262,144 pixels; constraint
scoring still uses original-resolution measurements. `plan_node_coverage` reports
how much of the tracked node set appears in the plan, so a trivial partial plan
is not mistaken for complete reconstruction.

The frozen VLM is the visual planner. FrameDiff is newly trained with
`--conditioning plan`; it has a plan encoder, tree attention and edit/stop/value
heads. Old box-conditioned checkpoints are rejected for the new methods. The
value target is plan violation, not target IoU. The method is supervised iterative
tree-edit denoising, not a claim of reproducing an image diffusion model.

## Install

Use the existing conda environment (`environment-4090.yml` on the training
server), then `pip install -e '.[web,vlm,test]'` and
`python -m playwright install chromium`. No new mandatory dependency is added.

## Synthetic training and evaluation

```bash
CUDA_VISIBLE_DEVICES=0 DATA_COUNT=5000 TRAIN_STEPS=10000 TEST_LIMIT=100 \
  bash scripts/run_plan_synthetic.sh data/plan-synthetic runs/plan-main
```

Equivalent training command for already generated data:

```bash
CUDA_VISIBLE_DEVICES=0 python -m framediff train \
  --conditioning plan \
  --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl \
  --out runs/plan-main/train --device cuda --bf16 \
  --hidden 256 --layers 4 --batch-size 8 --accumulation 4 --steps 10000

python -m framediff plan-evaluate --data data/synthetic/test.jsonl \
  --checkpoint runs/plan-main/train/best.pt --out runs/plan-main/evaluation \
  --device cuda --limit 100 --steps 10 --beam 2 --topk 32 --budget 320
```

Teacher plans are generated from clean geometry with coarse intervals (default
width 0.05 of viewport) and sibling/containment relations. The same clean plan is
fixed while corruptions change. Labels are improving reverse mutations when
available, otherwise reverse mutations that can escape multi-edit dependencies.
States already satisfying the plan teach STOP. This is **oracle-plan policy
training/evaluation**, not evidence that the VLM can plan accurately. Explicit
`plan` fields in records override generated teacher plans.

`plan-evaluate` compares initial, coordinate and learned policy under the same
candidate limit, steps, beam and plan objective. `summary.json` reports plan
satisfaction alongside held-out box IoU, center/size error and relation accuracy.
The latter metrics use independent truth only after repair. A satisfied coarse
plan can still have poor IoU. Synthetic evaluation uses the first recorded
viewport; there is no multi-viewport claim.

## Real CSS training examples (recommended before real-web experiments)

Prepare a JSONL manifest from a **separate training corpus**, never the
Design2Code/Hard or WebUI test pages. Each line needs explicit group and split:

```json
{"html":"/datasets/train/site-a/page.html","group":"site-a","split":"train","viewport":[1280,720]}
{"html":"/datasets/train/site-b/page.html","group":"site-b","split":"val","viewport":[1280,720]}
```

```bash
python -m framediff plan-build-html-data \
  --manifest data/html-training.jsonl --out data/plan-html --per-page 16

CUDA_VISIBLE_DEVICES=0 python -m framediff train --conditioning plan \
  --train data/plan-html/train.jsonl --val data/plan-html/val.jsonl \
  --init-checkpoint runs/plan-main/train/best.pt --out runs/plan-html \
  --hidden 256 --layers 4 --device cuda --bf16 \
  --batch-size 8 --accumulation 4 --steps 5000
```

The builder uses exact DOM measurements and oracle plans to create real-browser
corruption examples. Each correction is re-executed in Chromium and retained
only if it decreases plan loss. Clean examples teach STOP. `report.json` records
failures and shortfalls; inspect it before training. This is verified one-edit
supervision, not a long-horizon optimal trajectory dataset. Serialized corrupted
HTML and target screenshots are retained for inspection. The loader uses stored
DOM geometry, never the approximate IR executor, for these training features.

Architecture options must match the initialization checkpoint. Initialization
resets optimizer/step; `--resume` instead restores both. Historical training groups
are retained to reject validation/evaluation overlap. Identical source HTML
across training splits is also rejected by the builder. Dataset curation must
still prevent near-duplicate sites crossing splits.

## Design2Code / Hard and raw WebUI

```bash
CUDA_VISIBLE_DEVICES=0 PAGE_LIMIT=5 \
  bash scripts/run_plan_web.sh "$D2C_ROOT" runs/plan-d2c runs/plan-html/best.pt

CUDA_VISIBLE_DEVICES=0 PAGE_LIMIT=5 \
  bash scripts/run_plan_web.sh "$D2C_HARD_ROOT" runs/plan-hard runs/plan-html/best.pt

CUDA_VISIBLE_DEVICES=0 WEB_DATASET=webui PAGE_LIMIT=5 \
  bash scripts/run_plan_web.sh "$WEBUI_ROOT" runs/plan-webui runs/plan-html/best.pt
```

Use `runs/plan-main/train/best.pt` if no real-HTML fine-tuning is available, and
report this as synthetic-only transfer. Set `PAGE_LIMIT=0` for all pages.
`WEBUI_VIEW` selects the raw WebUI viewport prefix. The WebUI root must contain
the downloaded screenshot/metadata files, not fitted `test.jsonl` records.

The baseline `coordinate-plan` searches local geometry edits using the same plan
and execution budget; `model-plan` proposes edits with FrameDiff. Set
`REVISION_ROUNDS=1` to also evaluate the existing independent VLM self-revision
branch from the same initial HTML. That baseline has its own image calls and
screenshots; they are not part of the plan repair loop. `VLM_RETRIES=2` enables
format retries, with all attempts charged. Model loading is excluded from page
timing as in the existing evaluator. Optional `OFFICIAL_REPO` enables official
Design2Code metrics; diagnostic geometry IoU is not its official metric.

For an already available initial HTML, use `INPUT_MANIFEST` with lines such as:

```json
{"id":"example","group":"held-out-site","screenshot":"/data/target.png","html":"/data/reference.html","initial_html":"/data/initial.html"}
```

The first positional dataset-root argument is ignored when `INPUT_MANIFEST` is
set. `html` is the reference for evaluation only. Existing prepared manifests
can be adapted with `initial_html` pointing to their initial method artifact.
Use a fresh output directory when switching from box to plan conditioning.

Results:

- `prepare/pages/*/plan.json`: fixed plan and planner prompt/raw response.
- `repair/pages/*/*-steps/step-*.png`: abstract frames, not full webpage captures.
- `repair/pages/*/*-trace.json`: plan loss, changed elements, failures, counts.
- `evaluation/report.md`, `summary.json`: end-to-end metrics, failures, time.
- `evaluation/pages/*/*.png`: final browser screenshots for visual inspection.

## Scope and interpretation

Real HTML actions currently modify width, height, margin-left and margin-top.
These can restore positions, sizes, alignment and spatial relationships and
observe coupled reflow. They do not reparent DOM nodes or reconstruct flex/grid
definitions. Synthetic IR additionally supports its existing flow/gap/order edits;
real-browser fine-tuning restricts actions to the real CSS action set.

The model reads structured plan features, not abstract raster pixels. Plan
features aggregate constraint measurements/bounds/residuals by kind and endpoint
at each node, with tree attention for context. The planner is frozen; no VLM or
image encoder training is performed. Incorrect, contradictory, incomplete plans
can still fail. Plan satisfaction must always be paired with independent final
image/geometry evaluation; do not claim that one-shot planning solves grounding.
Colors, fonts, image contents, missing elements and responsive reconstruction
remain outside this experiment's claimed scope.

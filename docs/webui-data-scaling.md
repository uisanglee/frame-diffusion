# WebUI 7K / 70K / 350K experiments

This is a custom 8:1:1 experiment, not the official WebUI split. Sizes denote
unique HTML pages selected before exclusions: 5,600/700/700,
56,000/7,000/7,000 and 280,000/35,000/35,000. One hostname-disjoint master
assignment is made first. Smaller samples are deterministic nested subsets of
each split. The same page/domain never switches from test to training when
increasing size. Exact-HTML duplicates are removed by the importer.

Pass the directory containing all downloaded raw pages, not just the historical
`test` directory. The selected viewport defaults to `default_1280-720`.
Missing HTML, duplicate HTML and viewport mismatches reduce availability.
Preparation fails if there are not enough domain-disjoint eligible pages; it
does not silently turn a 350K request into a smaller experiment. The paper's
reported total corpus size does not guarantee 350K eligible downloaded pages.

## Preparation

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
mkdir -p log

nohup env PREPARE_WORKERS=4 WEBUI_SIZES="7000 70000 350000" \
  bash scripts/prepare_webui_scales.sh data/raw/webui data/webui-scaling-811 \
  > log/prepare-webui-scaling-811.log 2>&1 &
```

No old 10K corpus or trained detector is required. Source HTML is first rendered
once into clean supervision (no offline corruption trajectories). This initial
`visual-build-data` pass remains sequential; normalization/candidate probing and
fixed validation/test corruption use `PREPARE_WORKERS`. Completion records and
existing configuration guards allow rerunning the same command. Keep settings
and paths stable; the normalization/freeze worker count may change on resume.

All sizes share images/HTML under `master/`; only manifests are duplicated.
**Do not delete `master/` after creating subsets.** Per-size `coverage.json`
reports originally selected, rendered, normalized and fixed-corruption counts.
Failed pages are not replaced with other pages to inflate coverage.

Detector supervision uses one clean browser-rendered screenshot per usable
page, with the existing TUIDE element classes (including painted regions).
It is not native WebUI AX-only supervision. Detector preparation failures are
counted at rendering; policy preparation adds normalization/corruption exclusions.
These differing denominators are recorded separately.

## Training and evaluation

After preparation completes, launch independent runs (one per GPU):

```bash
nohup env CUDA_VISIBLE_DEVICES=1 bash scripts/run_webui_scale.sh \
  7000 data/webui-scaling-811 runs/webui-scaling-811 > log/webui-7k.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=3 bash scripts/run_webui_scale.sh \
  70000 data/webui-scaling-811 runs/webui-scaling-811 > log/webui-70k.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=4 bash scripts/run_webui_scale.sh \
  350000 data/webui-scaling-811 runs/webui-scaling-811 > log/webui-350k.log 2>&1 &
```

Each run trains its own detector from the same default pretrained backbone,
evaluates it, creates a cache using that detector, trains TUIDE stage 1/2, then
evaluates fixed CSS rollouts. It never initializes from the old WebUI detector.
Default policy modes are abstract and screenshot, sequentially on the selected
GPU; `POLICY_MODES=abstract` selects just abstract. `POLICY_SCALE=s|m|l` and
`PARENT_FIRST=1` are supported through the existing training wrapper. Use a new
run root when changing experiment settings (especially parent-first).

`STAGE=detector`, `STAGE=policy`, or `STAGE=evaluate` runs just that stage;
default `STAGE=all` runs the pipeline. Stage checkpoints resume automatically
from `last.pt`. Detector defaults: 20,000 maximum steps, LR 1e-4. Policy defaults:
30,000/15,000 maximum steps. BATCH_SIZE=2, ACCUMULATION=4, VAL_SAMPLES=2000,
EVAL_EVERY=500, EARLY_STOP_PATIENCE=10. These are fixed-budget starting settings,
not a claim of equal epochs or convergence across scales. Override them when
needed and report actual steps/examples seen and early stopping for each run.

Both detector and policy are evaluated on:

- `own`: that size's full available test set;
- `common`: the smallest size's available test set, identical across models.

Use `common` to compare data scaling without changing evaluation pages. Best
checkpoints are selected by each size's own validation set. Detector metrics
are class-aware IoU50 precision/recall/F1 and small-object recall, not COCO AP.
Policy rollout outputs include repaired HTML/PNG, quality, timing and coverage
using the existing fixed-test evaluator. This is controlled held-out CSS repair;
Design2Code screenshot-to-HTML experiments remain a separate benchmark.

After all jobs finish:

```bash
python -m framediff visual-webui-scales-summary \
  --data data/webui-scaling-811 --runs runs/webui-scaling-811
```

The combined detector table is `runs/webui-scaling-811/detector-comparison.md`.
Machine-readable metrics with coverage are in `detector-comparison.json`.
Combined policy tables and coverage are in `policy-comparison.md/json`.
Policy reports are under each size's `eval-s-abstract-own`,
`eval-s-abstract-common`, `eval-s-screenshot-own`, and `eval-s-screenshot-common`.
Only run the aggregate command once concurrent jobs are finished.

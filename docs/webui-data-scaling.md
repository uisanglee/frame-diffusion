# WebUI 7K / 70K / 350K experiments

This is a custom 8:1:1 experiment, not the official WebUI split. Sizes denote
unique WebUI pages selected before exclusions: 5,600/700/700,
56,000/7,000/7,000 and 280,000/35,000/35,000. One hostname-disjoint master
assignment is made first. Smaller samples are deterministic nested subsets of
each split. Every available viewport of one page inherits that page's split,
and the same page/domain never switches from test to training when increasing
size. Exact-HTML duplicates across distinct pages are removed by the importer.

Pass the directory containing all downloaded raw pages, not just the historical
`test` directory. By default the importer uses four desktop captures plus
`iPad-Pro` and `iPhone-13 Pro`. `WEBUI_VIEWS` is comma-separated, for example
`WEBUI_VIEWS="default_1280-720,iPhone-13 Pro"`; spaces are part of view names. The
legacy `WEBUI_VIEW` variable deliberately selects one view. Missing HTML,
duplicate HTML and viewport mismatches reduce availability.
Preparation fails if there are not enough domain-disjoint eligible pages; it
does not silently turn a 350K request into a smaller experiment. The paper's
reported total corpus size does not guarantee 350K eligible downloaded pages.

## Preparation

### Diagnose insufficient raw pages before preparing again

This command does not render HTML, download archives, or change datasets:

```bash
python -m framediff visual-webui-audit \
  --root data/raw/webui-scale-source \
  --downloads data/downloads/webui-scale-source \
  --out runs/audit-webui-70k-source-v1 \
  --size 70000 --check-remote
```

`--check-remote` fetches Hugging Face file-list/size metadata only. Install
`huggingface_hub` in the active environment if unavailable. `7z` or `7zz` is
needed to list archive members. Add `--test-archives` for a full local CRC test;
this reads all compressed volumes and can be slow. Without CRC testing, matching
file sizes are not a guarantee of intact contents. Missing tools/network access
are reported as unknown/unavailable, never as successful checks.

- `audit-summary.json`: combined raw scan, allocation, download and extraction results.
- `source-scan.json`: unique pages and views at each eligibility stage.
- `source-exclusions.jsonl`: exact page/view, reason, file paths, duplicate target or mismatched hosts.
- `pages-with-no-usable-view.jsonl`: pages for which no view survived file checks.
- `selection-audit.json` / `selection-unselected.jsonl`: domain assignment shortfalls versus quota exclusions.
- `{train,val,test}/extraction-issues.jsonl`: requested archive members absent or wrong-sized after extraction.
- `{train,val,test}/archive-test.log`: optional full archive integrity test output.

The normal importer also saves source and selection audits under its source
output folder **before** raising an insufficient-pages error. Page-view losses
must not be added to whole-page losses. The existing any-shared-view-HTML
deduplication rule is reported explicitly; the audit does not relax it.

### Run preparation

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
mkdir -p log

nohup env PREPARE_WORKERS=4 WEBUI_SIZES="7000 70000" \
  bash scripts/prepare_webui_scales.sh data/raw/webui data/webui-scaling-811 \
  > log/prepare-webui-scaling-811.log 2>&1 &
```

No old 10K corpus or trained detector is required. Validation/test source HTML is rendered
once into clean supervision (no offline corruption trajectories). Initial
rendering, normalization/candidate probing and fixed validation/test corruption
use `PREPARE_WORKERS`. Set `RENDER_WORKERS` to override just the initial renderer
(for example `PREPARE_WORKERS=8 RENDER_WORKERS=4`). Completion records and
existing configuration guards allow rerunning the same command. Keep settings
and paths stable; all browser worker counts may change on resume.

Train manifests contain **lazy source descriptors**, not six pre-rendered images
per page. After selecting a page and viewport, the detector renders that view;
the policy renders, normalizes and probes that view before online corruption.
Each process retains at most two generated views in a temporary LRU cache.
Evicted views are regenerated when selected again; temporary caches are not
checkpointed. This trades training CPU time for bounded generated-train storage.
Stage 2 applies the frozen detector on demand in the training process (extra
GPU memory), not in browser workers. Validation detector outputs remain cached.
Train coverage is marked deferred, not falsely reported as 100% usable.
Raw extracted HTML/screenshots and fixed val/test assets still occupy disk;
this is not a guarantee the entire 70K experiment fits in 285 GB.

All sizes share held-out images/HTML under `master/`; only manifests are duplicated.
**Do not delete `master/` after creating subsets.** Per-size `coverage.json`
reports originally selected, rendered, normalized and fixed-corruption counts.
Failed pages are not replaced with other pages to inflate coverage.

Scale names count unique pages, not page-view observations. Training samples a
page uniformly and then one of its usable views uniformly, preventing pages
with six captures from receiving six times the probability. Validation and
test retain every usable page-view. Reports include unique-page counts,
page-view counts and per-view distributions.

Detector supervision uses a clean browser-rendered screenshot of the sampled
page-view, with the existing TUIDE element classes (including painted regions).
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
```

The default is 7K/70K only. 350K is opt-in and requires a larger raw source pool.

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
30,000/15,000 maximum steps. BATCH_SIZE=2, ACCUMULATION=4, VAL_SAMPLES=0 (all views),
EVAL_EVERY=500, EARLY_STOP_PATIENCE=10. These are fixed-budget starting settings,
not a claim of equal epochs or convergence across scales. Override them when
needed and report actual steps/examples seen and early stopping for each run.
A positive VAL_SAMPLES explicitly caps validation. Logs report available and
evaluated counts plus validation_complete. Symbolic action diagnostics remain
a separately labelled subset; they are not full validation rollout metrics.
Use a new data/run root for this lazy protocol, not an eager run's resume path.

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

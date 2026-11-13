# TUIDE paper experiment suite

## Current comparison

The default initial models are **LLaVA-NeXT-7B, GLM-4.1V-9B-Thinking,
WebSight VLM-8B, and UI2CodeN**. Qwen, Gemma, Pixtral and Design2Code-18B
implementations remain available through explicit `VLM_SPECS` selection, but
are not part of the default paper matrix.

| Initial model | Methods |
|---|---|
| LLaVA-NeXT-7B, GLM-4.1V-9B | Initial, RGB Self Revision, UI2CodeN polishing, TUIDE |
| WebSight VLM-8B | Initial, UI2CodeN polishing, TUIDE |
| UI2CodeN | Initial, its own Self Revision, TUIDE |

WebSight's official interface accepts one screenshot rather than the generic
current-HTML/two-image refinement prompt. Its Self Revision row is omitted.
UI2CodeN's own polishing is its Self Revision, not a second duplicate experiment.
`RUN_TUIDE=0` omits TUIDE while preserving Initial HTML for later repair.

The pilots use **two VLM refinement rounds**, 20 TUIDE steps, initial seed 42,
`INITIAL_TEMPERATURE=auto`, and one repair repeat. The generic runners allow
other settings; set `REVISION_ROUNDS=2` explicitly when using them directly.
WebUI uses a maximum of **16,384 output tokens**; Design2Code and
Design2Code-Hard use **4,096**. The central runner applies the dataset budget to
Initial, supported Self Revision and UI2CodeN polishing. EOS/complete HTML can
end generation early. These output caps are an explicit adaptation of the
specialist examples' total-sequence cap, not an extension of model context.

All methods within a page/model/initial seed start from the same saved Initial
HTML. Detector and TUIDE checkpoints remain fixed across initial seeds. WebSight
uses greedy decoding, so changing its seed does not create independent sampled
drafts; the combined default matrix uses seed 42. For genuinely sampled Initial
seeds, reports contain mean and sample SD across seed-level scores.

## Targets and evaluation

The matrix requires `TARGET_IMAGE_SOURCE=reference-render`. It renders each
reference HTML at the provided screenshot's viewport, saves `A_rendering.png`
and DOM boxes from the same browser observation, and shares those artifacts
across initial models and repairs. This is a **local-render layout evaluation**,
not standard Design2Code evaluation on the supplied PNG. Reports identify the
protocol. Reference-render time is separate from method-specific repair time.

For WebUI, use the final manifest from the CSS-first preparation described in
[webui-css-first-preparation.md](webui-css-first-preparation.md). Check its
renderings before running VLM generation. Changing target preparation requires a
new experiment output root; valid download caches remain reusable. Design2Code
and Hard keep their `rick.jpg` resource policy.

Quality is computed on each initial model/seed's successfully generated Initial
cohort. All repair methods use that same cohort. Initial failures stay in the
success-rate denominator. A repair failure retains the last valid implementation
and remains in the quality evaluation. Missing evaluator scores stop final table
export rather than silently becoming zero or excluding a successful page.

The current exporter reports Position, CLIP, Box IoU and Success / Improved (%):

- Initial's last column is generation success over all attempted pages.
- Repair rows report the fraction of successful Initial pages with higher Box IoU.
- CLIP follows the Design2Code adapter, including text-region masking.
- Block remains available in diagnostic reports.

Scroll-preservation checks and geometry-based repair protections remain active.
They are distinct from the success-cohort rule. Evaluation preprocessing does
not truncate HTML to remove repetitions. See [evaluation and cache details](paper-evaluation.md)
and individual run metadata for measured costs.

## Entry points

| Script | Purpose |
|---|---|
| `scripts/run_d2c_pilot10_bf16_2gpu.sh` | D2C/Hard VLM-only pilot; `D2C_PILOT_PAGES=3` selects three per dataset |
| `scripts/run_d2c_vlm_baselines.sh` | D2C/Hard VLM-only runs; default all pages, `D2C_BASELINE_PAGES` limits them |
| `scripts/run_d2c_vlm_parallel.sh` | Independent queues: GPU 0,1 for LLaVA/GLM; GPU 2,4 for WebSight/UI2CodeN |
| `scripts/run_paper_pilot10_bf16_2gpu.sh` | WebUI/D2C/Hard ten-page pilot with trained TUIDE |
| `scripts/run_paper_matrix.sh` | Full three-dataset matrix from explicit data/checkpoint paths |
| `scripts/run_paper_experiments.sh` | General per-dataset runner with explicit model/settings overrides |

See [D2C pilot details](d2c-vlm-pilot.md) for VLM-only experiments independent of
WebUI preparation and training. These retain Initial HTML, targets, self-revision
and polishing results for later TUIDE comparison.

### Setup and three-dataset pilot

Install the main optional dependencies, official metrics and isolated baseline
runtimes first. WebSight/UI2CodeN environments do not replace the main `.venv`.
See [specialist setup](screenshot-specialist-models.md) for optional Design2Code-18B.

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
pip install -e '.[vlm,visual,visual-metrics,test]'
bash baselines/ui2coden/setup_4090.sh
bash baselines/websight/setup.sh
mkdir -p log
```

Set `TRAIN_RUN` to the completed training run for the **same CSS-first data**,
and export its `DETECTOR_CHECKPOINT` and `ABSTRACT_CHECKPOINT` paths. The command
below requires these values instead of silently using checkpoints from an older
dataset. Do not run it until that training has completed.

```bash
: "${TRAIN_RUN:?Set the completed CSS-first training run}"
: "${DETECTOR_CHECKPOINT:?Set its detector best.pt}"
: "${ABSTRACT_CHECKPOINT:?Set its abstract Stage2 best.pt}"

nohup setsid env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  TRAIN_RUN="$TRAIN_RUN" DETECTOR_CHECKPOINT="$DETECTOR_CHECKPOINT" \
  ABSTRACT_CHECKPOINT="$ABSTRACT_CHECKPOINT" \
  WEBUI_TEST="$PWD/data/new-css-v1/7000/css/pages-test.jsonl" \
  OFFICIAL_REPO="$PWD/external/design2code-official" \
  PILOT_OUTPUT=runs/paper-css-first-pilot10-seed42-n2-v1 \
  bash scripts/run_paper_pilot10_bf16_2gpu.sh \
  >> log/paper-css-first-pilot10-seed42-n2-v1.log 2>&1 < /dev/null &
```

The pilot fixes ten unique WebUI pages (preferring 1280x720) and ten image/HTML
pairs from each Design2Code dataset. Saved `inputs/*.jsonl` manifests are checked
on resume. These deterministic smoke-test samples are not a representative
accuracy estimate. To use a different CSS-first output root, update `WEBUI_TEST`.

For the full matrix, use the same exported checkpoints and explicit input paths:

```bash
CUDA_VISIBLE_DEVICES=0,1 INITIAL_SEEDS=42 REVISION_ROUNDS=2 \
OFFICIAL_REPO="$PWD/external/design2code-official" \
bash scripts/run_paper_matrix.sh \
  data/new-css-v1/7000/css/pages-test.jsonl \
  data/raw/design2code/testset_final data/raw/design2code-hard \
  runs/paper-css-first-matrix-seed42-n2-v1 "$TRAIN_RUN"
```

### GPU placement and reproducibility

The default BF16 matrix splits each VLM across two GPUs, running models
sequentially within a queue. `VLM_DEVICE_MAP=balanced` requires two visible GPUs;
`VLM_GPU_MEMORY_GIB=20` is a per-GPU weight placement budget, not a total runtime
memory limit. The implementation rejects CPU/disk weight offload. WebSight uses
its dedicated two-GPU decoder partition. TUIDE and metric models run after VLM
workers release their memory. Two GPUs do not imply twice the throughput.

Model revision pins, resolved generation settings, token counts, timing and
per-device peak memory are saved. Keep `model-revision.json` with the results.
`sum_device_peak_cuda_bytes` sums per-device peaks; it is not a measurement of
simultaneous peak total memory. Report hardware and precision with results.
Optional single-GPU NF4 experiments remain available through explicit generic
runner settings and separate output directories; they are not this BF16 matrix.

### Optional models, APIs and ablations

The generic spec format is a comma-separated list of
`alias|backend|model|precision|endpoint|api_key_env|reasoning_effort|revision`.
Trailing fields may be omitted. Backends include `hf`, `qwen`, `websight`,
`design2code-sat` and `openai-compatible`. Provider credentials remain environment
variables. API adapters and usage/cost accounting remain available; the old
hardcoded provider/pricing launcher is removed. Select current provider settings
and pricing explicitly for any separately reported API experiment.

`RUN_ABLATION=1` in the per-dataset runner enables the representative-VLM
abstractions/RGB-policy comparisons. Supply the matching screenshot-policy
checkpoint when required. The main matrix does not load that unused checkpoint.
Externally generated HTML can still be evaluated via `scripts/run_external_html.sh`;
it does not need to reload or call the model that created the HTML.

## Resume and reports

Rerun the same command/output root to reuse compatible completed generation,
repair and evaluation results. Completed failed generations are recorded results,
not automatically retried. Model/input/prompt/checkpoint/protocol changes require
a new root or an explicit supported re-evaluation path. Never delete source data
or model-download caches to resume an experiment.

Each `<dataset>/<model>/seeds/42/` directory contains preparation, repair (when
enabled), UI2CodeN and evaluation artifacts. Per-model reports are under
`<dataset>/paper/main/<model>/`; `<dataset>/paper/README.md` indexes them.
`initial-seed-summary.json` records the Initial coverage and per-method cohort
sizes. The combined table and coverage are `table/vlm-quality.tex` and
`table/coverage.json`. LaTeX requires `graphicx` and `multirow`.

To rebuild a completed matrix table without generation or rendering:

```bash
python -m framediff.paper_matrix export \
  --root runs/paper-css-first-pilot10-seed42-n2-v1 \
  --out runs/paper-css-first-pilot10-seed42-n2-v1/table \
  --revision-rounds 2
```

This consumes existing per-dataset reports. If metrics or aggregation rules
changed, first regenerate those reports through the same runner; compatible
result/cache entries are reused. A missing metric must be resolved before export.

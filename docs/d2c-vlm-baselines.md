# Design2Code / Hard baselines before TUIDE training

## Parallel GPU pairs

`bash scripts/run_d2c_vlm_parallel.sh` runs two concurrent queues:
GPU 0,1 runs LLaVA and GLM; GPU 2,4 runs WebSight and UI2CodeN.
Each queue also performs polishing and evaluation on its assigned pair.
Both default to all pages and the same fixed seed/configuration as below.
Do not simultaneously run the earlier sequential command on those GPUs.

```bash
cd /home/uisang/2027/frame-diffusion
mkdir -p log
nohup setsid env PYTHONUNBUFFERED=1 \
  bash scripts/run_d2c_vlm_parallel.sh \
  >> log/d2c-vlm-parallel-seed42-n2-v1.log 2>&1 < /dev/null &
echo $! > log/d2c-vlm-parallel-seed42-n2-v1.pid
```

The default root is `runs/d2c-vlm-parallel-seed42-n2-v1` (override with
`D2C_PARALLEL_OUTPUT`). Logs are `gpu01.log` and `gpu24.log` inside it;
each queue retains complete artifacts in its own `gpu01/` or `gpu24/` directory.
The launcher waits for both queues and then checks identical input manifests
and completed metrics before exporting `table/vlm-quality.tex`.
`combined/<dataset>/<model>` links to the original artifacts for later TUIDE.
Keep both queue folders: combined outputs are links, not independent copies.
Same-directory reruns reuse compatible caches. Do not start two launchers on
the same output simultaneously. This new output does not automatically import
results from a previous sequential run.

To rebuild only the merged table after completion (no VLM calls):

```bash
.venv/bin/python -m framediff.d2c_parallel_merge \
  --root runs/d2c-vlm-parallel-seed42-n2-v1
```

## Sequential GPU pair

Run from the repository on the GPU server:

```bash
cd /home/uisang/2027/frame-diffusion
mkdir -p log
nohup setsid env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  bash scripts/run_d2c_vlm_baselines.sh \
  >> log/d2c-vlm-baselines-seed42-n2-v1-gpu01.log 2>&1 < /dev/null &
echo $! > log/d2c-vlm-baselines-seed42-n2-v1-gpu01.pid
```

Defaults: **all pages** from both datasets, seed 42, BF16 with balanced GPU
placement, 4096 maximum new tokens, two refinement rounds. Models execute
sequentially: LLaVA-NeXT-7B, GLM-4.1V-9B-Thinking, WebSight VLM-8B, UI2CodeN.
Existing model environments and the official metric dependencies are required.
No TUIDE or Detector checkpoint is loaded.

| Initial model | Subsequent methods |
|---|---|
| LLaVA, GLM | Self Revision, UI2CodeN polishing |
| WebSight | UI2CodeN polishing (image-only checkpoint; no Self Revision) |
| UI2CodeN | UI2CodeN Self Revision |

Set `D2C_BASELINE_PAGES=3` (or 10) and a **different**
`D2C_BASELINE_OUTPUT=runs/d2c-vlm-pilot3-seed42-n2-v1` for a small pilot.
The fixed input manifests reject changing the sample set within one output.
Dataset roots default to `data/raw/design2code/testset_final` and
`data/raw/design2code-hard`; override with `D2C_ROOT` / `D2C_HARD_ROOT`.
The target is the locally rendered reference HTML, not the supplied PNG.

## Results and later TUIDE input

Under `runs/d2c-vlm-baselines-seed42-n2-v1/`:

- `table/vlm-quality.tex`: combined results for the two datasets.
- `<dataset>/paper/README.md`: dataset reports and per-model summaries.
- `<dataset>/<model>/seeds/42/prepare/prepared.jsonl`: saved Initials and
  Self Revision metadata, directly consumable by the existing TUIDE evaluation
  pipeline. `methods.initial.html` identifies the Initial HTML for each page.
- `<dataset>/<model>/seeds/42/prepare/pages/`: raw responses, Initial HTML,
  generation metadata and per-page artifacts.
- `<dataset>/<model>/seeds/42/ui2coden/`: polishing results.

Initial generation failures are reported in success rates and excluded from the
shared quality cohort. A repair failure retains the last available valid result.
Position, CLIP and Box IoU use the existing evaluation implementation.

Rerun the **same command and output directory** to reuse compatible completed
generation and evaluation caches. Do not delete or move the output, source
datasets, `.cache/paper-reference-renders` or `.cache/application-assets`:
saved manifests/HTML may reference their files. Keeping only `initial.html`
is insufficient to preserve all local assets. TUIDE will consume these stored
Initials later without generating them again; it will still need its Detector
and policy checkpoints and its own repair/evaluation output.

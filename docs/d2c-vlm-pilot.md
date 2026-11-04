# Design2Code VLM-only pilot

Runs independently of WebUI preprocessing and Detector/TUIDE training. Uses ten
fixed, lexicographically selected page IDs from each Design2Code dataset (a smoke
pilot, not a representative benchmark estimate). Input manifests are frozen under
the output directory and shared by every model. On resume, changed selections are
rejected. Source HTML/PNG pairs and `rick.jpg` must exist in both input directories.

Models run sequentially: LLaVA-NeXT-7B, Gemma-3-12B, Pixtral-12B, UI2CodeN,
GLM-4.1V-9B-Thinking, Qwen3-VL-8B-Instruct. Each model is distributed over GPUs 0
and 1 with BF16 weights, no 4-bit quantization and no CPU weight offload. Default
weight budget is 20 GiB per GPU; runtime memory still depends on context/image size.
Requires access to the model repositories, including accepted Gemma access terms,
the main evaluation environment and `baselines/ui2coden/setup_4090.sh` setup.

```bash
cd /home/uisang/2027/frame-diffusion
mkdir -p log
nohup setsid env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  bash scripts/run_d2c_pilot10_bf16_2gpu.sh \
  >> log/d2c-pilot10-seed42-n2-bf16-2gpu.log 2>&1 < /dev/null &
echo $! > log/d2c-pilot10-seed42-n2-bf16-2gpu.pid
```

Defaults are in the script: seed 42, model-specific initial sampling settings,
16,384 maximum output tokens, two refinement rounds. Each ordinary VLM's Initial
is the common starting HTML for its own RGB Self-Revision and UI2CodeN polishing.
For UI2CodeN's own Initial, its polishing is its Self-Revision (not a duplicate
second experiment). Intermediate rounds are retained; the comparison selects N=2.
Original HTML is locally rendered for the target PNG/reference DOM, with the
existing Design2Code placeholder protocol. Valid shared reference caches are reused.

Default output: `runs/d2c-pilot10-seed42-n2-bf16-2gpu-v1`. Override `D2C_ROOT`,
`D2C_HARD_ROOT`, `OFFICIAL_REPO`, or `D2C_PILOT_OUTPUT` if needed. The output is
separate from prior WebUI/paper pilots. Rerun the same command after interruption
to resume completed stages; it does not automatically retry completed failed
generations. Generation parameter/input changes require a new output directory.

For each dataset and model, `<dataset>/<model>/seeds/42/` contains:

- `prepare/`: Initial, Self-Revision HTML/PNG, raw answers and usage metadata.
- `evaluation/metrics.jsonl`: per-page Initial/Self-Revision Position, CLIP, Box IoU,
  timing and token metrics.
- `ui2coden/`: polishing rounds and `evaluation/metrics.jsonl`.

After each dataset finishes, `<dataset>/paper/main/<model>/main-table.md` and
`paper-report.md` report quality, generation success, improvement and resource
metrics. After both datasets finish, `table/vlm-quality.tex` contains the combined
table and `table/coverage.json` records the attempted/successful counts.

Quality is calculated only on each model's successfully generated Initial cohort;
all methods share that cohort. Initial failures remain in the generation-success
denominator. Repair failures retain the last valid result and remain in the quality
comparison. Missing required evaluation metrics fail table export rather than
silently becoming zeros. Seed 42 alone does not estimate between-seed variance.

No Detector/TUIDE checkpoints are read or trained. The generated Initial files are
preserved for later TUIDE evaluation with matching targets and inputs.

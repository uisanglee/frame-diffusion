# Design2Code VLM-only pilot

Runs independently of WebUI preprocessing and Detector/TUIDE training. Uses ten
fixed, lexicographically selected page IDs from each Design2Code dataset (a smoke
pilot, not a representative benchmark estimate). Input manifests are frozen under
the output directory and shared by every model. On resume, changed selections are
rejected. Source HTML/PNG pairs and `rick.jpg` must exist in both input directories.

The official table uses four initial models, in this execution order:
LLaVA-NeXT-7B, GLM-4.1V-9B-Thinking, WebSight VLM-8B, and UI2CodeN.
Design2Code-18B, Qwen, Gemma, and Pixtral adapters remain available through
explicit `VLM_SPECS` selection but are excluded from the default matrix.

The BF16 two-GPU launcher uses GPUs 0 and 1. General Transformers models use
layer placement, including the dedicated WebSight worker. Set up the separate
WebSight and UI2CodeN environments before starting the matrix. `WEBSIGHT_PYTHON`
may override WebSight's Python executable. The optional Design2Code adapter uses a
separate SAT model-parallel runtime; it is not required for this matrix. No 4-bit quantization is enabled by this launcher. Runtime
memory depends on the model's image processing and context size; GPU execution
must still be verified on the server.

```bash
cd /home/uisang/2027/frame-diffusion
mkdir -p log
nohup setsid env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  D2C_PILOT_OUTPUT=runs/d2c-pilot10-seed42-n2-output4k-v1 \
  bash scripts/run_d2c_pilot10_bf16_2gpu.sh \
  >> log/d2c-pilot10-seed42-n2-bf16-2gpu.log 2>&1 < /dev/null &
echo $! > log/d2c-pilot10-seed42-n2-bf16-2gpu.pid
```

Defaults are in the script: seed 42, model-specific initial sampling settings,
4,096 maximum output tokens for every model and repair round, and two refinement
rounds. WebSight retains its official image preprocessing and decoding settings,
while its published total-length cap of 4096 is intentionally
replaced by an output-only cap of 4096 (input tokens are separate). Completed HTML or EOS ends generation early.
The checkpoint's input-plus-output context limit and available GPU memory still
apply; the output budget is not a context-length extension. WebUI experiments use
16,384 output tokens; both Design2Code datasets use 4,096. The central paper runner
applies this policy to Initial, supported Self Revision, and UI2CodeN polishing,
even if the launching shell still contains an older global token-budget variable.
WebSight does not receive the generic text-augmented chat prompt. Its official
protocol accepts one screenshot and does not implement two-image self-revision;
therefore its table group contains Initial, UI2CodeN polishing, and (when
requested) TUIDE, with no fabricated Self Revision scores. Each ordinary VLM's Initial
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

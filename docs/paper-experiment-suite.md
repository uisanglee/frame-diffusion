# TUIDE paper experiment suite

This suite produces two disjoint result groups from the same prepared benchmark protocol.

## Main comparison (run for every VLM)

1. `Initial VLM`: one screenshot-to-HTML generation.
2. `D2C Self-Revision (RGB)`: the Design2Code-style RGB self-revision round.
3. `TUIDE`: detector abstraction plus the learned abstract replacement policy.

## Ablation (one representative VLM)

1. `TUIDE`: learned policy on abstractions.
2. `TUIDE w/o abstraction`: the same learned-policy family using RGB screenshot feedback.
3. `TUIDE w/o policy (VLM abstract revision)`: target/current abstractions are passed to the VLM for one revision, replacing TUIDE's policy.

The abstraction VLM condition runs in a separate process from target detection, so a local VLM and detector do not occupy GPU memory simultaneously. The final reports include repair-only wall time, VLM calls, input/output tokens, browser executions, geometry, pixel MAE, failure rates, page-weighted confidence intervals, and paired gains.

## Run

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate

export VLM_SPECS='qwen3vl8b|qwen|Qwen/Qwen3-VL-8B-Instruct|bf16'
export ABLATION_VLM_ALIAS=qwen3vl8b
export CUDA_VISIBLE_DEVICES=1
export PAGE_LIMIT=20
export REPEATS=3
export REPAIR_STEPS=20

# Use these when detector and policies live in separate run directories.
export DETECTOR_CHECKPOINT=runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt
export RAW_CHECKPOINT=runs/css-owners-online-v3-screenshot/replacement/raw-stage2/best.pt
export ABSTRACT_CHECKPOINT=runs/css-owners-online-v3-abstract/replacement/abstract-stage2/best.pt

bash scripts/run_paper_experiments.sh \
  "$D2C_ROOT" \
  runs/paper-d2c \
  runs/webui-10k-v3-nospacing-fresh-webui
```

The fixed RTX 4090 comparison requested for the paper has a dedicated wrapper:

```bash
pip install -e '.[vlm,visual,visual-metrics,test]'
hf auth login  # required once for the gated Gemma weights

bash scripts/run_paper_4090_vlms.sh \
  "$D2C_ROOT" runs/paper-d2c runs/webui-10k-v3-nospacing-fresh-webui
```

It runs models sequentially with the following exact settings:

- `Qwen/Qwen3-VL-8B-Instruct`: BF16.
- `google/gemma-3-12b-it`: bitsandbytes NF4 4-bit with BF16 compute and double quantization.
- `mistral-community/pixtral-12b`: bitsandbytes NF4 4-bit with BF16 compute and double quantization. This is the Transformers-compatible conversion referenced by the Hugging Face Pixtral documentation.

Gemma is gated on Hugging Face, so accept its license and run `hf auth login` first. Local models are loaded in separate `web-prepare` processes and released before the detector/policies are loaded.

Run a two-page smoke test before the full benchmark:

```bash
CUDA_VISIBLE_DEVICES=1 PAGE_LIMIT=2 REPEATS=1 REPAIR_STEPS=2 \
bash scripts/run_paper_4090_vlms.sh \
  "$D2C_ROOT" runs/paper-d2c-smoke runs/webui-10k-v3-nospacing-fresh-webui
```

The general format is a comma-separated list of `alias|backend|model|precision` records. `backend` is `qwen`, `hf`, or `openai-compatible`; local precision is `bf16` or `4bit`:

```bash
export VLM_SPECS='qwen3vl8b|qwen|Qwen/Qwen3-VL-8B-Instruct|bf16,gemma3-12b|hf|google/gemma-3-12b-it|4bit,pixtral-12b|hf|mistral-community/pixtral-12b|4bit'
```

For a per-model OpenAI-compatible endpoint, append a fifth field: `alias|openai-compatible|model|server|endpoint`. Set `VLM_API_KEY` when required. The older `alias=model` format remains supported.

The full record also accepts provider-specific credentials and reasoning control:
`alias|backend|model|precision|endpoint|api_key_env|reasoning_effort`.

## GPT-4o and Gemini 3.5 Flash APIs

The API wrapper runs the main comparison only—`Initial VLM`, RGB Design2Code Self-Revision, and TUIDE—because the abstract-VLM condition belongs in the separate ablation. Both providers receive the same selected pages and Design2Code prompt protocol. Gemini uses low reasoning effort; GPT-4o does not receive a reasoning parameter.

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate

read -rsp 'OpenAI API key: ' OPENAI_API_KEY; export OPENAI_API_KEY; echo
read -rsp 'Gemini API key: ' GEMINI_API_KEY; export GEMINI_API_KEY; echo

CUDA_VISIBLE_DEVICES=1 PAGE_LIMIT=5 REPEATS=1 REPAIR_STEPS=20 \
bash scripts/run_paper_api_vlms.sh \
  "$D2C_ROOT" runs/paper-api-smoke runs/webui-10k-v3-nospacing-fresh-webui
```

After checking the smoke-test HTML and failure rate, run 100 pages with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=1 PAGE_LIMIT=100 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_paper_api_vlms.sh \
  "$D2C_ROOT" runs/paper-api-100 runs/webui-10k-v3-nospacing-fresh-webui
```

Each model normally makes two calls per page: one initial screenshot-to-HTML call and one RGB Self-Revision call. TUIDE reuses the same initial HTML and adds no VLM calls. Validation failures are retained and reported rather than silently removed. Measured provider usage is written to each method row, and the wrapper creates `api-cost.json` and `api-cost.md` under each model's paper output. Default cost rates are a dated snapshot and can be overridden with `GPT4O_INPUT_USD_PER_M`, `GPT4O_OUTPUT_USD_PER_M`, `GEMINI_INPUT_USD_PER_M`, and `GEMINI_OUTPUT_USD_PER_M`.

The wrapper uses the official Chat Completions endpoints directly; API keys remain environment variables and are not written to run configuration files.

For the official Design2Code evaluator, set `OFFICIAL_REPO` before running. If omitted, the suite still reports diagnostic geometry and pixel metrics.

## Outputs

- `runs/paper-d2c/<vlm-alias>/`: complete raw preparation, repair, and evaluation artifacts.
- `runs/paper-d2c/paper/main/<vlm-alias>/paper-report.md`: main comparison for each VLM.
- `runs/paper-d2c/paper/ablation/paper-report.md`: representative-VLM ablation.
- `runs/paper-d2c/paper/README.md`: index of all paper tables.

Use a fresh output directory if the model, page selection, checkpoint, or protocol changes. Resume is safe only when those inputs are unchanged.

## Externally generated GPT/Gemini HTML

You can generate HTML manually in GPT, Gemini, Claude, or another product and evaluate TUIDE without making any VLM request from this repository. Save the target screenshot and the complete generated HTML, then run:

```bash
SOURCE_MODEL='gemini-3.5-flash' \
CUDA_VISIBLE_DEVICES=1 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_external_html.sh \
  examples/my-page/target.png \
  examples/my-page/gemini-initial.html \
  runs/external-gemini-my-page \
  runs/webui-10k-v3-nospacing-fresh-webui
```

The two policy checkpoint variables may point at separate runs as in the main experiment. The script records `SOURCE_MODEL`, never loads that VLM, and evaluates the unchanged external HTML as `initial` before running both screenshot-policy and TUIDE.

If the original reference HTML is available, provide it as the fifth positional argument. This enables diagnostic DOM geometry and, when `OFFICIAL_REPO` is set, official Design2Code metrics:

```bash
bash scripts/run_external_html.sh \
  target.png gpt-initial.html runs/external-gpt "$TRAIN_RUN" reference.html
```

Without reference HTML, target-screenshot pixel MAE, repair time, actions, browser executions and failures are reported; DOM IoU and official HTML-reference metrics are intentionally unavailable. For multiple pages, create a JSONL manifest with the same fields (`id`, `screenshot`, `initial_html`, optional `html`, and `external_source_model`) and pass it through `PAGE_MANIFEST` to `scripts/run_visual_web.sh` with `SELF_REVISION_PROTOCOL=none`.

# TUIDE paper experiment suite

This suite produces two disjoint result groups from the same prepared benchmark protocol.

The paper wrapper defaults to `TARGET_IMAGE_SOURCE=reference-render`: it renders each
reference HTML once at the provided screenshot's viewport, saves `A_rendering.png`
and its DOM boxes from that same browser observation, and uses that PNG for the
initial VLM, all repair methods, and pixel MAE. This is a **local-render layout
evaluation**, not the standard Design2Code evaluation using the provided PNG.
Reports and JSONL rows identify the protocol. Use a new output directory when
changing it. The paper matrix requires this protocol. For separate provided-screenshot diagnostics, use `run_visual_web.sh` directly with `TARGET_IMAGE_SOURCE=provided`.
Reference rendering time is saved as `target_render_seconds` and is excluded
from method-specific pipeline and repair times because the same target is shared
by all methods on a page.

## Main comparison (run for every VLM)

1. `Initial VLM`: one screenshot-to-HTML generation.
2. `D2C Self-Revision (RGB)`: three consecutive RGB self-revision rounds; the table uses round 3.
3. `UI2CodeN polishing`: three consecutive polishing rounds; the table uses round 3.
4. `TUIDE`: detector abstraction plus the learned abstract replacement policy.

All repair methods start from the same saved `prepare/prepared.jsonl` initial HTML
for each page, VLM, and initial-generation seed. The TUIDE and detector checkpoints
remain fixed across seeds. Only the **initial VLM draft** is sampled with a
different seed. VLM self-revision and UI2CodeN polishing start from that seed's
draft. The target PNG and reference DOM are reused from the first VLM/seed run so the
evaluation target does not change. The suite verifies shared initial inputs
within each seed and identical checkpoints and targets across seeds, then reports
mean ± sample SD across initial-generation seeds. Page-bootstrap confidence
intervals remain available in each seed's report.

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
export INITIAL_SEEDS=11,22,33
export INITIAL_TEMPERATURE=0.7

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
bash baselines/ui2coden/setup_4090.sh  # isolated UI2CodeN environment

bash scripts/run_paper_4090_vlms.sh \
  "$D2C_ROOT" runs/paper-d2c runs/webui-10k-v3-nospacing-fresh-webui
```

It runs models sequentially with the following exact settings:

The 4090 wrapper runs both self-revision and UI2CodeN for exactly three rounds
(`RUN_UI2CODEN=1` by default). Use a **new output directory** if an older run
was prepared with one self-revision round; its resume configuration is different.
Set `RUN_UI2CODEN=0` only for a quick model smoke test; the resulting main table
will omit that baseline.

Set `INITIAL_SEEDS=11,22,33` and `INITIAL_TEMPERATURE=0.7` to generate three
independent initial drafts per VLM. The first VLM/seed run renders the target reference;
all later runs reuse its target PNG and DOM observation. Every seed has its own
`<VLM>/seeds/<seed>/prepare`, `repair`, `ui2coden`, and `evaluation` directory.
The fixed checkpoint hashes are checked before aggregation. The output
`paper/main/<VLM>/main-table.md` reports mean ± SD across initial seeds for
**all** methods; `initial-seed-summary.json` preserves the per-seed values.
With one seed, the wrapper uses greedy initial generation and reports no SD.

- `Qwen/Qwen3-VL-8B-Instruct`: BF16.
- `google/gemma-3-12b-it`: bitsandbytes NF4 4-bit with BF16 compute and double quantization.
- `mistral-experimental/pixtral-12b`: bitsandbytes NF4 4-bit with BF16 compute and double quantization. This is the Transformers-compatible conversion referenced by the Hugging Face Pixtral documentation.
- `zai-org/GLM-4.1V-9B-Thinking`: bitsandbytes NF4 4-bit with BF16 compute and double quantization.
- `llava-hf/llava-v1.6-mistral-7b-hf` (LLaVA-NeXT-7B): bitsandbytes NF4 4-bit with BF16 compute and double quantization. Its model-specific loader accepts the target/current two-image self-revision prompt.

Gemma is gated on Hugging Face, so accept its license and run `hf auth login` first. Local models are loaded in separate `web-prepare` processes and released before the detector/policies are loaded.

To smoke-test only LLaVA-NeXT-7B on a 24 GB RTX 4090, use a new output directory:

```bash
CUDA_VISIBLE_DEVICES=0 PAGE_LIMIT=2 REPEATS=1 REPAIR_STEPS=2 \
VLM_SPECS='llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|4bit' \
RUN_ABLATION=0 \
bash scripts/run_paper_4090_vlms.sh \
  "$D2C_ROOT" runs/paper-llava-next-smoke "$TRAIN_RUN_DIR"
```

The loader requires the quantized model to fit entirely on the selected GPU; it
raises an error rather than silently moving layers to CPU. Check the smoke-test
HTML and GPU peak-memory metadata before scaling to the full set.

Run a two-page smoke test before the full benchmark:

```bash
CUDA_VISIBLE_DEVICES=1 PAGE_LIMIT=2 REPEATS=1 REPAIR_STEPS=2 \
bash scripts/run_paper_4090_vlms.sh \
  "$D2C_ROOT" runs/paper-d2c-smoke runs/webui-10k-v3-nospacing-fresh-webui
```

The general format is a comma-separated list of `alias|backend|model|precision` records. `backend` is `qwen`, `hf`, or `openai-compatible`; local precision is `bf16` or `4bit`:

```bash
export VLM_SPECS='qwen3vl8b|qwen|Qwen/Qwen3-VL-8B-Instruct|bf16,gemma3-12b|hf|google/gemma-3-12b-it|4bit,pixtral-12b|hf|mistral-experimental/pixtral-12b|4bit'
```

For a per-model OpenAI-compatible endpoint, append a fifth field: `alias|openai-compatible|model|server|endpoint`. Set `VLM_API_KEY` when required. The older `alias=model` format remains supported.

The full record also accepts provider-specific credentials and reasoning control:
`alias|backend|model|precision|endpoint|api_key_env|reasoning_effort|revision`.

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

The paper runner requires `OFFICIAL_REPO`. Before loading any VLM, it checks actual official Block/Position scoring on a small identical HTML pair. Missing dependencies or incomplete evaluation scores stop the paper export; diagnostic-only runs remain available via `run_visual_web.sh`.

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

### Local model reproducibility and GPU placement

The paper runner resolves each local model revision (default `VLM_REVISION=main`, or the eighth spec field) to an immutable Hub commit SHA before generation. It saves the pin in `<output>/<alias>/model-revision.json` and reuses it on resume. Model and processor use that revision. Keep this file with experiment results. Existing outputs with `prepare/config.json` but no pin require a new output directory so results from unknown revisions are not mixed.

Local Qwen and HF loaders place model weights on one current CUDA device, with no automatic weight offload. Set `CUDA_VISIBLE_DEVICES` to one GPU for these experiments. Placement is checked after loading and recorded alongside generation metadata (`device`, `gpu_name`, `device_map`, `placement_contract`). BF16 Qwen explicitly offloads its non-quantized KV cache to CPU to reduce GPU memory pressure at the 16384-token output ceiling; NF4 Qwen uses a GPU dynamic cache. Cache offloading can increase latency and must be reported with timing results (`kv_cache_offloaded`, `memory_contract`). It does not guarantee that all inputs fit in GPU memory. These changes do not modify TUIDE training checkpoints or the shared page cache.

### GLM-4.1V-9B-Thinking on one RTX 4090

The 4090 matrix also includes `zai-org/GLM-4.1V-9B-Thinking` with NF4 4-bit weights and BF16 compute. Install updated optional dependencies (`pip install -e '.[vlm,visual,visual-metrics,test]'`); Transformers 4.57.1 or newer within the supported 4.x range is required. It uses `Glm4vForConditionalGeneration`, the shared revision pin, and the single-CUDA placement check.

Run GLM alone first, using the same dataset and trained-policy paths as the other experiments:

```bash
CUDA_VISIBLE_DEVICES=2 \
VLM_SPECS='glm4v-9b-thinking|hf|zai-org/GLM-4.1V-9B-Thinking|4bit' \
RUN_ABLATION=0 PAGE_LIMIT=2 REPEATS=1 REPAIR_STEPS=2 \
bash scripts/run_paper_4090_vlms.sh \
  "$D2C_ROOT" runs/paper-glm4v-smoke "$TRAIN_RUN_DIR"
```

Use a separate output directory for the full benchmark and match `PAGE_LIMIT`, `REPEATS`, `REPAIR_STEPS`, image resolution, and generation token budget across comparisons. Thinking tokens count toward the generation limit, token usage, and latency. Only the final answer after `</think>` is passed to HTML processing; an unfinished `<think>` response is rejected as empty output. Real 4090 memory usage must be checked in the smoke test; 4-bit does not guarantee that arbitrary image/context lengths fit.

## Three datasets × six initial models

The default 4090 wrapper includes UI2CodeN as the sixth initial model. Its group
contains **Initial / Self Revision / TUIDE**. Initial uses the shared text-augmented
drafting prompt and pretrained `zai-org/UI2Code_N` in 4-bit. Self Revision uses the
isolated UI2CodeN polishing adapter for three rounds, starting from its own saved
draft. This is our common evaluation protocol, not a reproduction of the original
paper's entire evaluation setup.

WebUI uses a held-out HTML/screenshot manifest (for example `css/pages-test.jsonl`).
Its target is rendered locally and the DOM comes from that same observation;
old recorded accessibility boxes are not used. Design2Code and Design2Code-Hard
use the same local-render protocol and the same official Block/Position adapter.
Improved is the fraction of pages whose Box IoU exceeds their corresponding Initial.

Activate the main environment, install the official metric dependencies and set
up `baselines/ui2coden/setup_4090.sh` first. Provide the actual paths below:

```bash
CUDA_VISIBLE_DEVICES=0 \
OFFICIAL_REPO="$PWD/.cache/design2code-official" \
DETECTOR_CHECKPOINT="/absolute/path/to/detector/best.pt" \
ABSTRACT_CHECKPOINT="/absolute/path/to/replacement/abstract-stage2/best.pt" \
INITIAL_SEEDS=11,22,33 \
bash scripts/run_paper_matrix.sh \
  data/webui-full-dom-screened-7k-v1/7000/css/pages-test.jsonl \
  /absolute/path/to/Design2Code \
  /absolute/path/to/Design2Code-Hard \
  runs/paper-layout-matrix \
  /absolute/path/to/trained-tuide-run
```

The wrapper runs sequentially on one GPU and exports
`runs/paper-layout-matrix/table/vlm-quality.tex` and `coverage.json`.
LaTeX requires `graphicx` and `multirow`. The main matrix does not load the
unused screenshot policy; optional ablations are separate (`RUN_ABLATION=1`
in the per-dataset runner). Model/detector checkpoints stay fixed, `REPEATS=1`,
and only Initial generation seeds vary. Mean and sample SD are across those seeds.

Repair failure retains the **last successfully produced HTML**, including when
UI2CodeN or a TUIDE browser operation fails after previous successful edits.
It does not reset that page to Initial. If no edit succeeded, Initial is retained.
Failed trials remain in the evaluation with their failure flags. Existing scroll
preservation rejection is still enforced; it is not a quality-score rollback.
Resume keeps existing saved results; use a fresh output directory to regenerate
results produced by an older failure-handling implementation.

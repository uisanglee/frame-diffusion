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

### RTX 4090 두 장, BF16 예비실험

기존 4bit 실험과 분리해 6개 initial VLM과 UI2CodeN polishing을 모두 BF16으로
실행하는 preset이다. GPU 두 장은 **하나의 모델 레이어를 분담**하며 모델들은 순서대로
실행한다. WebUI/Design2Code/Hard 각 10개, seed 42, 수정 2회, TUIDE 20 step을 유지한다.
TUIDE/평가 모델은 기존 단일 GPU 경로를 사용하며 VLM 프로세스 종료 후 실행된다.

```bash
cd /home/uisang/2027/frame-diffusion
mkdir -p log
nohup setsid env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  bash scripts/run_paper_pilot10_bf16_2gpu.sh \
  >> log/paper-pilot10-seed42-n2-bf16-2gpu.log 2>&1 < /dev/null &
```

출력은 `runs/paper-pilot10-seed42-n2-bf16-2gpu-v1`이다. 기존 initial/4bit 결과는
삭제하지 않는다. 동일한 새 명령어는 같은 설정의 중간 결과를 재사용한다.
전체 Qwen 실험만 실행하려면 다음을 사용한다(예비실험과 별도 출력).

```bash
CUDA_VISIBLE_DEVICES=0,1 VLM_DEVICE_MAP=balanced VLM_GPU_MEMORY_GIB=20 \
  REVISION_ROUNDS=2 bash scripts/run_paper_seed42.sh qwen
```

기본 단일 GPU 실행 방식은 유지한다. `VLM_DEVICE_MAP=balanced`는 정확히 두 장과
BF16을 요구하며, 모델 설정과 GPU 여유 메모리를 이용해 가중치 로딩 전에 배치를 계획한다.
CPU/disk offload는 허용하지 않는다. `VLM_GPU_MEMORY_GIB`(기본 20)는 각 GPU의
가중치 배치 예산이며 총 실행 메모리 상한이 아니다. SDPA와 KV 캐시를 사용하고,
지원되는 HF 모델에서는 마지막 토큰의 logits만 계산한다. 출력 길이/샘플링/이미지
해상도/반복 횟수를 속도를 위해 줄이지 않는다. 추가 FlashAttention 패키지는 필요 없다.

새 결과는 `config.json`에 deployment 설정을 기록하여 다른 배치 설정과 섞어
재개하지 못하게 한다. GPU 두 장에 분산해도 속도가 두 배가 되지 않으며, 통신 비용이
추가된다. 속도·최대 메모리는 실제 서버에서 측정해야 한다. 각 생성 meta와 UI2CodeN
round JSON의 `peak_cuda_bytes_by_device`는 GPU별 피크, `peak_cuda_bytes`는 그중
최댓값이다. `sum_device_peak_cuda_bytes`는 장치별 피크의 합으로, 동시 총사용량의
피크와는 다르다. 시간은 사용 GPU를 모두 동기화한 뒤 기록한다. 논문에서는 GPU 수와
정밀도를 명시하고, 동일 하드웨어 비교와 각 방법의 실제 자원 비용을 구분한다.

### Existing single-GPU suite

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
hf auth login  # if required for your selected repositories
bash baselines/ui2coden/setup_4090.sh  # isolated UI2CodeN environment
bash baselines/websight/setup.sh
bash baselines/design2code/setup.sh
export CUDA_VISIBLE_DEVICES=0,1

bash scripts/run_paper_4090_vlms.sh \
  "$D2C_ROOT" runs/paper-d2c runs/webui-10k-v3-nospacing-fresh-webui
```

The current default matrix runs five BF16 models sequentially: LLaVA-NeXT-7B,
GLM-4.1V-9B, WebSight VLM-8B, Design2Code-18B-v0, and UI2CodeN. Use
`CUDA_VISIBLE_DEVICES=0,1`; Design2Code uses SAT model parallelism. Install the
separate WebSight and Design2Code runtimes before running this wrapper (see
`docs/d2c-vlm-pilot.md`). Qwen/Gemma/Pixtral remain optional individual baselines;
they are no longer part of the official matrix.

See [specialist setup and protocol details](screenshot-specialist-models.md) for
the separate environments and SAT LoRA partitioning adaptation.

WebSight and Design2Code follow their official image-only drafting interfaces,
including a total `max_length=4096`. They do not implement the generic two-image,
current-HTML Self Revision interface, so that row is omitted. Their saved drafts
are still repaired by UI2CodeN and TUIDE. Other VLMs retain Self Revision.
Refinement rounds are selected by `REVISION_ROUNDS`; the pilot scripts use two.

Use seed 42 and `INITIAL_TEMPERATURE=auto` for the official model defaults.
WebSight's greedy decoding does not produce an independent sampling experiment
by changing the seed; the combined official matrix therefore uses one seed.
Do not override its temperature to manufacture seed variance. Model profiles,
image preprocessing, and output-length differences must be reported with results.
The first model renders each reference; later models reuse that same target PNG
and DOM observation. Each model stores its own `seeds/42/prepare`, `repair`,
`ui2coden`, and `evaluation` outputs. Checkpoint hashes are checked at aggregation.

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

The general format is a comma-separated list of `alias|backend|model|precision` records. `backend` is `qwen`, `hf`, `websight`, `design2code-sat`, or `openai-compatible`; local precision is `bf16` or `4bit`:

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

The paper runner requires `OFFICIAL_REPO`. Before loading any VLM, it checks actual official Position/CLIP scoring on a small identical HTML pair. Missing dependencies or incomplete evaluation scores stop the paper export; diagnostic-only runs remain available via `run_visual_web.sh`.

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

Local Qwen and HF loaders place model weights on one current CUDA device, with no automatic weight offload. Set `CUDA_VISIBLE_DEVICES` to one GPU for these experiments. Placement is checked after loading and recorded alongside generation metadata (`device`, `gpu_name`, `device_map`, `placement_contract`). Both BF16 and NF4 Qwen use a GPU dynamic KV cache, with `kv_cache_offloaded=false`. The BF16 memory contract is `bf16-dynamic-kv-v1`; previous CPU-offload results must use a separate output directory. BF16 weights plus a long KV cache can exceed 24 GB at the 16384-token ceiling. These changes do not modify TUIDE training checkpoints or the shared page cache.

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

## Three datasets × five initial models

The current roster is listed above. UI2CodeN's group contains
**Initial / Self Revision / TUIDE**; Self Revision uses its own polishing adapter
starting from its saved draft. The Design2Code/WebSight groups contain
**Initial / UI2CodeN polishing / TUIDE**. This is our common repair evaluation,
not a reproduction of each original paper's entire benchmark protocol.

WebUI uses a held-out HTML/screenshot manifest (for example `css/pages-test.jsonl`).
Its target is rendered locally and the DOM comes from that same observation;
old recorded accessibility boxes are not used. Design2Code and Design2Code-Hard
use the same local-render protocol and the same official Position/CLIP adapter.
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

The main quality table reports Position, CLIP, Box IoU, and Improved (%). CLIP uses the official Design2Code score (text regions are masked before CLIP encoding), not raw full-screenshot CLIP. Improved (%) remains the fraction of pages with increased Box IoU over their initial implementation. Existing evaluation metrics containing `official_clip` can be re-aggregated without regeneration or rendering; missing CLIP scores must be evaluated before final export. Block remains available in diagnostic reports.

## Ten-page pilot per dataset

Run `bash scripts/run_paper_pilot10.sh` from the repository virtual environment (the wrapper activates `.venv`). Default output: `runs/paper-pilot10-seed42-n2-v1`. It uses fixed seed 42, automatic model decoding settings, up to 16,384 generated tokens, two VLM refinement rounds, and 20 TUIDE steps. All five initial VLMs run sequentially on `CUDA_VISIBLE_DEVICES`; checkpoints are reused without training. The official metrics dependencies, Hugging Face model access, and UI2CodeN environment must already be installed.

Each dataset contributes ten fixed inputs. WebUI selects ten unique pages, preferring the 1280x720 view for each; the other benchmarks select ten image/HTML pairs. These are deterministic smoke-test samples, not a representative accuracy estimate. Input manifests are saved under `inputs/` and checked on resume. WebUI's external asset limitation remains; inspect pilot renderings before scaling up.

```bash
mkdir -p log
nohup setsid env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  bash scripts/run_paper_pilot10.sh \
  >> log/paper-pilot10-seed42-n2-gpu0.log 2>&1 < /dev/null &
```

After all groups finish, `table/vlm-quality.tex` contains Position, CLIP, Box IoU and Improved (%). To rebuild only the table:

```bash
python -m framediff.paper_matrix export \
  --root runs/paper-pilot10-seed42-n2-v1 \
  --out runs/paper-pilot10-seed42-n2-v1/table
```

### Initial 성공률과 성공 샘플 기준 품질 보고

논문 표의 마지막 열은 `Success / Improved (%)`입니다.
- Initial: 전체 생성 시도 중 성공한 비율. `prepared.jsonl`의 Initial 실패 상태로 판정하며, 블록 추출 경고나 점수 0은 생성 실패로 취급하지 않습니다.
- Position, CLIP, Box IoU: 해당 모델·시드에서 Initial 생성에 성공한 동일한 샘플 집합으로 모든 방법을 비교합니다.
- 복원 방법의 Improved: 위 집합에서 Initial보다 Box IoU가 증가한 비율입니다. 복원 실패도 마지막 유효 구현의 점수로 포함합니다.
- Initial 성공이 0개면 성공률은 0%, 품질과 개선율은 `--`입니다. 평가기 오류로 점수가 누락되면 성공 샘플을 조용히 제외하지 않고 완전성 검사에서 오류를 냅니다.

`initial-seed-summary.json`의 `initial_coverage_by_seed`에 전체/성공/실패 개수,
각 방법의 `n_pages_by_seed`에 품질 평가 분모가 기록됩니다. 최종 `table/coverage.json`에도 모델·데이터셋·시드별 개수가 저장됩니다.
시드 평균과 표준편차는 시드별 점수에서 계산합니다. 모델마다 성공 집합이 다를 수 있으므로 성공률과 함께 해석해야 합니다.

저장된 평가 결과가 있으면 생성·렌더링 없이 보고서만 다시 만들 수 있습니다. 아래는 pilot10의 WebUI 예입니다(완료된 모델만 aliases에 지정).

```bash
python -m framediff.paper_suite \
  --root runs/paper-pilot10-seed42-n2-v1/webui \
  --aliases llava-next-7b --representative llava-next-7b \
  --out runs/paper-pilot10-seed42-n2-v1/webui/paper \
  --initial-seeds 42 --revision-rounds 2 --ui2coden-rounds 2 \
  --include-ui2coden --skip-ablation --require-complete-metrics
```

같은 방식으로 각 데이터셋·모델 보고서를 갱신한 뒤 전체 결과가 있으면:

```bash
python -m framediff.paper_matrix export \
  --root runs/paper-pilot10-seed42-n2-v1 \
  --out runs/paper-pilot10-seed42-n2-v1/table
```

기존의 실패 포함 평균 보고서는 새 기준과 혼합하지 않도록 재집계를 요구합니다. 원래 metrics.jsonl 및 생성 결과는 변경하지 않습니다.

#!/usr/bin/env bash
# Preserve prior drafts; write updated UI2CodeN experiments separately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
old=${1:?Usage: rerun_pilot_ui2coden.sh PREVIOUS_PILOT_ROOT NEW_OUTPUT_ROOT}
out=${2:?Supply a separate output directory}
[[ "$old" != "$out" ]] || { echo 'Use a separate output directory' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONUNBUFFERED=1 VLM_DEVICE_MAP=balanced VLM_GPU_MEMORY_GIB=20
export UI2CODEN_PRECISION=bf16 PROTOCOL=simple
export MAX_NEW_TOKENS=16384 MAX_CONTEXT=0 ROUNDS=2 LIMIT=0
export EVALUATION_PYTHON="$(command -v python)"
export OFFICIAL_REPO=${OFFICIAL_REPO:-$PWD/external/design2code-official}
export PAPER_REFERENCE_CACHE=${PAPER_REFERENCE_CACHE:-$PWD/.cache/paper-reference-renders}
export INITIAL_SEEDS=42 INITIAL_TEMPERATURE=auto REVISION_ROUNDS=2 PAGE_LIMIT=0
export RUN_ABLATION=0 RUN_UI2CODEN=1 PAPER_DIAGNOSTICS=0
export RUN_TUIDE=${RUN_TUIDE:-$(python - "$old" <<'PY'
from pathlib import Path
import sys
print(int(any(Path(sys.argv[1]).glob('*/llava-next-7b/seeds/42/repair/config.json'))))
PY
)}
train=${TRAIN_RUN:-$PWD/runs/webui-full-dom-screened-7k-v1/7000}
export DETECTOR_CHECKPOINT=${DETECTOR_CHECKPOINT:-$train/detector/best.pt}
export ABSTRACT_CHECKPOINT=${ABSTRACT_CHECKPOINT:-$train/tuide-s-abstract/replacement/abstract-stage2/best.pt}
export REPEATS=1 REPAIR_STEPS=20
export UI2CODEN_MAX_NEW_TOKENS=16384 UI2CODEN_MAX_CONTEXT=0
aliases=llava-next-7b,glm4v-9b-thinking,websight-8b,ui2coden
datasets=()
found=0
for dataset in webui design2code design2code-hard; do
  input="$old/inputs/$dataset.jsonl"
  [[ -f "$input" ]] || continue
  found=1
  datasets+=("$dataset")
  kind=design2code
  [[ "$dataset" == webui ]] && kind=webui
  mkdir -p "$out/inputs"
  if [[ -f "$out/inputs/$dataset.jsonl" ]]; then
    cmp "$input" "$out/inputs/$dataset.jsonl"
  else
    cp "$input" "$out/inputs/$dataset.jsonl"
  fi
  # Finish polishing saved drafts first. Never invoke web-prepare on old runs.
  for alias in llava-next-7b glm4v-9b-thinking websight-8b; do
    manifest="$old/$dataset/$alias/seeds/42/prepare/prepared.jsonl"
    if [[ ! -f "$manifest" ]]; then
      case "$alias" in
        llava-next-7b) spec='llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|bf16';;
        websight-8b) spec='websight-8b|websight|HuggingFaceM4/VLM_WebSight_finetuned|bf16';;
        design2code-18b) spec='design2code-18b|design2code-sat|SALT-NLP/Design2Code-18B-v0|bf16';;
        gemma3-12b) spec='gemma3-12b|hf|google/gemma-3-12b-it|bf16';;
        pixtral-12b) spec='pixtral-12b|hf|mistral-experimental/pixtral-12b|bf16';;
        glm4v-9b-thinking) spec='glm4v-9b-thinking|hf|zai-org/GLM-4.1V-9B-Thinking|bf16';;
        qwen3vl8b) spec='qwen3vl8b|qwen|Qwen/Qwen3-VL-8B-Instruct|bf16';;
      esac
      echo "Generating missing model group: $dataset/$alias"
      VLM_SPECS="$spec" WEB_DATASET="$kind" PAGE_MANIFEST="$input" \
        bash scripts/run_paper_experiments.sh . "$out/$dataset" "$train"
      continue
    fi
    # A partial interrupted manifest is not the full pilot cohort.
    python - "$input" "$manifest" <<'PY'
import json,sys
def ids(path):
    return [str(json.loads(line)['id']) for line in open(path) if line.strip()]
expected,actual=map(ids,sys.argv[1:])
if len(actual)!=len(expected) or set(actual)!=set(expected):
    raise SystemExit(f'Incomplete/mismatched saved Initial manifest: {sys.argv[2]}')
PY
    # Read-only links preserve all previous method results and their config/hash
    # provenance. No generation/evaluation command writes into these links.
    python - "$old/$dataset/$alias" "$out/$dataset/$alias" "$RUN_TUIDE" <<'PY'
from pathlib import Path
import sys
source,dest=map(lambda p:Path(p).resolve(),sys.argv[1:3])
names=['seeds/42/prepare','seeds/42/evaluation']
if sys.argv[3]=='1':names.append('seeds/42/repair')
for name in names:
    src=source/name; dst=dest/name
    if not src.is_dir():raise SystemExit(f'Missing completed results: {src}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.resolve()!=src:raise SystemExit(f'Conflicting output: {dst}')
    else:dst.symlink_to(src,target_is_directory=True)
PY
    GPU="$CUDA_VISIBLE_DEVICES" bash baselines/ui2coden/run_4090.sh \
      "$manifest" "$out/$dataset/$alias/seeds/42/ui2coden"
  done
  # UI2CodeN's own draft uses a fresh run and the original fixed pilot inputs.
  VLM_SPECS='ui2coden|hf|zai-org/UI2Code_N|bf16' WEB_DATASET="$kind" PAGE_MANIFEST="$input" \
    bash scripts/run_paper_experiments.sh . "$out/$dataset" "$train"
  suite=(--root "$out/$dataset" --aliases "$aliases" --representative llava-next-7b
    --out "$out/$dataset/paper" --skip-ablation --include-ui2coden --ui2coden-rounds 2
    --revision-rounds 2 --initial-seeds 42 --require-complete-metrics)
  [[ "$RUN_TUIDE" == 0 ]] && suite+=(--skip-tuide)
  python -m framediff.paper_suite "${suite[@]}"
done
[[ "$found" == 1 ]] || { echo "No pilot inputs under $old/inputs" >&2; exit 2; }
export_options=(--root "$out" --out "$out/table" --datasets "${datasets[@]}" --revision-rounds 2)
[[ "$RUN_TUIDE" == 0 ]] && export_options+=(--skip-tuide)
python -m framediff.paper_matrix export "${export_options[@]}"
echo "Finished. Combined table: $out/table/vlm-quality.tex"

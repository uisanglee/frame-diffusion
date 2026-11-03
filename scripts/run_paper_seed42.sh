#!/usr/bin/env bash
# One initial VLM, seed 42, all three datasets. Run independently per GPU.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash scripts/run_paper_seed42.sh {qwen|llava}' >&2
  exit 2
fi
case "$1" in
  qwen)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    # BF16 weights and the dynamic KV cache remain on GPU.
    export VLM_SPECS='qwen3vl8b|qwen|Qwen/Qwen3-VL-8B-Instruct|bf16'
    output=runs/paper-qwen3vl8b-bf16-seed42-16k-gpu-kv ;;
  llava)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
    export VLM_SPECS='llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|4bit'
    output=runs/paper-llava-next-7b-seed42-16k-compact-v1-temp02 ;;
  *) echo 'Model must be qwen or llava' >&2; exit 2 ;;
esac
[[ -f .venv/bin/activate ]] || { echo 'Missing .venv/bin/activate' >&2; exit 2; }
source .venv/bin/activate
mkdir -p log

export WEBUI_TEST="$repo_root/data/webui-full-dom-screened-7k-v1/7000/css/pages-test.jsonl"
export D2C_ROOT="$repo_root/data/raw/design2code/testset_final"
export D2C_HARD_ROOT="$repo_root/data/raw/design2code-hard"
export TRAIN_RUN="$repo_root/runs/webui-full-dom-screened-7k-v1/7000"
export OFFICIAL_REPO="$repo_root/external/design2code-official"
export DETECTOR_CHECKPOINT="$TRAIN_RUN/detector/best.pt"
export ABSTRACT_CHECKPOINT="$TRAIN_RUN/tuide-s-abstract/replacement/abstract-stage2/best.pt"
export PAPER_REFERENCE_CACHE="$repo_root/.cache/paper-reference-renders"
export PYTHONUNBUFFERED=1 INITIAL_SEEDS=42 INITIAL_TEMPERATURE=auto
export MAX_NEW_TOKENS=16384 PAGE_LIMIT=0 REPAIR_STEPS=20
export REVISION_ROUNDS=${REVISION_ROUNDS:-3}
[[ "$REVISION_ROUNDS" =~ ^[1-9][0-9]*$ ]] || { echo "REVISION_ROUNDS must be positive" >&2; exit 2; }
[[ "$REVISION_ROUNDS" == 3 ]] || output="$output-n$REVISION_ROUNDS"
if [[ ${VLM_DEVICE_MAP:-single} == balanced ]]; then
  # Keep the existing single-GPU runs untouched when switching deployment.
  [[ "$1" == qwen ]] || { echo 'Use the BF16 two-GPU pilot for the other models' >&2; exit 2; }
  export UI2CODEN_PRECISION=bf16
  output="$output-bf16-2gpu"
fi
export RUN_UI2CODEN=1 RUN_ABLATION=0 REPEATS=1

echo "Initial model=$1, GPU=$CUDA_VISIBLE_DEVICES, seed=42, output=$output"
WEB_DATASET=webui PAGE_MANIFEST="$WEBUI_TEST" \
  bash scripts/run_paper_experiments.sh . "$output/webui" "$TRAIN_RUN"
WEB_DATASET=design2code PAGE_MANIFEST='' \
  bash scripts/run_paper_experiments.sh "$D2C_ROOT" "$output/design2code" "$TRAIN_RUN"
WEB_DATASET=design2code PAGE_MANIFEST='' \
  bash scripts/run_paper_experiments.sh "$D2C_HARD_ROOT" "$output/design2code-hard" "$TRAIN_RUN"

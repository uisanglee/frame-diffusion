#!/usr/bin/env bash
# Fixed paper matrix: Qwen3-VL-8B BF16, Gemma-3-12B 4-bit, Pixtral-12B 4-bit.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_paper_4090_vlms.sh D2C_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2
  exit 2
fi
export VLM_SPECS=${VLM_SPECS:-'qwen3vl8b|qwen|Qwen/Qwen3-VL-8B-Instruct|bf16,gemma3-12b|hf|google/gemma-3-12b-it|4bit,pixtral-12b|hf|mistral-community/pixtral-12b|4bit'}
export ABLATION_VLM_ALIAS=${ABLATION_VLM_ALIAS:-qwen3vl8b}
exec bash scripts/run_paper_experiments.sh "$1" "$2" "$3"

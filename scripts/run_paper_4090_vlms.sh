#!/usr/bin/env bash
# Fixed paper matrix: Qwen uses BF16 with GPU KV cache; other VLMs use NF4.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_paper_4090_vlms.sh D2C_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2
  exit 2
fi
# Execution priority, not a measured speed ranking: lightweight first, Qwen last.
export VLM_SPECS=${VLM_SPECS:-'llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|4bit,gemma3-12b|hf|google/gemma-3-12b-it|4bit,pixtral-12b|hf|mistral-experimental/pixtral-12b|4bit,ui2coden|hf|zai-org/UI2Code_N|4bit,glm4v-9b-thinking|hf|zai-org/GLM-4.1V-9B-Thinking|4bit,qwen3vl8b|qwen|Qwen/Qwen3-VL-8B-Instruct|bf16'}
export ABLATION_VLM_ALIAS=${ABLATION_VLM_ALIAS:-qwen3vl8b}
export RUN_UI2CODEN=${RUN_UI2CODEN:-1}
exec bash scripts/run_paper_experiments.sh "$1" "$2" "$3"

#!/usr/bin/env bash
# Official five-model matrix; BF16 execution requires two GPUs.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_paper_4090_vlms.sh D2C_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2
  exit 2
fi
# Dedicated screenshot models use their official image-only generation protocol.
export VLM_SPECS=${VLM_SPECS:-'llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|bf16,glm4v-9b-thinking|hf|zai-org/GLM-4.1V-9B-Thinking|bf16,design2code-18b|design2code-sat|SALT-NLP/Design2Code-18B-v0|bf16,ui2coden|hf|zai-org/UI2Code_N|bf16'}
export ABLATION_VLM_ALIAS=${ABLATION_VLM_ALIAS:-llava-next-7b}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLM_DEVICE_MAP=${VLM_DEVICE_MAP:-balanced}
export UI2CODEN_PRECISION=${UI2CODEN_PRECISION:-bf16}
export RUN_UI2CODEN=${RUN_UI2CODEN:-1}
exec bash scripts/run_paper_experiments.sh "$1" "$2" "$3"

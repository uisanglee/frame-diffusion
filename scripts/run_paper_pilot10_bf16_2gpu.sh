#!/usr/bin/env bash
# Same 30 pilot inputs and repair protocol, BF16 VLMs on two GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -ra selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#selected_gpus[@]} == 2 && "${selected_gpus[0]}" != "${selected_gpus[1]}" ]] || {
  echo 'Select two different GPUs, for example CUDA_VISIBLE_DEVICES=0,1' >&2; exit 2;
}
export VLM_DEVICE_MAP=balanced
export VLM_GPU_MEMORY_GIB=${VLM_GPU_MEMORY_GIB:-20}
export UI2CODEN_PRECISION=bf16
export UI2CODEN_MAX_NEW_TOKENS=16384 UI2CODEN_MAX_CONTEXT=0
export PILOT_OUTPUT=${PILOT_OUTPUT:-runs/paper-pilot10-seed42-n2-bf16-2gpu-v1}
export VLM_SPECS=${VLM_SPECS:-'llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|bf16,gemma3-12b|hf|google/gemma-3-12b-it|bf16,pixtral-12b|hf|mistral-experimental/pixtral-12b|bf16,ui2coden|hf|zai-org/UI2Code_N|bf16,glm4v-9b-thinking|hf|zai-org/GLM-4.1V-9B-Thinking|bf16,qwen3vl8b|qwen|Qwen/Qwen3-VL-8B-Instruct|bf16'}
echo "BF16 two-GPU pilot: GPUs=$CUDA_VISIBLE_DEVICES, weight budget=$VLM_GPU_MEMORY_GIB GiB/GPU, output=$PILOT_OUTPUT"
exec bash scripts/run_paper_pilot10.sh

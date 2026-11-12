#!/usr/bin/env bash
# Two independent two-GPU model queues; merge only after both complete.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
output=${D2C_PARALLEL_OUTPUT:-runs/d2c-vlm-parallel-seed42-n2-v1}
mkdir -p "$output"
export D2C_BASELINE_PAGES=${D2C_BASELINE_PAGES:-0}
echo "GPU 0,1: LLaVA + GLM; GPU 2,4: WebSight + UI2CodeN"
echo "Output: $output; pages/dataset=$D2C_BASELINE_PAGES (0=all)"
CUDA_VISIBLE_DEVICES=0,1 D2C_BASELINE_OUTPUT="$output/gpu01" \
  VLM_SPECS='llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|bf16,glm4v-9b-thinking|hf|zai-org/GLM-4.1V-9B-Thinking|bf16' \
  bash scripts/run_d2c_vlm_baselines.sh >> "$output/gpu01.log" 2>&1 &
pid01=$!
CUDA_VISIBLE_DEVICES=2,4 D2C_BASELINE_OUTPUT="$output/gpu24" \
  VLM_SPECS='websight-8b|websight|HuggingFaceM4/VLM_WebSight_finetuned|bf16,ui2coden|hf|zai-org/UI2Code_N|bf16' \
  bash scripts/run_d2c_vlm_baselines.sh >> "$output/gpu24.log" 2>&1 &
pid24=$!
printf '%s\n' "$pid01" > "$output/gpu01.pid"
printf '%s\n' "$pid24" > "$output/gpu24.pid"
status01=0; status24=0
wait "$pid01" || status01=$?
wait "$pid24" || status24=$?
if [[ "$status01" != 0 || "$status24" != 0 ]]; then
  echo "Queue failure: gpu01=$status01 gpu24=$status24; inspect group logs. Rerun the same command to resume." >&2
  exit 1
fi
python -m framediff.d2c_parallel_merge --root "$output"

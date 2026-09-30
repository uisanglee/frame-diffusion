#!/usr/bin/env bash
# Separate processes release planner VRAM before loading the repair policy.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_plan_web.sh DATASET_ROOT RUN_DIR PLAN_CHECKPOINT' >&2
  exit 2
fi
dataset_root=$1
run_dir=$2
checkpoint=$3
options=(--root "$dataset_root")
if [[ -n "${INPUT_MANIFEST:-}" ]]; then options=(--manifest "$INPUT_MANIFEST"); fi
options+=(--dataset "${WEB_DATASET:-design2code}" --webui-view "${WEBUI_VIEW:-default_1280-720}"
  --backend "${VLM_BACKEND:-qwen}" --model "${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
  --endpoint "${VLM_ENDPOINT:-http://localhost:8000/v1/chat/completions}"
  --repair-conditioning plan --initial-mode direct --rounds "${REVISION_ROUNDS:-0}"
  --vlm-retries "${VLM_RETRIES:-0}" --limit "${PAGE_LIMIT:-5}"
  --max-new-tokens "${MAX_NEW_TOKENS:-16384}" --max-pixels "${MAX_PIXELS:-1048576}")
if [[ "${VLM_BACKEND:-qwen}" == qwen ]]; then options+=(--four-bit); fi
python -m framediff web-prepare "${options[@]}" --out "$run_dir/prepare" --resume
python -m framediff web-repair --data "$run_dir/prepare/prepared.jsonl" \
  --checkpoint "$checkpoint" --out "$run_dir/repair" --device "${REPAIR_DEVICE:-cuda}" \
  --methods coordinate-plan,model-plan --steps "${REPAIR_STEPS:-10}" \
  --beam "${BEAM:-2}" --topk "${TOPK:-32}" --budget "${BUDGET:-320}" --resume
evaluation=()
if [[ "${WEB_DATASET:-design2code}" != webui && -n "${OFFICIAL_REPO:-}" ]]; then
  evaluation+=(--official-repo "$OFFICIAL_REPO")
fi
python -m framediff web-evaluate --data "$run_dir/repair/results.jsonl" \
  --out "$run_dir/evaluation" "${evaluation[@]}" --resume

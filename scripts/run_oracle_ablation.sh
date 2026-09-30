#!/usr/bin/env bash
# Controlled Design2Code 2x2: VLM/oracle target boxes x coordinate/FrameDiff repair.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_oracle_ablation.sh DATASET_ROOT OUTPUT_ROOT CHECKPOINT' >&2
  exit 2
fi
dataset_root=$1
output_root=$2
checkpoint_path=$3
prepare_options=(--backend "${VLM_BACKEND:-qwen}" --model "${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
  --endpoint "${VLM_ENDPOINT:-http://localhost:8000/v1/chat/completions}"
  --limit "${PAGE_LIMIT:-0}" --corruptions "${CORRUPTIONS:-4}" --vlm-retries "${VLM_RETRIES:-2}"
  --max-new-tokens "${MAX_NEW_TOKENS:-16384}" --max-pixels "${MAX_PIXELS:-1048576}"
  --max-nodes "${MAX_NODES:-64}" --seed "${SEED:-42}")
if [[ "${VLM_BACKEND:-qwen}" == qwen ]]; then prepare_options+=(--four-bit); fi
python -m framediff web-oracle-prepare --root "$dataset_root" --out "$output_root/prepare" \
  "${prepare_options[@]}" --resume
python -m framediff web-repair --data "$output_root/prepare/prepared.jsonl" \
  --checkpoint "$checkpoint_path" --out "$output_root/repair" --device "${REPAIR_DEVICE:-cuda}" \
  --methods coordinate-vlm,model-vlm,coordinate-oracle,model-oracle \
  --feedback-render "${FEEDBACK_RENDER:-frames}" --steps 10 --beam 2 --topk 8 --budget 160 --resume
evaluate_options=()
if [[ -n "${OFFICIAL_REPO:-}" ]]; then evaluate_options+=(--official-repo "$OFFICIAL_REPO"); fi
python -m framediff web-evaluate --data "$output_root/repair/results.jsonl" \
  --out "$output_root/evaluation" "${evaluate_options[@]}" --resume

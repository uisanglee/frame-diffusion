#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then echo 'Usage: bash scripts/run_visual_web.sh DATASET_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2; exit 2; fi
dataset_root=$1
output_root=$2
train_root=$3
source_options=(--root "$dataset_root" --dataset "${WEB_DATASET:-design2code}" --webui-view "${WEBUI_VIEW:-default_1280-720}")
if [[ -n "${PAGE_MANIFEST:-}" ]]; then source_options=(--manifest "$PAGE_MANIFEST"); fi
vlm_options=(--backend "${VLM_BACKEND:-qwen}" --model "${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
  --endpoint "${VLM_ENDPOINT:-http://localhost:8000/v1/chat/completions}")
if [[ "${VLM_BACKEND:-qwen}" == qwen ]]; then vlm_options+=(--four-bit); fi
python -m framediff web-prepare "${source_options[@]}" "${vlm_options[@]}" \
  --out "$output_root/prepare" --repair-conditioning visual --rounds 0 \
  --limit "${PAGE_LIMIT:-0}" --max-nodes 127 --vlm-retries 2 \
  --max-new-tokens "${MAX_NEW_TOKENS:-16384}" --max-pixels "${MAX_PIXELS:-1048576}" --resume
extra=()
if [[ "${ORACLE_ABLATION:-0}" == 1 ]]; then extra+=(--oracle-ablation); fi
python -m framediff visual-evaluate --data "$output_root/prepare/prepared.jsonl" \
  --raw-checkpoint "$train_root/policy-raw/best.pt" --abstract-checkpoint "$train_root/policy-abstract/best.pt" \
  --detector-checkpoint "$train_root/detector/best.pt" --out "$output_root/repair" \
  --device "${DEVICE:-cuda}" --steps "${REPAIR_STEPS:-20}" --repeats "${REPEATS:-3}" \
  --time-budget "${TIME_BUDGET:-0}" --limit "${PAGE_LIMIT:-0}" "${extra[@]}" --resume
metrics=()
if [[ -n "${OFFICIAL_REPO:-}" && "${WEB_DATASET:-design2code}" != webui ]]; then metrics+=(--official-repo "$OFFICIAL_REPO"); fi
python -m framediff web-evaluate --data "$output_root/repair/results.jsonl" \
  --out "$output_root/evaluation" "${metrics[@]}" --resume

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
revision_options=()
case "${SELF_REVISION_PROTOCOL:-none}" in
  none)
    revision_options=(--rounds 0 --initial-mode direct --revision-protocol shared)
    seed=${SEED:-42}; max_new_tokens=${MAX_NEW_TOKENS:-16384}; vlm_retries=${VLM_RETRIES:-2} ;;
  shared)
    revision_options=(--rounds "${REVISION_ROUNDS:-1}" --initial-mode "${INITIAL_MODE:-direct}" --revision-protocol shared)
    seed=${SEED:-42}; max_new_tokens=${MAX_NEW_TOKENS:-16384}; vlm_retries=${VLM_RETRIES:-2} ;;
  design2code)
    revision_options=(--rounds 1 --initial-mode text-augmented --revision-protocol design2code)
    seed=${SEED:-2024}; max_new_tokens=${MAX_NEW_TOKENS:-4096}; vlm_retries=${VLM_RETRIES:-0} ;;
  *) echo 'SELF_REVISION_PROTOCOL must be none, shared, or design2code' >&2; exit 2 ;;
esac
python -m framediff web-prepare "${source_options[@]}" "${vlm_options[@]}" \
  --out "$output_root/prepare" --repair-conditioning visual "${revision_options[@]}" \
  --limit "${PAGE_LIMIT:-0}" --max-nodes 127 --vlm-retries "$vlm_retries" --seed "$seed" \
  --max-new-tokens "$max_new_tokens" --max-pixels "${MAX_PIXELS:-1048576}" --resume
extra=()
if [[ "${ORACLE_ABLATION:-0}" == 1 ]]; then extra+=(--oracle-ablation); fi
python -m framediff visual-evaluate --data "$output_root/prepare/prepared.jsonl" \
  --raw-checkpoint "${RAW_CHECKPOINT:-$train_root/policy-raw/best.pt}" --abstract-checkpoint "${ABSTRACT_CHECKPOINT:-$train_root/policy-abstract/best.pt}" \
  --detector-checkpoint "${DETECTOR_CHECKPOINT:-$train_root/detector/best.pt}" --out "$output_root/repair" \
  --device "${DEVICE:-cuda}" --steps "${REPAIR_STEPS:-20}" --repeats "${REPEATS:-3}" \
  --time-budget "${TIME_BUDGET:-0}" --goal-threshold "${GOAL_THRESHOLD:--1}" \
  --abstract-goal-threshold "${ABSTRACT_GOAL_THRESHOLD:--1}" \
  --limit "${PAGE_LIMIT:-0}" "${extra[@]}" --resume
metrics=()
if [[ -n "${OFFICIAL_REPO:-}" && "${WEB_DATASET:-design2code}" != webui ]]; then metrics+=(--official-repo "$OFFICIAL_REPO"); fi
python -m framediff web-evaluate --data "$output_root/repair/results.jsonl" \
  --out "$output_root/evaluation" "${metrics[@]}" --resume

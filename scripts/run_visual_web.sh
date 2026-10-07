#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then echo 'Usage: bash scripts/run_visual_web.sh DATASET_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2; exit 2; fi
dataset_root=$1
output_root=$2
train_root=$3
source_options=(--root "$dataset_root" --dataset "${WEB_DATASET:-design2code}" --webui-view "${WEBUI_VIEW:-default_1280-720}")
if [[ -n "${PAGE_MANIFEST:-}" ]]; then source_options=(--manifest "$PAGE_MANIFEST"); fi
vlm_options=(--backend "${VLM_BACKEND:-qwen}" --model "${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
  --endpoint "${VLM_ENDPOINT:-http://localhost:8000/v1/chat/completions}"
  --api-key-env "${VLM_API_KEY_ENV:-VLM_API_KEY}")
if [[ -n "${VLM_REASONING_EFFORT:-}" ]]; then
  vlm_options+=(--reasoning-effort "$VLM_REASONING_EFFORT")
fi
case "${VLM_FOUR_BIT:-1}" in
  1) [[ "${VLM_BACKEND:-qwen}" != openai-compatible ]] && vlm_options+=(--four-bit) ;;
  0) ;;
  *) echo 'VLM_FOUR_BIT must be 0 or 1' >&2; exit 2 ;;
esac
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
prepared="$output_root/prepare/prepared.jsonl"
if [[ "${ABSTRACT_VLM_REVISION:-0}" == 1 ]]; then
  [[ "${SELF_REVISION_PROTOCOL:-none}" == design2code ]] || { echo 'ABSTRACT_VLM_REVISION=1 requires SELF_REVISION_PROTOCOL=design2code' >&2; exit 2; }
  detector_checkpoint=${DETECTOR_CHECKPOINT:-$train_root/detector/best.pt}
  python -m framediff web-cache-abstractions --data "$prepared" --detector-checkpoint "$detector_checkpoint" \
    --out "$output_root/abstract-cache" --device "${DEVICE:-cuda}" --resume
  python -m framediff web-abstract-self-revision --data "$output_root/abstract-cache/data.jsonl" \
    --out "$output_root/abstract-vlm" "${vlm_options[@]}" --seed "$seed" --vlm-retries "$vlm_retries" \
    --max-new-tokens "$max_new_tokens" --max-pixels "${MAX_PIXELS:-1048576}" --resume
  prepared="$output_root/abstract-vlm/prepared.jsonl"
fi
if [[ "${TUIDE_SCALES:-0}" == 1 ]]; then
  bash scripts/run_tuide_scales.sh "$prepared" "$output_root/scales"
  exit 0
fi
pick_checkpoint() {
  local explicit=$1
  shift
  if [[ -n "$explicit" ]]; then
    [[ -f "$explicit" ]] || { echo "Missing checkpoint: $explicit" >&2; exit 2; }
    printf '%s\n' "$explicit"
    return
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then printf '%s\n' "$candidate"; return; fi
  done
  echo "Could not find any checkpoint candidate: $*" >&2
  exit 2
}
raw_checkpoint=$(pick_checkpoint "${RAW_CHECKPOINT:-}" \
  "$train_root/replacement/raw-stage2/best.pt" "$train_root/policy-raw/best.pt" "$train_root/raw-stage2/best.pt")
abstract_checkpoint=$(pick_checkpoint "${ABSTRACT_CHECKPOINT:-}" \
  "$train_root/replacement/abstract-stage2/best.pt" "$train_root/policy-abstract/best.pt" "$train_root/abstract-stage2/best.pt")
detector_checkpoint=$(pick_checkpoint "${DETECTOR_CHECKPOINT:-}" "$train_root/detector/best.pt")
extra=()
if [[ "${ORACLE_ABLATION:-0}" == 1 ]]; then extra+=(--oracle-ablation); fi
python -m framediff visual-evaluate --data "$prepared" \
  --raw-checkpoint "$raw_checkpoint" --abstract-checkpoint "$abstract_checkpoint" \
  --detector-checkpoint "$detector_checkpoint" --out "$output_root/repair" \
  --device "${DEVICE:-cuda}" --steps "${REPAIR_STEPS:-20}" --repeats "${REPEATS:-3}" \
  --time-budget "${TIME_BUDGET:-0}" --goal-threshold "${GOAL_THRESHOLD:--1}" \
  --abstract-goal-threshold "${ABSTRACT_GOAL_THRESHOLD:--1}" \
  --limit "${PAGE_LIMIT:-0}" "${extra[@]}" --resume
metrics=()
if [[ -n "${OFFICIAL_REPO:-}" && "${WEB_DATASET:-design2code}" != webui ]]; then metrics+=(--official-repo "$OFFICIAL_REPO"); fi
python -m framediff web-evaluate --data "$output_root/repair/results.jsonl" \
  --out "$output_root/evaluation" "${metrics[@]}" --resume

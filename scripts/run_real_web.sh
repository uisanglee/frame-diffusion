#!/usr/bin/env bash
# Run each stage in its own process so VLM/denoiser/CLIP do not coexist in VRAM.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_real_web.sh DATASET_ROOT OUTPUT_ROOT CHECKPOINT' >&2
  exit 2
fi
dataset_root=$1
output_root=$2
checkpoint_path=$3
self_revision=${SELF_REVISION_PROTOCOL:-shared}
case "$self_revision" in
  none) rounds=0; initial_mode=direct; revision_protocol=shared ;;
  shared) rounds=${REVISION_ROUNDS:-1}; initial_mode=${INITIAL_MODE:-direct}; revision_protocol=shared ;;
  design2code) rounds=1; initial_mode=text-augmented; revision_protocol=design2code ;;
  *) echo 'SELF_REVISION_PROTOCOL must be none, shared, or design2code' >&2; exit 2 ;;
esac
if [[ "$self_revision" == design2code ]]; then
  seed=${SEED:-2024}; max_new_tokens=${MAX_NEW_TOKENS:-4096}; vlm_retries=${VLM_RETRIES:-0}
else
  seed=${SEED:-42}; max_new_tokens=${MAX_NEW_TOKENS:-16384}; vlm_retries=${VLM_RETRIES:-2}
fi
prepare_options=(--backend "${VLM_BACKEND:-qwen}" --model "${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
  --dataset "${WEB_DATASET:-design2code}" --webui-view "${WEBUI_VIEW:-default_1280-720}"
  --endpoint "${VLM_ENDPOINT:-http://localhost:8000/v1/chat/completions}"
  --vlm-retries "$vlm_retries" --seed "$seed"
  --max-new-tokens "$max_new_tokens" --max-pixels "${MAX_PIXELS:-1048576}"
  --rounds "$rounds" --limit "${PAGE_LIMIT:-0}" --initial-mode "$initial_mode"
  --revision-protocol "$revision_protocol")
if [[ "${VLM_BACKEND:-qwen}" == qwen ]]; then prepare_options+=(--four-bit); fi
python -m framediff web-prepare --root "$dataset_root" --out "$output_root/prepare" \
  "${prepare_options[@]}" --resume
python -m framediff web-repair --data "$output_root/prepare/prepared.jsonl" \
  --checkpoint "$checkpoint_path" --out "$output_root/repair" --device "${REPAIR_DEVICE:-cuda}" \
  --methods "${REPAIR_METHODS:-coordinate-feedback,model-feedback}" \
  --feedback-render "${FEEDBACK_RENDER:-frames}" --steps 10 --beam 2 --topk 8 --budget 160 --resume
evaluate_options=()
if [[ "${WEB_DATASET:-design2code}" != webui && -n "${OFFICIAL_REPO:-}" ]]; then
  evaluate_options+=(--official-repo "$OFFICIAL_REPO")
elif [[ "${WEB_DATASET:-design2code}" == webui && -n "${OFFICIAL_REPO:-}" ]]; then
  echo 'WebUI: OFFICIAL_REPO is not used; evaluating recorded screenshots and AX boxes.'
fi
python -m framediff web-evaluate --data "$output_root/repair/results.jsonl" \
  --out "$output_root/evaluation" "${evaluate_options[@]}" --resume

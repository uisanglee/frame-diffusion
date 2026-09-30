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
prepare_options=(--backend "${VLM_BACKEND:-qwen}" --model "${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
  --dataset "${WEB_DATASET:-design2code}" --webui-view "${WEBUI_VIEW:-default_1280-720}"
  --endpoint "${VLM_ENDPOINT:-http://localhost:8000/v1/chat/completions}"
  --rounds "${REVISION_ROUNDS:-1}" --limit "${PAGE_LIMIT:-0}" --initial-mode "${INITIAL_MODE:-direct}")
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

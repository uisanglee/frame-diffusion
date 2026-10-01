#!/usr/bin/env bash
# Evaluate one WebUI-fine-tuned visual run on controlled WebUI, D2C and D2C-Hard.
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo 'Usage: bash scripts/evaluate_webui_finetune.sh DATA_DIR RUN_DIR D2C_ROOT D2C_HARD_ROOT OUTPUT_ROOT' >&2
  exit 2
fi
data_dir=$1
run_dir=$2
d2c_root=$3
hard_root=$4
output_root=$5
repeats=${REPEATS:-3}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WEB_DATASET=design2code SELF_REVISION_PROTOCOL=none \
PAGE_MANIFEST="$data_dir/rendered/pages-test.jsonl" \
PAGE_LIMIT=${CONTROLLED_LIMIT:-0} REPEATS="$repeats" ORACLE_ABLATION=1 \
bash scripts/run_visual_web.sh unused "$output_root/webui-controlled" "$run_dir"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WEB_DATASET=design2code SELF_REVISION_PROTOCOL=none \
PAGE_LIMIT=${D2C_LIMIT:-0} REPEATS="$repeats" \
bash scripts/run_visual_web.sh "$d2c_root" "$output_root/design2code" "$run_dir"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WEB_DATASET=design2code SELF_REVISION_PROTOCOL=none \
PAGE_LIMIT=${HARD_LIMIT:-0} REPEATS="$repeats" \
bash scripts/run_visual_web.sh "$hard_root" "$output_root/design2code-hard" "$run_dir"

echo "Evaluation complete: $output_root"

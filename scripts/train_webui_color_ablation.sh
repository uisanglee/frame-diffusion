#!/usr/bin/env bash
# Shared completed preprocessing/inspection caches; no rebuilding or deletion.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ $# == 2 ]] || { echo 'Usage: bash scripts/train_webui_color_ablation.sh screenshot|abstract-rgb|occupancy GPU' >&2; exit 2; }
kind=$1
export CUDA_VISIBLE_DEVICES=$2
export PYTHONUNBUFFERED=1
export ONLINE_WORKERS=${ONLINE_WORKERS:-4}
export TUIDE_INSPECTION_WORKERS=${TUIDE_INSPECTION_WORKERS:-2}
export ONLINE_PREFETCH=${ONLINE_PREFETCH:-8} ONLINE_TIMEOUT=${ONLINE_TIMEOUT:-180}
export BATCH_SIZE=${BATCH_SIZE:-2} ACCUMULATION=${ACCUMULATION:-4}
export VAL_SAMPLES=${VAL_SAMPLES:-2000} EVAL_EVERY=${EVAL_EVERY:-500} SEED=${SEED:-42}
export TUIDE_SAVE_DIAGNOSTICS=${TUIDE_SAVE_DIAGNOSTICS:-0}
export TUIDE_SHARED_CACHE=${TUIDE_SHARED_CACHE:-$PWD/data/webui-shared-cache}
run=${TRAIN_RUN:-runs/webui-color-images-v1/7000}
data=${TRAIN_DATA:-data/new/7000/css}
detector=${DETECTOR_CHECKPOINT:-$run/detector/best.pt}
export POLICY_STAGE1_STEPS=15000 POLICY_STAGE2_STEPS=5000
case "$kind" in
  screenshot)
    export POLICY_MODES=screenshot POLICY_STAGES=1 ABSTRACTION_REPRESENTATION=occupancy-boundary
    export POLICY_STAGE1_STEPS=20000 POLICY_SCREENSHOT_STEPS=20000
    output=$run/tuide-s-screenshot ;;
  abstract-rgb)
    export POLICY_MODES=abstract POLICY_STAGES=all ABSTRACTION_REPRESENTATION=abstract-rgb
    output=$run/tuide-s-abstract-rgb ;;
  occupancy)
    export POLICY_MODES=abstract POLICY_STAGES=all ABSTRACTION_REPRESENTATION=occupancy
    output=$run/tuide-s-occupancy-only ;;
  *) echo "Unknown experiment: $kind" >&2; exit 2 ;;
esac
exec bash scripts/train_tuide_scale.sh s "$data" "$detector" "$run/shared" "$output"

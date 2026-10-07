#!/usr/bin/env bash
# Reuse prepared CSS owners, fixed validation and frozen detector cache.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -lt 1 || $# -gt 5 ]]; then
  echo 'Usage: bash scripts/train_tuide_scale.sh s|m|l [DATA DETECTOR SHARED_RUN OUTPUT_RUN]' >&2
  exit 2
fi
scale=$1
case "$scale" in s|m|l) ;; *) echo 'Scale must be s, m, or l' >&2; exit 2;; esac
export POLICY_SCALE=$scale
export CSS_STYLESHEETS=1 ONLINE_CORRUPTION=1 PREPARE_DATA=0 POLICY_HEADS=replacement
export POLICY_MODES=${POLICY_MODES:-abstract}
case "$POLICY_MODES" in abstract|screenshot) ;; *) echo 'Choose POLICY_MODES=abstract or screenshot' >&2; exit 2;; esac
export POLICY_RUN_DIR=${5:-runs/tuide-visible-$scale-$POLICY_MODES}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DEVICE=${DEVICE:-cuda}
export POLICY_STAGE1_STEPS=${POLICY_STAGE1_STEPS:-30000}
export POLICY_STAGE2_STEPS=${POLICY_STAGE2_STEPS:-15000}
export POLICY_LR=${POLICY_LR:-0.0001}
export POLICY_STAGE2_LR=${POLICY_STAGE2_LR:-0.00001}
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUMULATION=${ACCUMULATION:-4}
export VAL_SAMPLES=${VAL_SAMPLES:-2000}
export EVAL_EVERY=${EVAL_EVERY:-500}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-10}
echo "Training TUIDE-$scale ($POLICY_MODES), visible GPU=$CUDA_VISIBLE_DEVICES -> $POLICY_RUN_DIR"
exec bash scripts/train_numeric_policy.sh \
  "${2:-data/webui-css-owners-v5-visible}" \
  "${3:-runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt}" \
  "${4:-runs/css-owners-v5-visible-shared}"

#!/usr/bin/env bash
# Reuse prepared CSS owners, fixed validation and frozen detector cache.
# Lazy normalization inherits TUIDE_SAVE_DIAGNOSTICS (default 0; 1 for debugging).
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
suffix=""
[[ "${PARENT_FIRST:-0}" == 1 ]] && suffix=-parent-first
export ABSTRACTION_REPRESENTATION=${ABSTRACTION_REPRESENTATION:-occupancy-boundary}
case "$ABSTRACTION_REPRESENTATION" in
  occupancy-boundary) ;;
  abstract-rgb|occupancy|boundary)
    [[ "$POLICY_MODES" == abstract ]] || { echo 'Representation ablation requires abstract mode' >&2; exit 2; }
    suffix="$suffix-$ABSTRACTION_REPRESENTATION" ;;
  *) echo 'Invalid ABSTRACTION_REPRESENTATION' >&2; exit 2;;
esac
export POLICY_RUN_DIR=${5:-runs/tuide-safe-size-margin-$scale-$POLICY_MODES$suffix}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DEVICE=${DEVICE:-cuda}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export TUIDE_SAVE_DIAGNOSTICS=${TUIDE_SAVE_DIAGNOSTICS:-0}
export TUIDE_SHARED_CACHE="${TUIDE_SHARED_CACHE:-$repo_root/data/webui-shared-cache}"
export ONLINE_WORKERS=${ONLINE_WORKERS:-2}
export ONLINE_PREFETCH=${ONLINE_PREFETCH:-4}
export ONLINE_TIMEOUT=${ONLINE_TIMEOUT:-180}
export PARENT_FIRST=${PARENT_FIRST:-0}
export SEED=${SEED:-42}
export POLICY_STAGE1_STEPS=${POLICY_STAGE1_STEPS:-30000}
export POLICY_STAGE2_STEPS=${POLICY_STAGE2_STEPS:-15000}
export POLICY_LR=${POLICY_LR:-0.0001}
export POLICY_STAGE2_LR=${POLICY_STAGE2_LR:-0.00001}
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUMULATION=${ACCUMULATION:-4}
export VAL_SAMPLES=${VAL_SAMPLES:-2000}
export EVAL_EVERY=${EVAL_EVERY:-500}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-10}
export EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0}
echo "Training TUIDE-$scale ($POLICY_MODES), visible GPU=$CUDA_VISIBLE_DEVICES -> $POLICY_RUN_DIR"
exec bash scripts/train_numeric_policy.sh \
  "${2:-data/webui-safe-size-margin-v1}" \
  "${3:-runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt}" \
  "${4:-runs/webui-safe-size-margin-v1-shared}"

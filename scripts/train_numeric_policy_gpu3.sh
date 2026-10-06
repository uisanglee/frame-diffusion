#!/usr/bin/env bash
# Physical GPU 3: semantic abstract-feedback policy, stages 1 and 2.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ $# -gt 4 ]]; then
  echo 'Usage: bash scripts/train_numeric_policy_gpu3.sh [RENDERED] [DETECTOR_BEST_PT] [SHARED_RUN] [POLICY_RUN]' >&2
  exit 2
fi

rendered=${1:-data/webui-10k-v7-multipositive/rendered}
detector=${2:-runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt}
shared_run=${3:-runs/numeric-policy-v7-shared-fresh}
policy_run=${4:-runs/numeric-policy-v7-abstract}

for required in subset/policy-train.jsonl subset/policy-val.jsonl \
                predicted-train/data.jsonl predicted-val/data.jsonl; do
  [[ -s "$shared_run/$required" ]] || {
    echo "Missing prepared input: $shared_run/$required" >&2
    echo 'Run scripts/prepare_numeric_policy_v7.sh first.' >&2
    exit 2
  }
done
[[ -f "$detector" ]] || { echo "Missing detector checkpoint: $detector" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=3
export DEVICE=cuda
export PREPARE_DATA=0
export POLICY_HEADS=autoregressive
export POLICY_MODES=abstract
export POLICY_RUN_DIR=$policy_run
export POLICY_STAGE1_STEPS=${POLICY_STAGE1_STEPS:-30000}
export POLICY_STAGE2_STEPS=${POLICY_STAGE2_STEPS:-15000}
export POLICY_LR=${POLICY_LR:-0.0001}
export POLICY_STAGE2_LR=${POLICY_STAGE2_LR:-0.00001}
export BATCH_SIZE=${BATCH_SIZE:-2}
export ACCUMULATION=${ACCUMULATION:-4}
export VAL_SAMPLES=${VAL_SAMPLES:-2000}
export EVAL_EVERY=${EVAL_EVERY:-500}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-10}

echo "Training abstract policy on physical GPU 3 -> $policy_run"
exec bash scripts/train_numeric_policy.sh "$rendered" "$detector" "$shared_run"


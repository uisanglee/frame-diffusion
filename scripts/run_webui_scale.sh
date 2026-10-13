#!/usr/bin/env bash
# Independent detector + policy training and own/common held-out evaluation.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# != 3 ]]; then
  echo 'Usage: bash scripts/run_webui_scale.sh SIZE DATA_ROOT RUN_ROOT' >&2
  exit 2
fi
size=$1
data=$2
runs=$3
[[ "$size" =~ ^[0-9]+$ ]] || { echo 'SIZE must be numeric' >&2; exit 2; }
base="$data/$size"
output="$runs/$size"
[[ -s "$base/coverage.json" ]] || { echo "Complete prepare_webui_scales.sh first: $base/coverage.json missing" >&2; exit 2; }
common=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["common_test_scale"])' "$data/split-report.json")
stage=${STAGE:-all}
case "$stage" in all|detector|policy|evaluate) ;; *) echo 'STAGE must be all, detector, policy, or evaluate' >&2; exit 2;; esac
device=${DEVICE:-cuda}
export VAL_SAMPLES=${VAL_SAMPLES:-0} # All available validation views; 0 is unlimited.
detector="$output/detector/best.pt"
if [[ "$stage" == all || "$stage" == detector ]]; then
  options=()
  if [[ -f "$output/detector/last.pt" ]]; then options=(--resume "$output/detector/last.pt"); fi
  python -m framediff visual-train-detector \
    --train "$base/rendered/detector-train.jsonl" --val "$base/rendered/detector-val.jsonl" \
    --out "$output/detector" --device "$device" --seed "${SEED:-42}" \
    --steps "${DETECTOR_STEPS:-20000}" --lr "${DETECTOR_LR:-0.0001}" \
    --batch-size "${BATCH_SIZE:-2}" --accumulation "${ACCUMULATION:-4}" \
    --val-samples "${VAL_SAMPLES:-2000}" --eval-every "${EVAL_EVERY:-500}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA:-0}" "${options[@]}"
  for population in own common; do
    testbase=$base
    [[ "$population" == common ]] && testbase="$data/$common"
    python -m framediff visual-evaluate-detector --data "$testbase/rendered/detector-test.jsonl" \
      --checkpoint "$detector" --out "$output/detector-test-$population.json" \
      --device "$device" --threshold "${DETECTOR_THRESHOLD:-0.4}"
  done
fi
if [[ "$stage" == all || "$stage" == policy ]]; then
  for split in train val; do
    python -m framediff visual-cache-targets --data "$base/css/policy-$split.jsonl" \
      --checkpoint "$detector" --out "$output/shared/predicted-$split" \
      --device "$device" --size "${IMAGE_SIZE:-384}" --resume
  done
  for mode in ${POLICY_MODES:-abstract screenshot}; do
    POLICY_MODES="$mode" bash scripts/train_tuide_scale.sh "${POLICY_SCALE:-s}" \
      "$base/css" "$detector" "$output/shared" "$output/tuide-${POLICY_SCALE:-s}-$mode"
  done
fi
if [[ "$stage" == all || "$stage" == evaluate ]]; then
  for mode in ${POLICY_MODES:-abstract screenshot}; do
    label=abstract;flag=--abstract-checkpoint
    if [[ "$mode" == screenshot ]]; then label=raw;flag=--raw-checkpoint; fi
    checkpoint="$output/tuide-${POLICY_SCALE:-s}-$mode/replacement/$label-stage2/best.pt"
    for population in own common; do
      testbase=$base
      [[ "$population" == common ]] && testbase="$data/$common"
      python -m framediff visual-test-rollout --data "$testbase/css/policy-test.jsonl" \
        --pages "$testbase/css/pages-test.jsonl" "$flag" "$checkpoint" \
        --detector-checkpoint "$detector" --out "$output/eval-${POLICY_SCALE:-s}-$mode-$population" \
        --device "$device" --steps "${REPAIR_STEPS:-20}" --repeats "${REPEATS:-1}" --resume
    done
  done
fi
echo "Completed $stage for $size. Summarize after all scale jobs finish:"
echo "python -m framediff visual-webui-scales-summary --data $data --runs $runs"

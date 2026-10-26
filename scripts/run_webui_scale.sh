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
# Shared experiment defaults; caller-provided environment values take precedence.
# Export these so the nested policy scripts receive the same configuration.
export STAGE="${STAGE:-policy}"
export DEVICE="${DEVICE:-cuda}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TUIDE_SAVE_DIAGNOSTICS="${TUIDE_SAVE_DIAGNOSTICS:-0}"
export TUIDE_SHARED_CACHE="${TUIDE_SHARED_CACHE:-$repo_root/data/webui-shared-cache}"
export POLICY_SCALE="${POLICY_SCALE:-s}"
export POLICY_MODES="${POLICY_MODES:-abstract screenshot}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export ACCUMULATION="${ACCUMULATION:-4}"
export DETECTOR_STEPS="${DETECTOR_STEPS:-20000}"
export DETECTOR_LR="${DETECTOR_LR:-0.0001}"
export POLICY_STAGE1_STEPS="${POLICY_STAGE1_STEPS:-30000}"
export POLICY_STAGE2_STEPS="${POLICY_STAGE2_STEPS:-15000}"
export POLICY_LR="${POLICY_LR:-0.0001}"
export POLICY_STAGE2_LR="${POLICY_STAGE2_LR:-0.00001}"
export VAL_SAMPLES="${VAL_SAMPLES:-2000}" # Set 0 explicitly to use all validation views.
export EVAL_EVERY="${EVAL_EVERY:-500}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0}"
export ONLINE_WORKERS="${ONLINE_WORKERS:-2}"
export ONLINE_PREFETCH="${ONLINE_PREFETCH:-4}"
export ONLINE_TIMEOUT="${ONLINE_TIMEOUT:-180}"
export PARENT_FIRST="${PARENT_FIRST:-0}"
export SEED="${SEED:-42}"
export REPAIR_STEPS="${REPAIR_STEPS:-20}"
export REPEATS="${REPEATS:-3}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export ROLLOUT_BATCH_WAIT_MS="${ROLLOUT_BATCH_WAIT_MS:-10}"
[[ "$size" =~ ^[0-9]+$ ]] || { echo 'SIZE must be numeric' >&2; exit 2; }
base="$data/$size"
detector_data="${DETECTOR_DATA_ROOT:-$data}/$size/rendered"
output="$runs/$size"
stage=$STAGE
case "$stage" in all|detector|policy|evaluate) ;; *) echo 'STAGE must be all, detector, policy, or evaluate' >&2; exit 2;; esac
if [[ "$stage" == policy ]]; then
  # Training consumes manifests, not scale evaluation/provenance metadata.
  for input in policy-train.jsonl policy-val.jsonl pages-train.jsonl; do
    [[ -s "$base/css/$input" ]] || {
      echo "Missing policy training input: $base/css/$input" >&2
      echo 'Use the existing prepared DATA_ROOT and the original RUN_ROOT to resume. No automatic data regeneration or path fallback is performed.' >&2
      exit 2
    }
  done
else
  [[ -s "$base/coverage.json" ]] || { echo "Complete prepare_webui_scales.sh first: $base/coverage.json missing" >&2; exit 2; }
fi
if [[ "${REUSE_LEGACY_DOM:-0}" == 1 ]]; then
  python - "$data" "$size" <<'PY'
import json,sys
from framediff.visual import validate_legacy_dataset
print(json.dumps({'legacy_dom_preflight':validate_legacy_dataset(sys.argv[1],sys.argv[2]),
                  'regenerate_data':False}),flush=True)
PY
fi
if [[ "$stage" != policy ]]; then
  common=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["common_test_scale"])' "$data/split-report.json")
fi
device=${DEVICE:-cuda}
detector="${DETECTOR_CHECKPOINT:-$output/detector/best.pt}"
if [[ -n "${DETECTOR_CHECKPOINT:-}" && ( "$stage" == all || "$stage" == detector ) ]]; then
  echo 'Use STAGE=policy or evaluate with DETECTOR_CHECKPOINT' >&2; exit 2
fi
needs_detector=0
for mode in $POLICY_MODES; do
  case "$mode" in abstract) needs_detector=1;; screenshot) ;; *) echo 'Invalid POLICY_MODES' >&2; exit 2;; esac
done
if [[ "$stage" == detector || ( "$stage" == all && "$needs_detector" == 1 ) ]]; then
  options=()
  if [[ -f "$output/detector/last.pt" ]]; then options=(--resume "$output/detector/last.pt"); fi
  python -m framediff visual-train-detector \
    --train "$detector_data/detector-train.jsonl" --val "$detector_data/detector-val.jsonl" \
    --out "$output/detector" --device "$device" --seed "${SEED:-42}" \
    --steps "${DETECTOR_STEPS:-20000}" --lr "${DETECTOR_LR:-0.0001}" \
    --render-workers "${DETECTOR_RENDER_WORKERS:-2}" --render-prefetch "${DETECTOR_RENDER_PREFETCH:-4}" \
    --render-timeout "${DETECTOR_RENDER_TIMEOUT:-900}" \
    --loader-workers "${DETECTOR_LOADER_WORKERS:-2}" --loader-prefetch "${DETECTOR_LOADER_PREFETCH:-2}" \
    --batch-size "${BATCH_SIZE:-2}" --accumulation "${ACCUMULATION:-4}" \
    --val-samples "${VAL_SAMPLES:-2000}" --eval-every "${EVAL_EVERY:-500}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA:-0}" ${options[@]+"${options[@]}"}
  for population in own common; do
    testbase="${DETECTOR_DATA_ROOT:-$data}/$size"
    [[ "$population" == common ]] && testbase="${DETECTOR_DATA_ROOT:-$data}/$common"
    python -m framediff visual-evaluate-detector --data "$testbase/rendered/detector-test.jsonl" \
      --checkpoint "$detector" --out "$output/detector-test-$population.json" \
      --device "$device" --threshold "${DETECTOR_THRESHOLD:-0.4}"
  done
fi
if [[ "$stage" == all || "$stage" == policy ]]; then
  if [[ "$needs_detector" == 1 ]]; then
  for split in train val; do
    python -m framediff visual-cache-targets --data "$base/css/policy-$split.jsonl" \
      --checkpoint "$detector" --out "$output/shared/predicted-$split" \
      --device "$device" --size "${IMAGE_SIZE:-384}" --resume
  done
  fi
  for mode in ${POLICY_MODES:-abstract screenshot}; do
    POLICY_MODES="$mode" bash scripts/train_tuide_scale.sh "${POLICY_SCALE:-s}" \
      "$base/css" "$detector" "$output/shared" "$output/tuide-${POLICY_SCALE:-s}-$mode"
  done
fi
if [[ "$stage" == all || "$stage" == evaluate ]]; then
  for mode in ${POLICY_MODES:-abstract screenshot}; do
    label=abstract;flag=--abstract-checkpoint;policy_stage=2
    detector_options=(--detector-checkpoint "$detector")
    if [[ "$mode" == screenshot ]]; then
      label=raw;flag=--raw-checkpoint;policy_stage=1;detector_options=()
    fi
    checkpoint="$output/tuide-${POLICY_SCALE:-s}-$mode/replacement/$label-stage$policy_stage/best.pt"
    for population in own common; do
      testbase=$base
      [[ "$population" == common ]] && testbase="$data/$common"
      python -m framediff visual-test-rollout --data "$testbase/css/policy-test.jsonl" \
        --pages "$testbase/css/pages-test.jsonl" "$flag" "$checkpoint" \
        ${detector_options[@]+"${detector_options[@]}"} --out "$output/eval-${POLICY_SCALE:-s}-$mode-$population" \
        --device "$device" --steps "$REPAIR_STEPS" --repeats "$REPEATS" \
        --workers "${ROLLOUT_WORKERS:-4}" --inference-batch-size "${ROLLOUT_BATCH_SIZE:-4}" \
        --batch-wait-ms "${ROLLOUT_BATCH_WAIT_MS:-10}" --resume
    done
  done
fi
echo "Completed $stage for $size. Summarize after all scale jobs finish:"
echo "python -m framediff visual-webui-scales-summary --data $data --runs $runs"

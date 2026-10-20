#!/usr/bin/env bash
# Resume when last.pt exists; otherwise start a fresh training run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == legacy-* ]]; then
  case "$1" in
    legacy-detector-7k) size=7000; gpu=2; mode=detector ;;
    legacy-detector-56169) size=56169; gpu=3; mode=detector ;;
    legacy-tuide-7k) size=7000; gpu=2; mode=abstract ;;
    legacy-tuide-56169) size=56169; gpu=3; mode=abstract ;;
    legacy-screenshot-7k) size=7000; gpu=4; mode=screenshot ;;
    legacy-screenshot-56169) size=56169; gpu=2; mode=screenshot ;;
    *) echo 'Unknown legacy experiment' >&2; exit 2 ;;
  esac
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu}"
  export REUSE_LEGACY_DOM=1
  detector_steps=20000; stage1_steps=30000; stage2_steps=15000
  if [[ "$size" == 56169 ]]; then
    detector_steps=40000; stage1_steps=60000; stage2_steps=30000
  fi
  export DETECTOR_STEPS="${DETECTOR_STEPS:-$detector_steps}"
  export POLICY_STAGE1_STEPS="${POLICY_STAGE1_STEPS:-$stage1_steps}"
  export POLICY_STAGE2_STEPS="${POLICY_STAGE2_STEPS:-$stage2_steps}"
  export POLICY_SCREENSHOT_STEPS="${POLICY_SCREENSHOT_STEPS:-$((POLICY_STAGE1_STEPS+POLICY_STAGE2_STEPS))}"
  if [[ "$mode" == detector ]]; then
    export STAGE=detector BATCH_SIZE="${BATCH_SIZE:-1}" ACCUMULATION="${ACCUMULATION:-8}"
    export POLICY_MODES=abstract
  else
    export STAGE="${STAGE:-policy}" POLICY_MODES="$mode"
  fi
  exec bash scripts/run_webui_scale.sh "$size" \
    "${LEGACY_DATA_ROOT:-data/webui-7k-56169-safe-margin-v1}" \
    "${LEGACY_RUN_ROOT:-runs/webui-legacy-dom-shared-v1}"
fi
case "${1:-}" in
  detector-7k) size=7000; gpu=2; steps=20000; mode=detector ;;
  detector-56169) size=56169; gpu=3; steps=40000; mode=detector ;;
  screenshot-7k) size=7000; gpu=4; mode=screenshot ;;
  *) echo 'Usage: bash scripts/resume_webui_training.sh detector-7k|detector-56169|screenshot-7k' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$gpu
export DEVICE=cuda PYTHONUNBUFFERED=1 TUIDE_SAVE_DIAGNOSTICS=0
export TUIDE_SHARED_CACHE="${TUIDE_SHARED_CACHE:-$PWD/data/webui-shared-cache}"
if [[ "$mode" == detector ]]; then
  export BATCH_SIZE="${BATCH_SIZE:-1}" ACCUMULATION="${ACCUMULATION:-8}"
else
  export BATCH_SIZE="${BATCH_SIZE:-2}" ACCUMULATION="${ACCUMULATION:-4}"
fi
export VAL_SAMPLES="${VAL_SAMPLES:-2000}" EVAL_EVERY="${EVAL_EVERY:-500}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}" EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0}"
export SEED="${SEED:-42}"
require_file() { [[ -s "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
prepare_output() {
  options=()
  if [[ -f "$1/last.pt" ]]; then
    options=(--resume "$1/last.pt")
    echo "Resuming: $1/last.pt"
  else
    if [[ -d "$1" ]]; then
      backup=$(mktemp -d "$1.backup.XXXXXX")
      mv "$1" "$backup/run"
      echo "Previous output preserved: $backup/run"
    fi
    echo "Starting fresh: $1 (existing data/cache reused)"
  fi
}
if [[ "$mode" == detector ]]; then
  data="data/webui-detector-full-dom-v1/$size/rendered"
  output="runs/webui-detector-full-dom-v1/$size/detector"
  require_file "$data/detector-train.jsonl"
  require_file "$data/detector-val.jsonl"
  prepare_output "$output"
  exec python -m framediff visual-train-detector \
    --train "$data/detector-train.jsonl" --val "$data/detector-val.jsonl" \
    --out "$output" ${options[@]+"${options[@]}"} --device "$DEVICE" --seed "$SEED" \
    --steps "${DETECTOR_STEPS:-$steps}" --lr "${DETECTOR_LR:-0.0001}" \
    --batch-size "$BATCH_SIZE" --accumulation "$ACCUMULATION" \
    --loader-workers "${DETECTOR_LOADER_WORKERS:-2}" --loader-prefetch "${DETECTOR_LOADER_PREFETCH:-2}" \
    --render-workers "${DETECTOR_RENDER_WORKERS:-2}" --render-prefetch "${DETECTOR_RENDER_PREFETCH:-4}" \
    --render-timeout "${DETECTOR_RENDER_TIMEOUT:-900}" \
    --val-samples "$VAL_SAMPLES" --eval-every "$EVAL_EVERY" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" --early-stop-min-delta "$EARLY_STOP_MIN_DELTA"
fi
export POLICY_MODES=screenshot POLICY_SCALE=s
export POLICY_SCREENSHOT_STEPS="${POLICY_SCREENSHOT_STEPS:-30000}" POLICY_LR="${POLICY_LR:-0.0001}"
export ONLINE_WORKERS="${ONLINE_WORKERS:-2}" ONLINE_PREFETCH="${ONLINE_PREFETCH:-4}"
export ONLINE_TIMEOUT="${ONLINE_TIMEOUT:-900}" PARENT_FIRST=0
output=runs/webui-screenshot-single-v1/7000/tuide-s-screenshot
for input in policy-train.jsonl policy-val.jsonl pages-train.jsonl; do
  require_file "data/webui-7k-56169-safe-margin-v1/7000/css/$input"
done
prepare_output "$output/replacement/raw-stage1"
exec bash scripts/train_tuide_scale.sh s \
  data/webui-7k-56169-safe-margin-v1/7000/css /dev/null \
  runs/webui-screenshot-single-v1/7000/shared-unused "$output"

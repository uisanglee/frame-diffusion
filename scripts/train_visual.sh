#!/usr/bin/env bash
# Independent processes release GPU memory between parser/policy stages.
set -euo pipefail
if [[ $# -ne 2 ]]; then echo 'Usage: bash scripts/train_visual.sh DATA_DIR RUN_DIR' >&2; exit 2; fi
data_dir=$1
run_dir=$2
device=${DEVICE:-cuda}
abs_size=${ABS_SIZE:-384}
raw_size=${RAW_SIZE:-384}
manifest=${TRAIN_HTML_MANIFEST:-$data_dir/source/manifest.jsonl}
if [[ -z "${TRAIN_HTML_MANIFEST:-}" && ! -f "$manifest" ]]; then
  python -m framediff visual-generate --out "$data_dir/source" --count "${TRAIN_PAGES:-1000}"
fi
python -m framediff visual-build-data --manifest "$manifest" --out "$data_dir/rendered" \
  --abstract-size "$abs_size" --trajectories "${TRAJECTORIES:-3}" --max-noise "${MAX_NOISE:-4}" --resume
train_stage() {
  local command=$1 output=$2 init=$3
  shift 3
  local checkpoint_options=()
  if [[ -f "$output/last.pt" ]]; then checkpoint_options=(--resume "$output/last.pt")
  elif [[ -f "$output/best.pt" ]]; then checkpoint_options=(--resume "$output/best.pt")
  elif [[ -d "$output" ]]; then
    local archive="${output}.incomplete-$(date +%s)"
    echo "No checkpoint in partial stage; preserving it as $archive"
    mv "$output" "$archive"
    if [[ -n "$init" ]]; then checkpoint_options=(--init-checkpoint "$init"); fi
  elif [[ -n "$init" ]]; then checkpoint_options=(--init-checkpoint "$init"); fi
  python -m framediff "$command" --out "$output" --device "$device" \
    --batch-size "${BATCH_SIZE:-2}" --accumulation "${ACCUMULATION:-4}" \
    "${checkpoint_options[@]}" "$@"
}
train_stage visual-train-detector "$run_dir/detector" '' \
  --train "$data_dir/rendered/detector-train.jsonl" --val "$data_dir/rendered/detector-val.jsonl" \
  --steps "${DETECTOR_STEPS:-10000}"
python -m framediff visual-evaluate-detector --data "$data_dir/rendered/detector-test.jsonl" \
  --checkpoint "$run_dir/detector/best.pt" --out "$run_dir/detector-test.json" --device "$device"
for split in train val; do
  python -m framediff visual-cache-targets --data "$data_dir/rendered/policy-$split.jsonl" \
    --checkpoint "$run_dir/detector/best.pt" --out "$data_dir/predicted-$split" \
    --device "$device" --size "$abs_size" --resume
done
# Equal two-stage optimizer schedules for both policies; stage 2 exposes abstract
# policy to imperfect frozen-parser targets instead of only perfect DOM targets.
for mode in screenshot abstract; do
  size=$raw_size; label=raw
  if [[ "$mode" == abstract ]]; then size=$abs_size; label=abstract; fi
  train_stage visual-train-policy "$run_dir/$label-stage1" '' \
    --train "$data_dir/rendered/policy-train.jsonl" --val "$data_dir/rendered/policy-val.jsonl" \
    --mode "$mode" --size "$size" --steps "${POLICY_STAGE1_STEPS:-5000}" --bf16
  train_data=$data_dir/rendered/policy-train.jsonl
  val_data=$data_dir/rendered/policy-val.jsonl
  probability=0
  if [[ "$mode" == abstract ]]; then
    train_data=$data_dir/predicted-train/data.jsonl
    val_data=$data_dir/predicted-val/data.jsonl
    probability=${PREDICTION_PROBABILITY:-1.0}
  fi
  train_stage visual-train-policy "$run_dir/policy-$label" "$run_dir/$label-stage1/best.pt" \
    --train "$train_data" --val "$val_data" --mode "$mode" --size "$size" \
    --steps "${POLICY_STAGE2_STEPS:-5000}" --lr 0.00003 --prediction-probability "$probability" --bf16
done
echo "Training complete: $run_dir. See docs/visual-policy.md for held-out evaluation."

#!/usr/bin/env bash
# Fine-tune every visual component on a domain-disjoint WebUI subset.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo 'Usage: bash scripts/finetune_visual_webui.sh WEBUI_SPLIT_ROOT DATA_DIR RUN_DIR PROCEDURAL_RUN_DIR' >&2
  exit 2
fi
webui_root=$1
data_dir=$2
run_dir=$3
base_run=$4
device=${DEVICE:-cuda}
abs_size=${ABS_SIZE:-384}
raw_size=${RAW_SIZE:-384}
from_procedural=${FINETUNE_FROM_PROCEDURAL:-1}
detector_source=${DETECTOR_SOURCE:-rendered}
if [[ "$from_procedural" != 0 && "$from_procedural" != 1 ]]; then
  echo 'FINETUNE_FROM_PROCEDURAL must be 0 or 1' >&2; exit 2
fi
if [[ "$detector_source" != rendered && "$detector_source" != native ]]; then
  echo 'DETECTOR_SOURCE must be rendered or native' >&2; exit 2
fi

python -m framediff visual-import-webui --root "$webui_root" --out "$data_dir/source" \
  --view "${WEBUI_VIEW:-default_1280-720}" --train-count "${WEBUI_TRAIN:-600}" \
  --val-count "${WEBUI_VAL:-200}" --test-count "${WEBUI_TEST:-200}" --seed "${SEED:-42}" --resume

python -m framediff visual-build-data --manifest "$data_dir/source/manifest.jsonl" --out "$data_dir/rendered" \
  --abstract-size "$abs_size" --trajectories "${TRAJECTORIES:-3}" --max-noise "${MAX_NOISE:-4}" \
  --min-elements "${MIN_ELEMENTS:-3}" --max-source-mae "${MAX_SOURCE_MAE:-1.0}" --resume

train_stage() {
  local command=$1 output=$2 init=$3 lr=$4
  shift 4
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
    --batch-size "${BATCH_SIZE:-2}" --accumulation "${ACCUMULATION:-4}" --lr "$lr" \
    "${checkpoint_options[@]}" "$@"
}

detector_train="$data_dir/rendered/detector-train.jsonl"
detector_val="$data_dir/rendered/detector-val.jsonl"
detector_test="$data_dir/rendered/detector-test.jsonl"
if [[ "$detector_source" == native ]]; then
  detector_train="$data_dir/source/native-detector-train.jsonl"
  detector_val="$data_dir/source/native-detector-val.jsonl"
  detector_test="$data_dir/source/native-detector-test.jsonl"
  echo 'Using WebUI AX semantic detector labels; painted-region is not annotated in this ablation.'
fi

detector_init=''
raw_init=''
abstract_init=''
if [[ "$from_procedural" == 1 ]]; then
  detector_init="$base_run/detector/best.pt"
  raw_init="$base_run/policy-raw/best.pt"
  abstract_init="$base_run/policy-abstract/best.pt"
  for checkpoint in "$detector_init" "$raw_init" "$abstract_init"; do
    if [[ ! -f "$checkpoint" ]]; then echo "Missing procedural checkpoint: $checkpoint" >&2; exit 2; fi
  done
fi

train_stage visual-train-detector "$run_dir/detector" "$detector_init" "${DETECTOR_LR:-0.00003}" \
  --train "$detector_train" --val "$detector_val" --steps "${DETECTOR_STEPS:-5000}"
python -m framediff visual-evaluate-detector --data "$detector_test" \
  --checkpoint "$run_dir/detector/best.pt" --out "$run_dir/detector-test.json" --device "$device"

for split in train val; do
  python -m framediff visual-cache-targets --data "$data_dir/rendered/policy-$split.jsonl" \
    --checkpoint "$run_dir/detector/best.pt" --out "$data_dir/predicted-$split" \
    --device "$device" --size "$abs_size" --resume
done

for mode in screenshot abstract; do
  size=$raw_size; label=raw; init=$raw_init
  if [[ "$mode" == abstract ]]; then size=$abs_size; label=abstract; init=$abstract_init; fi
  train_stage visual-train-policy "$run_dir/$label-stage1" "$init" "${POLICY_LR:-0.00003}" \
    --train "$data_dir/rendered/policy-train.jsonl" --val "$data_dir/rendered/policy-val.jsonl" \
    --mode "$mode" --size "$size" --steps "${POLICY_STAGE1_STEPS:-3000}" --bf16
  train_data="$data_dir/rendered/policy-train.jsonl"
  val_data="$data_dir/rendered/policy-val.jsonl"
  probability=0
  if [[ "$mode" == abstract ]]; then
    train_data="$data_dir/predicted-train/data.jsonl"
    val_data="$data_dir/predicted-val/data.jsonl"
    probability=${PREDICTION_PROBABILITY:-1.0}
  fi
  train_stage visual-train-policy "$run_dir/policy-$label" "$run_dir/$label-stage1/best.pt" "${POLICY_STAGE2_LR:-0.00001}" \
    --train "$train_data" --val "$val_data" --mode "$mode" --size "$size" \
    --steps "${POLICY_STAGE2_STEPS:-3000}" --prediction-probability "$probability" --bf16
done

echo "WebUI fine-tuning complete: $run_dir"
echo "Held-out controlled manifest: $data_dir/rendered/pages-test.jsonl"
echo "Original WebUI test selection: $data_dir/source/manifest-test.jsonl"

#!/usr/bin/env bash
# Generate, train the NEW plan-conditioned policy, then evaluate held-out IR.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/run_plan_synthetic.sh DATA_DIR RUN_DIR' >&2
  exit 2
fi
data_dir=$1
run_dir=$2
if [[ ! -f "$data_dir/train.jsonl" ]]; then
  python -m framediff generate --out "$data_dir" --count "${DATA_COUNT:-5000}" --seed "${SEED:-42}"
fi
python -m framediff train --conditioning plan \
  --train "$data_dir/train.jsonl" --val "$data_dir/val.jsonl" --out "$run_dir/train" \
  --device "${DEVICE:-cuda}" --bf16 --hidden "${HIDDEN:-256}" --layers "${LAYERS:-4}" \
  --batch-size "${BATCH_SIZE:-8}" --accumulation "${ACCUMULATION:-4}" --steps "${TRAIN_STEPS:-10000}"
python -m framediff plan-evaluate --data "$data_dir/test.jsonl" \
  --checkpoint "$run_dir/train/best.pt" --out "$run_dir/evaluation" --device "${DEVICE:-cuda}" \
  --limit "${TEST_LIMIT:-100}" --steps "${REPAIR_STEPS:-10}" --beam 2 --topk 32 --budget 320

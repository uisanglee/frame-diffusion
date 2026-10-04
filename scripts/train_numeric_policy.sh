#!/usr/bin/env bash
# Never renders HTML, trains a detector, or modifies the original data/run.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: bash scripts/train_numeric_policy.sh EXISTING_RENDERED_DIR DETECTOR_BEST_PT NEW_RUN_DIR' >&2
  exit 2
fi
rendered=$1
detector=$2
run=$3
[[ -f "$detector" ]] || { echo "Missing detector: $detector" >&2; exit 2; }
python -m framediff visual-subset-numeric --rendered "$rendered" --out "$run/subset"
for split in train val; do
  python -m framediff visual-cache-targets --data "$run/subset/policy-$split.jsonl" \
    --checkpoint "$detector" --out "$run/predicted-$split" --device "${DEVICE:-cuda}" --resume
done
for head in ${POLICY_HEADS:-flat hierarchical}; do
  for mode in screenshot abstract; do
    label=raw
    [[ "$mode" == abstract ]] && label=abstract
    for stage in 1 2; do
      output="$run/$head/$label-stage$stage"
      train="$run/subset/policy-train.jsonl"
      val="$run/subset/policy-val.jsonl"
      probability=0
      steps=${POLICY_STAGE1_STEPS:-30000}
      lr=${POLICY_LR:-0.0001}
      options=()
      if [[ "$stage" == 2 ]]; then
        steps=${POLICY_STAGE2_STEPS:-15000};lr=${POLICY_STAGE2_LR:-0.00001}
        options=(--init-checkpoint "$run/$head/$label-stage1/best.pt")
        if [[ "$mode" == abstract ]]; then
          train="$run/predicted-train/data.jsonl";val="$run/predicted-val/data.jsonl";probability=1
        fi
      fi
      if [[ -f "$output/last.pt" ]]; then options=(--resume "$output/last.pt"); fi
      python -m framediff visual-train-policy --train "$train" --val "$val" --out "$output" \
        --mode "$mode" --policy-head "$head" --numeric-only --device "${DEVICE:-cuda}" \
        --steps "$steps" --lr "$lr" --prediction-probability "$probability" \
        --batch-size "${BATCH_SIZE:-2}" --accumulation "${ACCUMULATION:-4}" \
        --val-samples "${VAL_SAMPLES:-2000}" --eval-every "${EVAL_EVERY:-500}" \
        --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" --bf16 "${options[@]}"
    done
  done
done
echo "Finished policy-only training in $run. Detector and source renders unchanged."

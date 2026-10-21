#!/usr/bin/env bash
# Online workers render new CURRENT states; detector and source assets stay frozen.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: bash scripts/train_numeric_policy.sh EXISTING_RENDERED_DIR DETECTOR_BEST_PT NEW_RUN_DIR' >&2
  exit 2
fi
rendered=$1
detector=$2
run=$3
policy_run=${POLICY_RUN_DIR:-$run}
representation=${ABSTRACTION_REPRESENTATION:-occupancy-boundary}
case "$representation" in
  abstract-rgb|occupancy|boundary|occupancy-boundary) ;;
  *) echo 'Invalid ABSTRACTION_REPRESENTATION' >&2; exit 2;;
esac
scale_options=()
if [[ -n "${POLICY_SCALE:-}" ]]; then
  case "$POLICY_SCALE" in s|m|l) scale_options=(--policy-scale "$POLICY_SCALE");;
    *) echo 'POLICY_SCALE must be s, m, or l' >&2; exit 2;; esac
fi
online=()
owner_options=()
case "${PARENT_FIRST:-0}" in
  1) owner_options+=(--parent-first --parent-first-max-probes "${PARENT_FIRST_MAX_PROBES:-200}") ;;
  0) ;;
  *) echo 'PARENT_FIRST must be 0 or 1' >&2; exit 2 ;;
esac
case "${CSS_STYLESHEETS:-0}" in
  1) owner_options+=(--stylesheets --max-nodes "${MAX_CSS_OWNERS:-512}") ;;
  0) ;;
  *) echo 'CSS_STYLESHEETS must be 0 or 1' >&2; exit 2 ;;
esac
case "${ONLINE_CORRUPTION:-1}" in
  1) online=(--online-corruption --online-targets "$rendered/pages-train.jsonl"
        --online-workers "${ONLINE_WORKERS:-2}" --online-prefetch "${ONLINE_PREFETCH:-4}"
        --online-max-noise "${ONLINE_MAX_NOISE:-4}" --online-timeout "${ONLINE_TIMEOUT:-180}") ;;
  0) ;;
  *) echo 'ONLINE_CORRUPTION must be 0 or 1' >&2; exit 2 ;;
esac
prepare=${PREPARE_DATA:-1}
if [[ "$prepare" != 0 && "$prepare" != 1 ]]; then
  echo 'PREPARE_DATA must be 0 or 1' >&2;exit 2
fi
needs_predicted_targets=0
for requested_mode in ${POLICY_MODES:-screenshot abstract}; do
  [[ "$requested_mode" == screenshot || "$requested_mode" == abstract ]] || {
    echo "Invalid POLICY_MODES item: $requested_mode" >&2; exit 2;
  }
  [[ "$requested_mode" == abstract ]] && needs_predicted_targets=1
done
if [[ "$needs_predicted_targets" == 1 ]]; then
  [[ -f "$detector" ]] || { echo "Missing detector: $detector" >&2; exit 2; }
  if [[ "$prepare" == 1 ]]; then
    echo 'Run scripts/prepare_numeric_policy_tree.sh once, then set PREPARE_DATA=0.' >&2
    exit 2
  fi
  for required in predicted-train/data.jsonl predicted-val/data.jsonl; do
    [[ -s "$run/$required" ]] || { echo "PREPARE_DATA=0 but missing $run/$required" >&2; exit 2; }
  done
fi
for head in ${POLICY_HEADS:-replacement}; do
  for mode in ${POLICY_MODES:-screenshot abstract}; do
    [[ "$head" == replacement ]] || { echo 'Use replacement head; old gain/set-loss policies are retired.' >&2; exit 2; }
    [[ "$mode" == screenshot || "$mode" == abstract ]] || { echo "Invalid POLICY_MODES item: $mode" >&2; exit 2; }
    label=raw
    representation_options=()
    if [[ "$mode" == abstract ]]; then
      representation_options=(--abstraction-representation "$representation")
    elif [[ "$representation" != occupancy-boundary ]]; then
      echo 'ABSTRACTION_REPRESENTATION requires POLICY_MODES=abstract' >&2; exit 2
    fi
    [[ "$mode" == abstract ]] && label=abstract
    stages=(1)
    [[ "$mode" == abstract ]] && stages+=(2)
    for stage in "${stages[@]}"; do
      output="$policy_run/$head/$label-stage$stage"
      train="$rendered/policy-train.jsonl"
      val="$rendered/policy-val.jsonl"
      condition=()
      steps=${POLICY_STAGE1_STEPS:-30000}
      # Keep the stage1 path so existing screenshot checkpoints resume in place.
      [[ "$mode" == screenshot ]] && steps=${POLICY_SCREENSHOT_STEPS:-$steps}
      lr=${POLICY_LR:-0.0001}
      options=()
      if [[ "$stage" == 2 ]]; then
        steps=${POLICY_STAGE2_STEPS:-15000};lr=${POLICY_STAGE2_LR:-0.00001}
        options=(--init-checkpoint "$policy_run/$head/$label-stage1/best.pt")
        if [[ "$mode" == abstract ]]; then
          train="$run/predicted-train/data.jsonl";val="$run/predicted-val/data.jsonl";condition=(--predicted-targets)
        fi
      fi
      if [[ -f "$output/last.pt" ]]; then options=(--resume "$output/last.pt"); fi
      python -m framediff visual-tree-train --train "$train" --val "$val" --out "$output" \
        --mode "$mode" --device "${DEVICE:-cuda}" ${representation_options[@]+"${representation_options[@]}"} ${owner_options[@]+"${owner_options[@]}"} ${scale_options[@]+"${scale_options[@]}"} \
        --steps "$steps" --lr "$lr" ${condition[@]+"${condition[@]}"} ${online[@]+"${online[@]}"} \
        --batch-size "${BATCH_SIZE:-2}" --accumulation "${ACCUMULATION:-4}" \
        --seed "${SEED:-42}" \
        --val-samples "${VAL_SAMPLES:-2000}" --eval-every "${EVAL_EVERY:-500}" \
        --policy-metric-samples "${POLICY_METRIC_SAMPLES:-128}" \
        --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
        --early-stop-min-delta "${EARLY_STOP_MIN_DELTA:-0}" --bf16 ${options[@]+"${options[@]}"}
    done
  done
done
echo "Finished policy-only training in $policy_run. Detector and source renders unchanged."

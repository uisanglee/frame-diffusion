#!/usr/bin/env bash
# Independent experiment. Does not call train_tuide_scale.sh or alter its outputs.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
data=${1:-data/webui-css-diffusion-v3-size-margin}
mode=${POLICY_MODES:-abstract}
scale=${POLICY_SCALE:-s}
target=${TARGET_SOURCE:-oracle}
out=${2:-runs/css-diffusion-v3-size-margin-$scale-$mode-$target}
for split in train val; do
  if [[ ! -s "$data/$split.jsonl" ]]; then
    echo "Missing $data/$split.jsonl; run visual-css-diffusion-prepare first." >&2
    exit 1
  fi
done
extra=()
if [[ ${RESUME:-0} == 1 ]]; then extra+=(--resume); fi
if [[ ${NO_PRETRAINED:-0} == 1 ]]; then extra+=(--no-pretrained); fi
exec python -m framediff visual-css-diffusion-train \
  --data "$data" --out "$out" --mode "$mode" --policy-scale "$scale" \
  --target-source "$target" --device "${DEVICE:-cuda}" \
  --steps "${TRAIN_STEPS:-30000}" --diffusion-steps "${DIFFUSION_STEPS:-200}" \
  --batch-size "${BATCH_SIZE:-2}" --accumulation "${ACCUMULATION:-4}" \
  --lr "${LR:-0.0001}" --eval-every "${EVAL_EVERY:-500}" \
  --val-samples "${VAL_SAMPLES:-128}" --log-every "${LOG_EVERY:-50}" \
  --max-nodes "${MAX_NODES:-512}" --size "${IMAGE_SIZE:-384}" \
  --sigma-min "${SIGMA_MIN:-0.35}" --sigma-max "${SIGMA_MAX:-2.0}" \
  --move-rate "${MOVE_RATE:-0.2}" --auxiliary-weight "${AUXILIARY_WEIGHT:-0.01}" \
  --seed "${SEED:-42}" --cpu-threads "${CPU_THREADS:-4}" "${extra[@]}"

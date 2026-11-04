#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
train=${TRAIN_RUN:-runs/webui-full-dom-screened-7k-v1/7000}
policy=${ABSTRACT_CHECKPOINT:-$train/tuide-s-abstract/replacement/abstract-stage2/best.pt}
detector=${DETECTOR_CHECKPOINT:-$train/detector/best.pt}
data=${PILOT_DATA:-data/webui-full-dom-screened-7k-v1/7000/css}
out=${PILOT_OUT:-runs/webui-image-assets-pilot10}
for path in "$policy" "$detector" "$data/policy-test.jsonl" "$data/../source/manifest-test.jsonl"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
python -m framediff.image_asset_pilot prepare --data "$data" --out "$out" \
  --count 10 --max-candidates "${PILOT_MAX_CANDIDATES:-50}" \
  --max-images "${PILOT_MAX_IMAGES:-32}" --timeout "${IMAGE_DOWNLOAD_TIMEOUT:-10}"
for mode in blocked images; do
  python -m framediff visual-test-rollout \
    --data "$out/$mode/policy-test.jsonl" --pages "$out/$mode/pages-test.jsonl" \
    --abstract-checkpoint "$policy" --detector-checkpoint "$detector" \
    --out "$out/$mode/repair" --device "${DEVICE:-cuda}" \
    --steps "${REPAIR_STEPS:-20}" --repeats 1 --workers 1 --inference-batch-size 1 \
    --pixel-goal-fraction -1 --resume
done
python -m framediff.image_asset_pilot report --out "$out"

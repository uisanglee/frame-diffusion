#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
train=${TRAIN_RUN:-runs/webui-full-dom-screened-7k-v1/7000}
policy=${ABSTRACT_CHECKPOINT:-$train/tuide-s-abstract/replacement/abstract-stage2/best.pt}
detector=${DETECTOR_CHECKPOINT:-$train/detector/best.pt}
root=${PILOT_OUT:-runs/webui-image-assets-pilot10}
steps=${REPAIR_STEPS:-20}
# Check the frozen samples and original checkpoint hashes before starting GPU work.
python - "$root" "$policy" "$detector" "$steps" <<'PY'
import json,sys
from pathlib import Path
from framediff.image_asset_oracle import paired_inputs
from framediff.web_experiment import digest
root=Path(sys.argv[1]);paired=paired_inputs(root)
for mode in paired:
    previous=json.loads((root/mode/'repair/config.json').read_text())
    for setting,path in [('abstract_checkpoint',sys.argv[2]),('detector_checkpoint',sys.argv[3])]:
        old=previous['settings'][setting]
        if previous['inputs'][old]!=digest(path):
            raise SystemExit(f'{setting} differs from original pilot; use the original checkpoint')
    if previous['settings']['steps']!=int(sys.argv[4]) or previous['settings']['pixel_goal_fraction']!=-1:
        raise SystemExit('Original pilot repair-step/stopping settings differ')
    if previous['settings']['threshold']!=0.4 or previous['settings']['repeats']!=1:
        raise SystemExit('Original pilot detector threshold/repeats differ')
    data=str(root/mode/'policy-test.jsonl');pages=str(root/mode/'pages-test.jsonl')
    for setting,path in [('data',data),('pages',pages)]:
        if previous['inputs'][previous['settings'][setting]]!=digest(path):
            raise SystemExit('Prepared pilot manifests changed since original evaluation')
print(f'Using the same {len(paired["blocked"])} samples and checkpoint weights; no selection/download/generation.')
PY
python -m framediff.image_asset_oracle audit --root "$root" \
  --detector-checkpoint "$detector" --device "${DEVICE:-cuda}" --threshold 0.4
for mode in blocked images; do
  python -m framediff visual-test-rollout \
    --data "$root/$mode/policy-test.jsonl" --pages "$root/$mode/pages-test.jsonl" \
    --abstract-checkpoint "$policy" --detector-checkpoint "$detector" --oracle-ablation \
    --out "$root/oracle-analysis/$mode/repair" --device "${DEVICE:-cuda}" \
    --steps "$steps" --repeats 1 --workers 1 --inference-batch-size 1 \
    --threshold 0.4 --pixel-goal-fraction -1 --resume
done
python -m framediff.image_asset_oracle report --root "$root"

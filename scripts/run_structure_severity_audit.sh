#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_RUN=${TRAIN_RUN:-runs/webui-color-images-v1/7000}
SCALE=${SCALE:-s}
case "$SCALE" in s|m|l) ;; *) echo 'SCALE must be s, m or l' >&2; exit 2;; esac
OUT=${SEVERITY_OUTPUT:-runs/tuide-$SCALE-structure-severity-v1}
POLICY=${ABSTRACT_CHECKPOINT:-$TRAIN_RUN/tuide-$SCALE-abstract/replacement/abstract-stage2/best.pt}
DETECTOR=${DETECTOR_CHECKPOINT:-$TRAIN_RUN/detector/best.pt}
for file in "$POLICY" "$DETECTOR"; do
  [[ -f "$file" ]] || { echo "Missing checkpoint: $file" >&2; exit 2; }
done
python -u -m framediff.structure_severity_audit prepare --out "$OUT/inputs" \
  --data "${SEVERITY_DATA:-data/new/7000/css/policy-test.jsonl}" \
  --samples "${SEVERITY_SAMPLES:-20}" --sites "${SEVERITY_SITES:-4}" --seed 42
for level in clean mild medium strong; do
  for budget in ${SEVERITY_BUDGETS:-1 5 20}; do
    echo "=== Same-DOM severity: $level | budget=$budget | TUIDE-$SCALE ==="
    python -u -m framediff visual-test-rollout \
      --data "$OUT/inputs/$level.jsonl" --pages "$OUT/inputs/pages.jsonl" \
      --out "$OUT/evaluations/$level/budget-$budget" \
      --abstract-checkpoint "$POLICY" --detector-checkpoint "$DETECTOR" \
      --oracle-ablation --steps "$budget" --repeats 1 --device cuda \
      --pixel-goal-fraction -1 --workers "${SEVERITY_WORKERS:-1}" --resume
    python -m framediff.structure_severity_audit report --out "$OUT"
  done
done

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_RUN=${TRAIN_RUN:-runs/webui-color-images-v1/7000}
SCALE=${SCALE:-s}
INPUTS=${SEVERITY_INPUTS:-runs/tuide-$SCALE-structure-severity-v1/inputs}
OUT=${PRESERVATION_OUTPUT:-runs/tuide-$SCALE-matched-region-v1}
POLICY=${ABSTRACT_CHECKPOINT:-$TRAIN_RUN/tuide-$SCALE-abstract/replacement/abstract-stage2/best.pt}
DETECTOR=${DETECTOR_CHECKPOINT:-$TRAIN_RUN/detector/best.pt}
for file in "$INPUTS/pages.jsonl" "$INPUTS/severity.jsonl" "$INPUTS/coverage.json" "$POLICY" "$DETECTOR"; do
  [[ -f "$file" ]] || { echo "Missing input: $file" >&2; exit 2; }
done
for level in clean mild medium strong; do
  for budget in ${SEVERITY_BUDGETS:-1 5 20}; do
    echo "=== Matched-region protection | $level | budget=$budget ==="
    python -u -m framediff visual-test-rollout \
      --data "$INPUTS/$level.jsonl" --pages "$INPUTS/pages.jsonl" \
      --out "$OUT/evaluations/$level/budget-$budget" \
      --abstract-checkpoint "$POLICY" --detector-checkpoint "$DETECTOR" \
      --oracle-ablation --steps "$budget" --repeats 1 --device cuda \
      --pixel-goal-fraction -1 --preserve-matched-regions \
      --workers "${SEVERITY_WORKERS:-1}" --resume
    python -m framediff.structure_severity_audit report --out "$OUT" --inputs "$INPUTS"
  done
done

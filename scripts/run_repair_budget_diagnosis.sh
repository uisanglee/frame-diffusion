#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_RUN=${TRAIN_RUN:-runs/webui-color-images-v1/7000}
SOURCE_RUN=${SOURCE_RUN:-runs/paper-color-pilot10-seed42-n2-v1}
DIAG_OUTPUT=${DIAG_OUTPUT:-runs/paper-color-pilot10-seed42-n2-target-budget-diagnosis-v1}
exec python -m framediff.repair_budget_diagnosis \
  --root "$SOURCE_RUN" --out "$DIAG_OUTPUT" \
  --abstract-checkpoint "${ABSTRACT_CHECKPOINT:-$TRAIN_RUN/tuide-s-abstract/replacement/abstract-stage2/best.pt}" \
  --detector-checkpoint "${DETECTOR_CHECKPOINT:-$TRAIN_RUN/detector/best.pt}" \
  --official-repo "${OFFICIAL_REPO:-external/design2code-official}" "$@"

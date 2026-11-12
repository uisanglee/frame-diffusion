#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_RUN=${TRAIN_RUN:-runs/webui-color-images-v1/7000}
exec python -u -m framediff.vlm_first_edit_audit \
  --root "${SOURCE_RUN:-runs/paper-color-pilot10-seed42-n2-v1}" \
  --trace-root "${TRACE_RUN:-runs/paper-color-pilot10-seed42-n2-target-budget-diagnosis-v1}" \
  --out "${AUDIT_OUTPUT:-runs/vlm-first-edit-audit-v1}" \
  --stage1 "${STAGE1_CHECKPOINT:-$TRAIN_RUN/tuide-s-abstract/replacement/abstract-stage1/best.pt}" \
  --stage2 "${STAGE2_CHECKPOINT:-$TRAIN_RUN/tuide-s-abstract/replacement/abstract-stage2/best.pt}" \
  --detector "${DETECTOR_CHECKPOINT:-$TRAIN_RUN/detector/best.pt}" \
  --candidates "${AUDIT_CANDIDATES:-120}" "$@"

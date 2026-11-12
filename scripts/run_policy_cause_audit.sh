#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_RUN=${TRAIN_RUN:-runs/webui-color-images-v1/7000}
SCALE=${SCALE:-s}
case "$SCALE" in s|m|l) ;; *) echo 'SCALE must be s, m or l' >&2; exit 2;; esac
exec python -u -m framediff.policy_cause_audit \
  --data "${AUDIT_DATA:-data/new/7000/css/policy-val.jsonl}" \
  --pages "${AUDIT_PAGES:-data/new/7000/css/pages-val.jsonl}" \
  --stage1 "${STAGE1_CHECKPOINT:-$TRAIN_RUN/tuide-$SCALE-abstract/replacement/abstract-stage1/best.pt}" \
  --stage2 "${STAGE2_CHECKPOINT:-$TRAIN_RUN/tuide-$SCALE-abstract/replacement/abstract-stage2/best.pt}" \
  --detector "${DETECTOR_CHECKPOINT:-$TRAIN_RUN/detector/best.pt}" \
  --out "${AUDIT_OUTPUT:-runs/tuide-$SCALE-cause-audit-v1}" \
  --samples "${AUDIT_SAMPLES:-20}" --online-samples "${AUDIT_ONLINE_SAMPLES:-8}" "$@"

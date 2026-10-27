#!/usr/bin/env bash
# Existing master assets -> timeout-screened 32000/4000/4000 page manifests.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONUNBUFFERED=1 TUIDE_SAVE_DIAGNOSTICS=0
exec python -m framediff.webui_screen \
  --data "${SOURCE_DATA:-data/webui-full-dom-policy-v2}" \
  --detector-rendered "${DETECTOR_RENDERED:-data/webui-full-dom-detector-v2/56169/rendered}" \
  --out "${SCREENED_DATA:-data/webui-full-dom-screened-40k-v1}" \
  --size 40000 --workers "${PREPARE_WORKERS:-4}" \
  --timeout "${SCREEN_TIMEOUT:-180}" --samples "${SCREEN_SAMPLES:-3}" \
  --seed "${SEED:-42}" --max-noise "${ONLINE_MAX_NOISE:-4}" \
  --shared-cache "${TUIDE_SHARED_CACHE:-$repo_root/data/webui-shared-cache}" --resume

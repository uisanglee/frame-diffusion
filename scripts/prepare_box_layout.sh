#!/usr/bin/env bash
# Reuse original rendered assets; never reuse a different normalization contract.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
[[ $# == 2 ]] || { echo 'Usage: bash scripts/prepare_box_layout.sh RENDERED NEW_ROOT' >&2; exit 2; }
python -m framediff visual-tree-normalize --rendered "$1" --out "$2/labels" \
  --box-layout --workers "${PREPARE_WORKERS:-4}" --seed "${SEED:-42}" \
  --max-css-owners "${MAX_CSS_OWNERS:-512}" --observation-size "${IMAGE_SIZE:-384}" \
  --max-pixel-mae "${NORMALIZE_MAX_PIXEL_MAE:-0.001}" --max-box-error "${NORMALIZE_MAX_BOX_ERROR:-1}" --resume
python -m framediff visual-tree-freeze --rendered "$2/labels" --out "$2/css" \
  --workers "${PREPARE_WORKERS:-4}" --seed "${TEST_SEED:-90210}" \
  --max-noise "${ONLINE_MAX_NOISE:-4}" --samples-per-page "${FIXED_SAMPLES_PER_PAGE:-1}" \
  --observation-size "${IMAGE_SIZE:-384}" --resume
echo "Box-layout data ready: $2/css"

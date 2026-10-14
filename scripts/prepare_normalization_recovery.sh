#!/usr/bin/env bash
# Read old rendered/normalization assets; write an independent recovery corpus.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
[[ $# == 3 ]] || { echo 'Usage: bash scripts/prepare_normalization_recovery.sh RENDERED OLD_LABELS NEW_ROOT' >&2; exit 2; }
rendered=$1
old_labels=$2
output=$3
python -m framediff visual-tree-normalize \
  --rendered "$rendered" --reuse-normalized "$old_labels" \
  --out "$output/labels" --recover-normalization \
  --workers "${PREPARE_WORKERS:-4}" --seed "${SEED:-42}" \
  --max-css-owners "${MAX_CSS_OWNERS:-512}" \
  --observation-size "${IMAGE_SIZE:-384}" --resume
python -m framediff visual-tree-freeze \
  --rendered "$output/labels" --out "$output/css" \
  --workers "${PREPARE_WORKERS:-4}" --seed "${TEST_SEED:-90210}" \
  --max-noise "${ONLINE_MAX_NOISE:-4}" \
  --samples-per-page "${FIXED_SAMPLES_PER_PAGE:-1}" \
  --observation-size "${IMAGE_SIZE:-384}" --resume
echo "Recovery corpus ready: $output/css. Keep the original rendered assets."

#!/usr/bin/env bash
# Raw WebUI -> shared assets -> nested 8:1:1 dataset manifests.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/prepare_webui_scales.sh RAW_WEBUI_ROOT DATA_ROOT' >&2
  exit 2
fi
raw=$1
data=$2
if [[ "${BOX_LAYOUT:-0}" != 0 || "${RECOVER_NORMALIZATION:-0}" != 0 || -n "${REUSE_NORMALIZED:-}" ]]; then
  echo 'Retired normalization/xywh options: use safe size/margin preparation with a new labels directory.' >&2
  exit 2
fi
read -r -a sizes <<< "${WEBUI_SIZES:-7000 70000}"
source "$repo_root/scripts/webui_views.sh"
split_options=()
if [[ -n "${MASTER_COUNTS:-}" ]]; then
  read -r -a master_counts <<< "$MASTER_COUNTS"
  split_options+=(--master-counts "${master_counts[@]}")
fi
if [[ -n "${ALLOCATION_COUNTS:-}" ]]; then
  read -r -a allocation_counts <<< "$ALLOCATION_COUNTS"
  split_options+=(--allocation-counts "${allocation_counts[@]}")
fi
python -m framediff visual-webui-scales-prepare --root "$raw" --out "$data" \
  --sizes "${sizes[@]}" --views "${views[@]}" --seed "${SEED:-42}" "${split_options[@]}" --resume
# No historical 10K input, detector checkpoint or offline trajectories required.
python -m framediff visual-build-data --manifest "$data/master/source/manifest.jsonl" \
  --workers "${RENDER_WORKERS:-${PREPARE_WORKERS:-4}}" \
  --out "$data/master/rendered" --clean-only --lazy-train --abstract-size "${IMAGE_SIZE:-384}" \
  --seed "${SEED:-42}" --min-elements "${MIN_ELEMENTS:-3}" \
  --max-source-mae "${MAX_SOURCE_MAE:-1.0}" --resume
python -m framediff visual-tree-normalize --rendered "$data/master/rendered" \
  --out "$data/master/labels" --workers "${PREPARE_WORKERS:-4}" \
  --seed "${SEED:-42}" --max-css-owners "${MAX_CSS_OWNERS:-512}" \
  --observation-size "${IMAGE_SIZE:-384}" \
  --max-pixel-mae "${NORMALIZE_MAX_PIXEL_MAE:-0.01}" \
  --max-box-error "${NORMALIZE_MAX_BOX_ERROR:-1}" --resume
python -m framediff visual-tree-freeze --rendered "$data/master/labels" \
  --out "$data/master/css" --workers "${PREPARE_WORKERS:-4}" \
  --samples-per-page "${FIXED_SAMPLES_PER_PAGE:-1}" --max-noise "${ONLINE_MAX_NOISE:-4}" \
  --observation-size "${IMAGE_SIZE:-384}" --seed "${TEST_SEED:-90210}" --resume
python -m framediff visual-webui-scales-subset --data "$data"
echo "Prepared nested scale datasets in $data. Keep master/ assets."

#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -gt 4 ]]; then
  echo 'Usage: prepare_stylesheet_policy.sh [SOURCE_CORPUS] [DETECTOR] [DATA_OUT] [CACHE_OUT]' >&2
  exit 2
fi
source_corpus=${1:-data/webui-10k-v5-improvement-distribution}
detector=${2:-runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt}
data_out=${3:-data/webui-css-owners-v5-visible}
cache_out=${4:-runs/css-owners-v5-visible-shared}
[[ -f "$source_corpus/rendered/policy-train.jsonl" ]] && source_corpus="$source_corpus/rendered"
# Gap adds two declaration slots. Reparse HTML (reuse existing images), never
# reuse six-field labels as though they contained gap supervision.
labels=${SOURCE_LABELS:-${data_out}-labels}
if [[ $# -eq 0 && -z "${SOURCE_LABELS:-}" && -s data/webui-css-owners-v4-gap-labels/prepare-report.json ]]; then
  labels=data/webui-css-owners-v4-gap-labels
fi
if [[ -s "$labels/prepare-report.json" ]]; then
  python - "$labels/config.json" <<'PY'
import json, sys
from framediff.css_owners import CONTRACT
if json.load(open(sys.argv[1])).get('kind') != CONTRACT:
    raise SystemExit('Old parsed labels do not contain gap slots. Use a new DATA_OUT/SOURCE_LABELS directory.')
PY
  for split in train val test; do
    for kind in policy pages; do
      [[ -s "$labels/$kind-$split.jsonl" ]] || { echo "Incomplete parsed corpus: $labels/$kind-$split.jsonl" >&2; exit 2; }
    done
  done
  echo "Reusing parsed CSS owners and clean target assets: $labels"
else
  python -m framediff visual-tree-prepare --rendered "$source_corpus" \
    --out "$labels" --stylesheets --max-css-owners "${MAX_CSS_OWNERS:-512}" --resume
fi
python -m framediff visual-tree-freeze --rendered "$labels" \
  --out "$data_out" --samples-per-page "${FIXED_SAMPLES_PER_PAGE:-1}" \
  --max-noise "${ONLINE_MAX_NOISE:-4}" --observation-size "${IMAGE_SIZE:-384}" --resume
for split in train val; do
  reuse=()
  previous="${REUSE_TARGET_CACHE:-runs/css-owners-v2-shared}/predicted-$split"
  if [[ -f "$previous/config.json" ]]; then reuse=(--reuse-cache "$previous"); fi
  CUDA_VISIBLE_DEVICES=${CACHE_GPU:-4} python -m framediff visual-cache-targets \
    --data "$data_out/policy-$split.jsonl" --checkpoint "$detector" \
    --out "$cache_out/predicted-$split" --device cuda --resume "${reuse[@]}"
done
echo "Ready: $data_out. Inspect freeze-report.json before training."

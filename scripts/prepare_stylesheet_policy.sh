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
data_out=${3:-data/webui-css-owners-v2}
cache_out=${4:-runs/css-owners-v2-shared}
[[ -f "$source_corpus/rendered/policy-train.jsonl" ]] && source_corpus="$source_corpus/rendered"
python -m framediff visual-tree-prepare --rendered "$source_corpus" \
  --out "${data_out}-labels" --stylesheets --max-css-owners "${MAX_CSS_OWNERS:-512}" --resume
python -m framediff visual-tree-freeze --rendered "${data_out}-labels" \
  --out "$data_out" --samples-per-page "${FIXED_SAMPLES_PER_PAGE:-1}" \
  --max-noise "${ONLINE_MAX_NOISE:-4}" --resume
for split in train val; do
  reuse=()
  previous="${REUSE_TARGET_CACHE:-runs/css-tree-v1-shared}/predicted-$split"
  if [[ -f "$previous/config.json" ]]; then reuse=(--reuse-cache "$previous"); fi
  CUDA_VISIBLE_DEVICES=${CACHE_GPU:-4} python -m framediff visual-cache-targets \
    --data "$data_out/policy-$split.jsonl" --checkpoint "$detector" \
    --out "$cache_out/predicted-$split" --device cuda --resume "${reuse[@]}"
done
echo "Ready: $data_out. Inspect freeze-report.json before training."

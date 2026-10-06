#!/usr/bin/env bash
# Exact CSS replacement teachers. Parse cached HTML; do not rerender it.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ $# -gt 4 ]]; then
  echo 'Usage: bash scripts/prepare_numeric_policy_tree.sh [SOURCE_RENDERED] [DETECTOR_BEST_PT] [TREE_RENDERED] [SHARED_RUN]' >&2
  exit 2
fi

source_rendered=${1:-data/webui-10k-v5-improvement-distribution}
detector=${2:-runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt}
tree_rendered=${3:-data/webui-css-tree-v1}
shared_run=${4:-runs/css-tree-v1-shared}
python_bin=${PYTHON:-python}
cache_gpu=${CACHE_GPU:-3}

if [[ -f "$source_rendered/rendered/policy-train.jsonl" ]]; then
  source_rendered="$source_rendered/rendered"
fi
for split in train val test; do
  for kind in policy pages; do
    path="$source_rendered/$kind-$split.jsonl"
    [[ -f "$path" ]] || { echo "Missing source file: $path" >&2; exit 2; }
  done
done
[[ -f "$detector" ]] || { echo "Missing detector checkpoint: $detector" >&2; exit 2; }

echo '[1/2] Exact declaration paths (inert CSS parsing; no screenshots or layout rendering)'
"$python_bin" -m framediff visual-tree-prepare \
  --rendered "$source_rendered" \
  --out "$tree_rendered" \
  --resume

echo "[2/2] Frozen detector target cache on physical GPU $cache_gpu"
for split in train val; do
  CUDA_VISIBLE_DEVICES=$cache_gpu "$python_bin" -m framediff visual-cache-targets \
    --data "$tree_rendered/policy-$split.jsonl" \
    --checkpoint "$detector" \
    --out "$shared_run/predicted-$split" \
    --device cuda \
    --resume
done

for required in predicted-train/data.jsonl predicted-val/data.jsonl; do
  [[ -s "$shared_run/$required" ]] || { echo "Preparation produced no data: $shared_run/$required" >&2; exit 1; }
done

echo 'Tree-path policy data preparation complete.'
echo "  source assets: $source_rendered"
echo "  tree labels:   $tree_rendered"
echo "  shared inputs: $shared_run"
echo "  detector:      $detector (unchanged)"

#!/usr/bin/env bash
# Rebuild labels and frozen corruption from raw source HTML, never old normalized pages.
set -euo pipefail
resume_after_images=0
case "${1:-}" in
  '') ;;
  --resume-after-images) resume_after_images=1 ;;
  -h|--help)
    echo 'Usage: bash scripts/prepare_webui_color_images.sh [--resume-after-images]'
    echo 'Resume skips image conversion, reuses rendering/normalization caches, and publishes the final 8:1:1 page split.'
    exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo 'Expected at most one option' >&2; exit 2; fi
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
source_manifest=${SOURCE_MANIFEST:-data/webui-full-dom-screened-7k-v1/7000/source/manifest.jsonl}
data=${COLOR_DATA:-data/new}
size=${DATA_SIZE:-7000}
workers=${PREPARE_WORKERS:-4}
if [[ "$resume_after_images" == 1 ]]; then
  if [[ ! -s "$data/master/source/manifest.jsonl" ]]; then
    echo "Cannot resume: missing or empty $data/master/source/manifest.jsonl" >&2
    exit 1
  fi
  echo "Reusing converted image manifest: $data/master/source/manifest.jsonl"
else
  python -m framediff.webui_placeholders --manifest "$source_manifest" --out "$data/master/source" --workers "$workers"
fi
python -m framediff visual-build-data --manifest "$data/master/source/manifest.jsonl" \
  --out "$data/master/rendered" --workers "$workers" --clean-only \
  --abstract-size 384 --seed 42 --min-elements 3 --max-source-mae 1 --resume
python -m framediff visual-tree-normalize --rendered "$data/master/rendered" \
  --out "$data/master/labels" --workers "$workers" --latest-visual-labels --clean-targets-only --final-repartition \
  --max-css-owners 512 --observation-size 384 --seed 42 --resume
python -m framediff.webui_final_split --data "$data" --size "$size" \
  --seed "${SPLIT_SEED:-42}" --workers "$workers" --timeout "${PREPARE_TIMEOUT:-180}"
echo "Preparation complete. Final 8:1:1 page split: $data/$size/coverage.json"

#!/usr/bin/env bash
# Rebuild labels and frozen corruption from raw source HTML, never old normalized pages.
set -euo pipefail
resume_after_images=0
rebuild_after_images=0
case "${1:-}" in
  '') ;;
  --resume-after-images) resume_after_images=1 ;;
  --rebuild-after-images) resume_after_images=1; rebuild_after_images=1 ;;
  -h|--help)
    echo 'Usage: bash scripts/prepare_webui_color_images.sh [--resume-after-images|--rebuild-after-images]'
    echo 'Resume skips image conversion, reuses rendering/normalization caches, and publishes the final 8:1:1 page split.'
    echo 'Rebuild preserves converted images, backs up incompatible downstream caches, and rebuilds rendering/labels before the final split.'
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
echo "WebUI preparation: output=$data, workers=$workers (progress counts page/viewport views)"
if [[ "$resume_after_images" == 1 ]]; then
  if [[ ! -s "$data/master/source/manifest.jsonl" ]]; then
    echo "Cannot resume: missing or empty $data/master/source/manifest.jsonl" >&2
    exit 1
  fi
  echo "[1/4] Image conversion skipped; reusing $data/master/source/manifest.jsonl"
else
  echo '[1/4] Image download and colour replacement'
  python -m framediff.webui_placeholders --manifest "$source_manifest" --out "$data/master/source" --workers "$workers"
fi
if [[ "$rebuild_after_images" == 1 ]]; then
  backup="$data/preparation-history/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  for stage in rendered labels css final-split-cache; do
    if [[ -e "$data/master/$stage" ]]; then
      mkdir -p "$backup"
      mv "$data/master/$stage" "$backup/$stage"
      echo "Backed up $data/master/$stage -> $backup/$stage"
    fi
  done
  echo 'Rebuilding downstream data from converted images; image downloads/conversion are skipped.'
fi
echo '[2/4] Detector rendering and label generation'
python -m framediff visual-build-data --manifest "$data/master/source/manifest.jsonl" \
  --out "$data/master/rendered" --workers "$workers" --clean-only \
  --abstract-size 384 --seed 42 --min-elements 3 --max-source-mae 1 --resume
echo '[3/4] HTML normalization and editable CSS inspection (per source split)'
python -m framediff visual-tree-normalize --rendered "$data/master/rendered" \
  --out "$data/master/labels" --workers "$workers" --latest-visual-labels --clean-targets-only --final-repartition \
  --max-css-owners 512 --observation-size 384 --seed 42 --resume
echo '[4/4] Fixed validation/test corruption and final 8:1:1 page split'
python -m framediff.webui_final_split --data "$data" --size "$size" \
  --seed "${SPLIT_SEED:-42}" --workers "$workers" --timeout "${PREPARE_TIMEOUT:-180}"
echo "Preparation complete. Final 8:1:1 page split: $data/$size/coverage.json"

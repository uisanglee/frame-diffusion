#!/usr/bin/env bash
# Rebuild labels and frozen corruption from raw source HTML, never old normalized pages.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
source_manifest=${SOURCE_MANIFEST:-data/webui-full-dom-screened-7k-v1/7000/source/manifest.jsonl}
data=${COLOR_DATA:-data/new}
size=${DATA_SIZE:-7000}
workers=${PREPARE_WORKERS:-4}
python -m framediff.webui_placeholders --manifest "$source_manifest" --out "$data/master/source"
python -m framediff visual-build-data --manifest "$data/master/source/manifest.jsonl" \
  --out "$data/master/rendered" --workers "$workers" --clean-only \
  --abstract-size 384 --seed 42 --min-elements 3 --max-source-mae 1 --resume
python -m framediff visual-tree-normalize --rendered "$data/master/rendered" \
  --out "$data/master/labels" --workers "$workers" --latest-visual-labels --clean-targets-only \
  --max-css-owners 512 --observation-size 384 --seed 42 --resume
python -m framediff.webui_final_split --data "$data" --size "$size" \
  --seed "${SPLIT_SEED:-42}" --workers "$workers"

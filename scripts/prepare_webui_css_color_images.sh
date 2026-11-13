#!/usr/bin/env bash
# CSS first -> five-color images -> new labels and fixed corruption; keep old splits.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
old=${PREVIOUS_COLOR_DATA:-data/new}
data=${CSS_COLOR_DATA:-data/new-css-v1}
size=${DATA_SIZE:-7000}
workers=${PREPARE_WORKERS:-4}
source_manifest=${SOURCE_MANIFEST:-$old/$size/source/manifest.jsonl}
# Abort before writing if the destination overlaps the previous tree.
python - "$old" "$data" <<'PY'
import sys
from pathlib import Path
old,new=map(lambda x:Path(x).resolve(),sys.argv[1:])
if old==new or old in new.parents or new in old.parents:
    raise SystemExit('CSS_COLOR_DATA must be separate from PREVIOUS_COLOR_DATA')
PY
mkdir -p "$data"
# The server runs Linux. Prevent simultaneous rebuilds of one version directory.
exec 9>"$data/.css-prepare.lock"
flock -n 9 || { echo 'This output is already being prepared' >&2; exit 2; }
echo "CSS-first preparation: source=$source_manifest output=$data workers=$workers"
echo '[1/4] Archived CSS applied BEFORE image sizing/color replacement'
python -m framediff.webui_css_prepare --manifest "$source_manifest" --out "$data/master/source" \
  --workers "$workers" --image-cache "$old/master/source/downloads" \
  --css-cache "${CSS_DOWNLOAD_CACHE:-runs/webui-css-replay10-v1/css-cache/downloads}"
echo '[2/4] New target rendering and Detector labels'
python -m framediff visual-build-data --manifest "$data/master/source/manifest.jsonl" \
  --out "$data/master/rendered" --workers "$workers" --clean-only \
  --abstract-size 384 --seed 42 --min-elements 3 --max-source-mae 1 --resume
echo '[3/4] New normalization and editable CSS inspection'
python -m framediff visual-tree-normalize --rendered "$data/master/rendered" \
  --out "$data/master/labels" --workers "$workers" --latest-visual-labels --clean-targets-only --final-repartition \
  --max-css-owners 512 --observation-size 384 --seed 42 --resume
echo '[4/4] Fixed validation/test corruption; KEEP previous page splits'
python -m framediff.webui_final_split --data "$data" --size "$size" --seed 42 \
  --workers "$workers" --timeout "${PREPARE_TIMEOUT:-180}" \
  --preserve-splits-from "$data/master/source/preserved-inputs.jsonl"
echo "Complete: $data/$size/css/policy-{train,val,test}.jsonl"
echo "Coverage and exclusions: $data/split-report.json"

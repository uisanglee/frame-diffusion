#!/usr/bin/env bash
# Evaluate an HTML generated outside this repository against its target screenshot.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo 'Usage: bash scripts/run_external_html.sh TARGET_SCREENSHOT INITIAL_HTML OUTPUT_ROOT TRAIN_RUN_DIR [REFERENCE_HTML]' >&2
  exit 2
fi
screenshot=$1;initial=$2;output=$3;train=$4;reference=${5:-}
options=(--screenshot "$screenshot" --initial-html "$initial" --out "$output/input"
  --id "${PAGE_ID:-external-page}" --source-model "${SOURCE_MODEL:-external-vlm}" --resume)
if [[ -n "$reference" ]]; then options+=(--reference-html "$reference"); fi
python -m framediff web-external-manifest "${options[@]}"
official=${OFFICIAL_REPO:-}
if [[ -z "$reference" && -n "$official" ]]; then
  echo 'No REFERENCE_HTML: disabling official Design2Code evaluator; pixel MAE remains available.' >&2
  official=
fi
PAGE_MANIFEST="$output/input/manifest.jsonl" PAGE_LIMIT=0 WEB_DATASET=design2code \
SELF_REVISION_PROTOCOL=none ABSTRACT_VLM_REVISION=0 OFFICIAL_REPO="$official" \
bash scripts/run_visual_web.sh . "$output" "$train"

#!/usr/bin/env bash
# Regenerate only frozen validation/test corruptions from existing normalized targets.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
data=${CSS_COLOR_DATA:-data/new-css-v1}
size=${DATA_SIZE:-7000}
preserved="$data/master/source/preserved-inputs.jsonl"
[[ -f "$preserved" && -f "$data/master/labels/policy-train.jsonl" ]] || {
  echo 'Finish CSS-first normalization first: bash scripts/prepare_webui_css_color_images.sh' >&2
  exit 2
}
exec 9>"$data/.css-prepare.lock"
flock -n 9 || { echo 'This preprocessing output is already in use' >&2; exit 2; }
python -m framediff.webui_final_split --data "$data" --size "$size" --seed 42 \
  --workers "${PREPARE_WORKERS:-4}" --timeout "${PREPARE_TIMEOUT:-900}" \
  --preserve-splits-from "$preserved"
echo "Mixed corruption reports: $data/$size/css/corruption-{val,test}.jsonl"
echo "Summary: $data/$size/css/freeze-report.json"

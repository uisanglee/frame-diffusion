#!/usr/bin/env bash
# Sequential execution on one GPU; trained checkpoints remain fixed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ $# == 5 ]] || { echo 'Usage: run_paper_matrix.sh WEBUI_TEST_MANIFEST D2C_ROOT D2C_HARD_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2; exit 2; }
webui=$1; d2c=$2; hard=$3; output=$4; train=$5
: "${OFFICIAL_REPO:?Set OFFICIAL_REPO}"
export INITIAL_SEEDS=${INITIAL_SEEDS:-11,22,33} REPEATS=1 RUN_UI2CODEN=1 RUN_ABLATION=0
# Validate every dataset before starting any generation.
python - "$webui" "$d2c" "$hard" <<'PY'
import sys
from framediff.paper_matrix import check_sources
for name, options in [('WebUI', dict(manifest=sys.argv[1], dataset='webui')),
                      ('Design2Code', dict(root=sys.argv[2])),
                      ('Design2Code-Hard', dict(root=sys.argv[3]))]:
    print(f'Checking {name}: {options}', flush=True)
    rows = check_sources(**options)
    print(f'{name}: {len(rows)} records OK', flush=True)
PY
WEB_DATASET=webui PAGE_MANIFEST="$webui" bash scripts/run_paper_4090_vlms.sh . "$output/webui" "$train"
WEB_DATASET=design2code PAGE_MANIFEST='' bash scripts/run_paper_4090_vlms.sh "$d2c" "$output/design2code" "$train"
WEB_DATASET=design2code PAGE_MANIFEST='' bash scripts/run_paper_4090_vlms.sh "$hard" "$output/design2code-hard" "$train"
python -m framediff.paper_matrix export --root "$output" --out "$output/table"

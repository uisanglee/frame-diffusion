#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
[[ $# -le 4 ]] || { echo 'Expected at most DATA DETECTOR SHARED_RUN POLICY_RUN' >&2; exit 2; }
export CSS_STYLESHEETS=1 ONLINE_CORRUPTION=1
exec bash scripts/train_numeric_policy_gpu3.sh \
  "${1:-data/webui-css-owners-v8-selective-size-margin}" \
  "${2:-runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt}" \
  "${3:-runs/css-owners-v8-selective-size-margin-shared}" "${4:-runs/css-owners-online-v8-selective-size-margin-abstract}"

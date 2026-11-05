#!/usr/bin/env bash
# GPU=2 bash baselines/ui2coden/run_4090.sh INPUT.jsonl OUTPUT_DIR
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
cd "$root"
[[ $# == 2 ]] || { echo 'Usage: run_4090.sh INPUT.jsonl OUTPUT_DIR' >&2; exit 2; }
source "$root/scripts/experiment_log.sh"
experiment_log_init "$2"
python_bin="$here/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Run setup_4090.sh first.' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
options=(--precision "${UI2CODEN_PRECISION:-4bit}" --manifest "$1" --out "$2" --resume --rounds "${ROUNDS:-3}"
  --limit "${LIMIT:-0}" --max-new-tokens "${MAX_NEW_TOKENS:-16384}"
  --max-context "${MAX_CONTEXT:-0}" --max-pixels "${MAX_PIXELS:-262144}"
  --revision "${MODEL_REVISION:-main}" --protocol "${PROTOCOL:-simple}")
# Reject contradictory global budgets once, before model loading or cached failures.
"$python_bin" - "${MAX_NEW_TOKENS:-16384}" "${MAX_CONTEXT:-0}" <<'PY'
import sys
output, context = map(int, sys.argv[1:])
if output <= 0 or context < 0 or (context and output >= context):
    raise SystemExit('UI2CodeN requires positive MAX_NEW_TOKENS; MAX_CONTEXT=0 uses model limit, otherwise must exceed output budget.')
PY
# No GPU workers/prefetch. The model process exits before metric models are loaded.
run_logged "UI2CodeN polishing (rounds=${ROUNDS:-3}, cached rounds reused)" "$python_bin" "$here/runner.py" "${options[@]}"
evaluation=(--data "$2/results.jsonl" --out "$2/evaluation" --resume
  --max-nodes "${MAX_NODES:-2048}")
if [[ -n ${OFFICIAL_REPO:-} ]]; then evaluation+=(--official-repo "$OFFICIAL_REPO"); fi
run_logged "UI2CodeN quality evaluation" "${EVALUATION_PYTHON:-$python_bin}" -m framediff web-evaluate "${evaluation[@]}"

#!/usr/bin/env bash
# GPU=2 bash baselines/ui2coden/run_4090.sh INPUT.jsonl OUTPUT_DIR
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
cd "$root"
[[ $# == 2 ]] || { echo 'Usage: run_4090.sh INPUT.jsonl OUTPUT_DIR' >&2; exit 2; }
python_bin="$here/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Run setup_4090.sh first.' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
options=(--manifest "$1" --out "$2" --resume --rounds "${ROUNDS:-3}"
  --limit "${LIMIT:-0}" --max-new-tokens "${MAX_NEW_TOKENS:-4096}"
  --max-context "${MAX_CONTEXT:-12288}" --max-pixels "${MAX_PIXELS:-262144}"
  --revision "${MODEL_REVISION:-main}" --protocol "${PROTOCOL:-layout}")
# No GPU workers/prefetch. The model process exits before metric models are loaded.
"$python_bin" "$here/runner.py" "${options[@]}"
evaluation=(--data "$2/results.jsonl" --out "$2/evaluation" --resume
  --max-nodes "${MAX_NODES:-2048}")
if [[ -n ${OFFICIAL_REPO:-} ]]; then evaluation+=(--official-repo "$OFFICIAL_REPO"); fi
"$python_bin" -m framediff web-evaluate "${evaluation[@]}"

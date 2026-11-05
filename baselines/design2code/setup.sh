#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
repo=${DESIGN2CODE_REPO:-$root/external/design2code-official}
[[ -d "$repo/CogVLM" ]] && repo="$repo/CogVLM"
if [[ ! -f "$repo/requirements.txt" || ! -d "$repo/utils/models" ]]; then
  echo 'DESIGN2CODE_REPO must point to a cloned NoviScl/Design2Code repository (including CogVLM).' >&2
  exit 2
fi
bootstrap=${DESIGN2CODE_BOOTSTRAP_PYTHON:-python3.10}
"$bootstrap" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,12), "Use Python 3.10 or 3.11 for this legacy runtime"'
venv="$root/baselines/design2code/.venv"
[[ -x "$venv/bin/python" ]] || "$bootstrap" -m venv "$venv"
python="$venv/bin/python"
"$python" -m pip install --upgrade pip 'setuptools<70' wheel
"$python" -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
DS_BUILD_OPS=0 "$python" -m pip install \
  -c "$root/baselines/design2code/constraints.txt" \
  -r "$repo/requirements.txt" 'SwissArmyTransformer==0.4.10' sentencepiece
"$python" -m pip check
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "$python" - <<'PY'
from importlib.metadata import version
import torch
from utils.models import FineTuneTestCogAgentModel
from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
print('SAT:', version('SwissArmyTransformer'), 'torch:', torch.__version__)
print('Official Design2Code model imports OK. Model weights are downloaded at first inference.')
PY
printf 'DESIGN2CODE_PYTHON=%s\n' "$python"

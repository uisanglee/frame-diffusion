#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -s) == Linux ]] || { echo 'This installer targets Linux + NVIDIA CUDA.' >&2; exit 2; }
"${PYTHON_BIN:-python3}" -m venv "$here/.venv"
"$here/.venv/bin/python" -m pip install -r "$here/requirements.txt"
"$here/.venv/bin/python" -m playwright install chromium
"$here/.venv/bin/python" -m pip freeze > "$here/.venv/installed-requirements.txt"
"$here/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'

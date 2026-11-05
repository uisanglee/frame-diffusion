#!/usr/bin/env bash
# Separate compatibility environment for the legacy custom VMistral checkpoint.
# Version pins below are our integration environment, NOT official paper settings.
set -euo pipefail
cd "$(dirname "$0")/../.."
python_bin="${WEBSIGHT_SETUP_PYTHON:-python3}"
venv="${WEBSIGHT_VENV:-$PWD/baselines/websight/.venv}"
"$python_bin" -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install 'torch==2.2.2' 'torchvision==0.17.2' --index-url https://download.pytorch.org/whl/cu121
"$venv/bin/python" -m pip install 'numpy<2' 'transformers==4.37.2' 'accelerate==0.27.2' \
  'einops==0.7.0' 'sentencepiece==0.2.0' 'Pillow>=10,<12' safetensors
"$venv/bin/python" -c 'import torch, transformers; from transformers import AutoModelForCausalLM, AutoProcessor; print("WebSight environment:", torch.__version__, transformers.__version__, "CUDA:", torch.cuda.is_available())'
echo "Set WEBSIGHT_PYTHON=$venv/bin/python (the paper launchers use this location by default)."
echo 'Environment setup does not verify checkpoint loading; the first model run performs that check.'

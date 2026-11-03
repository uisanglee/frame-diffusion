#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -s) == Linux ]] || { echo 'This installer targets Linux + NVIDIA CUDA.' >&2; exit 2; }
"${PYTHON_BIN:-python3}" -m venv "$here/.venv"
# Pin CUDA wheels too: bitsandbytes 0.48 has no CUDA 13.2 backend.
"$here/.venv/bin/python" -m pip install \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$here/.venv/bin/python" -m pip install -r "$here/requirements.txt"
"$here/.venv/bin/python" - <<'PY'
import os
import torch

assert torch.version.cuda == '12.8', f'Unexpected PyTorch CUDA: {torch.version.cuda}'
assert os.environ.get('BNB_CUDA_VERSION', '') in ('', '128'), 'Unset BNB_CUDA_VERSION before installing/running this baseline'
assert torch.cuda.is_available(), 'CUDA unavailable'
import bitsandbytes as bnb

# Exercise the actual NF4 kernel before downloading/loading model weights.
x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
packed, state = bnb.functional.quantize_4bit(x, quant_type='nf4')
restored = bnb.functional.dequantize_4bit(packed, state)
torch.cuda.synchronize()
assert restored.shape == x.shape and torch.isfinite(restored).all().item()
print({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
       'cuda': torch.version.cuda, 'bitsandbytes': bnb.__version__, 'nf4_check': 'passed'})
PY
"$here/.venv/bin/python" -m playwright install chromium
"$here/.venv/bin/python" -m pip freeze > "$here/.venv/installed-requirements.txt"

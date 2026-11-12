#!/usr/bin/env bash
# Full Design2Code + Hard baseline evaluation, retaining Initials for later TUIDE.
# D2C_BASELINE_PAGES=3 or 10 selects a fixed pilot; 0 (default) selects all pages.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export D2C_PILOT_PAGES=${D2C_BASELINE_PAGES:-0}
export D2C_PILOT_OUTPUT=${D2C_BASELINE_OUTPUT:-runs/d2c-vlm-baselines-seed42-n2-v1}
# Delegate to the same generation, resource preservation, resume and metrics
# pipeline used by the pilot. It forces RUN_TUIDE=0; no checkpoints are needed.
echo "Design2Code + Hard baselines: pages/dataset=$D2C_PILOT_PAGES (0=all)"
echo "Initial HTML/assets and prepared.jsonl are retained under $D2C_PILOT_OUTPUT"
echo 'Seed=42; Self Revision/Polishing N=2; output budget=4096; TUIDE deferred.'
exec bash scripts/run_d2c_pilot10_bf16_2gpu.sh

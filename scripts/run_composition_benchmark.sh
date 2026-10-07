#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
# Usage: bash scripts/run_composition_benchmark.sh ABSTRACT.pt DETECTOR.pt [RGB.pt]
ABSTRACT=${1:?Supply abstract policy best.pt}
DETECTOR=${2:?Supply frozen detector best.pt}
RGB=${3:-}
DATA=${COMPOSITION_DATA:-data/composition-v1}
OUT=${COMPOSITION_OUT:-runs/composition-v1}
for checkpoint in "$ABSTRACT" "$DETECTOR"; do
  test -f "$checkpoint" || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
done
if [[ -n "$RGB" && ! -f "$RGB" ]]; then echo "Missing checkpoint: $RGB" >&2; exit 1; fi
python -m framediff visual-composition-build --out "$DATA" \
  --samples-per-cell "${SAMPLES_PER_CELL:-24}" --seed "${COMPOSITION_SEED:-73129}" --resume
COMMON=(--data "$DATA/manifest.jsonl" --steps "${REPAIR_STEPS:-20}" --device "${DEVICE:-cuda}" --resume)
python -m framediff visual-composition-evaluate "${COMMON[@]}" \
  --checkpoint "$ABSTRACT" --detector-checkpoint "$DETECTOR" --out "$OUT/abstract"
python -m framediff visual-composition-evaluate "${COMMON[@]}" \
  --checkpoint "$ABSTRACT" --oracle --out "$OUT/abstract-oracle"
if [[ -n "$RGB" ]]; then
  python -m framediff visual-composition-evaluate "${COMMON[@]}" \
    --checkpoint "$RGB" --out "$OUT/screenshot"
fi
echo "Reports: $OUT/{abstract,abstract-oracle,screenshot}/report.md"

#!/usr/bin/env bash
# One prepared page set, one S RGB baseline, sequential S/M/L abstract rollouts.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
[[ $# == 2 ]] || { echo 'Usage: bash scripts/run_tuide_scales.sh PREPARED_JSONL OUTPUT_ROOT' >&2; exit 2; }
data=$1;output=$2
detector=${DETECTOR_CHECKPOINT:-runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt}
raw=${RAW_CHECKPOINT:-runs/tuide-selective-size-margin-s-screenshot/replacement/raw-stage2/best.pt}
s=${TUIDE_S_CHECKPOINT:-runs/tuide-selective-size-margin-s-abstract/replacement/abstract-stage2/best.pt}
m=${TUIDE_M_CHECKPOINT:-runs/tuide-selective-size-margin-m-abstract/replacement/abstract-stage2/best.pt}
l=${TUIDE_L_CHECKPOINT:-runs/tuide-selective-size-margin-l-abstract/replacement/abstract-stage2/best.pt}
for path in "$data" "$detector" "$raw" "$s" "$m" "$l"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
metrics=()
[[ -n "${OFFICIAL_REPO:-}" ]] && metrics+=(--official-repo "$OFFICIAL_REPO")
for scale in s m l; do
  case "$scale" in
    s) checkpoint=$s;options=(--raw-checkpoint "$raw");;
    m) checkpoint=$m;options=(--abstract-only);;
    l) checkpoint=$l;options=(--abstract-only);;
  esac
  python -m framediff visual-evaluate --data "$data" --abstract-checkpoint "$checkpoint" \
    --detector-checkpoint "$detector" "${options[@]}" --out "$output/$scale/repair" \
    --device "${DEVICE:-cuda}" --steps "${REPAIR_STEPS:-20}" --repeats "${REPEATS:-3}" \
    --limit "${PAGE_LIMIT:-0}" --seed "${SEED:-42}" --time-budget "${TIME_BUDGET:-0}" --resume
  python -m framediff web-evaluate --data "$output/$scale/repair/results.jsonl" \
    --out "$output/$scale/evaluation" "${metrics[@]}" --resume
done
python -m framediff.scale_report --root "$output"

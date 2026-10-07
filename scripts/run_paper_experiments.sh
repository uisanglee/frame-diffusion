#!/usr/bin/env bash
# Main multi-VLM comparison plus one representative-VLM ablation.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_paper_experiments.sh D2C_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2
  exit 2
fi
root=$1;output=$2;train=$3
specs=${VLM_SPECS:-qwen3vl8b\|qwen\|Qwen/Qwen3-VL-8B-Instruct\|bf16}
IFS=',' read -ra entries <<< "$specs"
first=${entries[0]};first_alias=${first%%|*};[[ "$first_alias" == "$first" ]] && first_alias=${first%%=*}
representative=${ABLATION_VLM_ALIAS:-$first_alias}
aliases=()
for entry in "${entries[@]}"; do
  if [[ "$entry" == *'|'* ]]; then
    IFS='|' read -r alias backend model precision endpoint <<< "$entry"
  else
    alias=${entry%%=*};model=${entry#*=};backend=${VLM_BACKEND:-qwen}
    precision=${VLM_PRECISION:-4bit};endpoint=${VLM_ENDPOINT:-http://localhost:8000/v1/chat/completions}
  fi
  [[ -n "$alias" && -n "$backend" && -n "$model" ]] || { echo "Invalid VLM spec: $entry" >&2; exit 2; }
  [[ "$backend" == qwen || "$backend" == hf || "$backend" == openai-compatible ]] || { echo "Invalid backend in: $entry" >&2; exit 2; }
  case "$precision" in bf16) four_bit=0;; 4bit) four_bit=1;; server) four_bit=0;; *) echo "Precision must be bf16, 4bit, or server: $entry" >&2;exit 2;; esac
  endpoint=${endpoint:-http://localhost:8000/v1/chat/completions}
  aliases+=("$alias")
  abstract=0;[[ "$alias" == "$representative" ]] && abstract=1
  echo "Running $alias ($backend, $model, $precision), abstract ablation=$abstract"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WEB_DATASET=design2code SELF_REVISION_PROTOCOL=design2code \
    ABSTRACT_VLM_REVISION=$abstract VLM_BACKEND="$backend" VLM_MODEL="$model" \
    VLM_ENDPOINT="$endpoint" VLM_FOUR_BIT="$four_bit" \
    bash scripts/run_visual_web.sh "$root" "$output/$alias" "$train"
done
aliases_csv=$(IFS=,; echo "${aliases[*]}")
python -m framediff.paper_suite --root "$output" --aliases "$aliases_csv" \
  --representative "$representative" --out "$output/paper"
echo "Completed ${#entries[@]} VLM(s). Paper tables: $output/paper/README.md"

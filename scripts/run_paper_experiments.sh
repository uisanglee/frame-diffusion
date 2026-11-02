#!/usr/bin/env bash
# Main multi-VLM comparison plus an optional representative-VLM ablation.
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
run_ablation=${RUN_ABLATION:-0}
run_ui2coden=${RUN_UI2CODEN:-0}
[[ "$run_ablation" == 0 || "$run_ablation" == 1 ]] || { echo 'RUN_ABLATION must be 0 or 1' >&2; exit 2; }
[[ "$run_ui2coden" == 0 || "$run_ui2coden" == 1 ]] || { echo 'RUN_UI2CODEN must be 0 or 1' >&2; exit 2; }
[[ "${TARGET_IMAGE_SOURCE:-reference-render}" == reference-render ]] || { echo 'Paper runs require reference-render targets' >&2; exit 2; }
: "${OFFICIAL_REPO:?Set OFFICIAL_REPO to the official Design2Code repository}"
source_check=(--root "$root")
[[ -n "${PAGE_MANIFEST:-}" ]] && source_check=(--manifest "$PAGE_MANIFEST")
python -m framediff.paper_matrix check "${source_check[@]}" --dataset "${WEB_DATASET:-design2code}" \
  --webui-view "${WEBUI_VIEW:-default_1280-720}" --limit "${PAGE_LIMIT:-0}" --official-repo "$OFFICIAL_REPO"
evaluation_python=$(command -v python)
ui_revision=${MODEL_REVISION:-main}
if [[ "$run_ui2coden" == 1 ]]; then
  [[ -x baselines/ui2coden/.venv/bin/python ]] || { echo 'Run baselines/ui2coden/setup_4090.sh first' >&2; exit 2; }
  ui_revision=$(python - "$ui_revision" "$output/ui2coden-weights" <<'PYREV'
import sys
from framediff.vlm import pin_revision
print(pin_revision('zai-org/UI2Code_N', *sys.argv[1:]))
PYREV
  )
fi
export REPEATS=${REPEATS:-1}
export GENERATE_INITIAL=1
abstract_only=1
[[ "$run_ablation" == 1 ]] && abstract_only=0
aliases=()
shared_target_manifest=
initial_seeds=${INITIAL_SEEDS:-2024}
IFS=',' read -ra seed_entries <<< "$initial_seeds"
[[ ${#seed_entries[@]} -ge 1 ]] || { echo 'INITIAL_SEEDS must list at least one seed' >&2; exit 2; }
seen_seeds=','
for seed in "${seed_entries[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ && "$seen_seeds" != *",$seed,"* ]] || { echo "Invalid or duplicate initial seed: $seed" >&2; exit 2; }
  seen_seeds+="$seed,"
done
if [[ ${#seed_entries[@]} -gt 1 ]]; then
  initial_temperature=${INITIAL_TEMPERATURE:-0.7}
  [[ "$initial_temperature" != 0 && "$initial_temperature" != 0.0 ]] || { echo 'Multiple initial seeds require sampling; set INITIAL_TEMPERATURE above zero' >&2; exit 2; }
else
  initial_temperature=${INITIAL_TEMPERATURE:-0}
fi
for entry in "${entries[@]}"; do
  revision=${VLM_REVISION:-main}
  if [[ "$entry" == *'|'* ]]; then
    IFS='|' read -r alias backend model precision endpoint api_key_env reasoning_effort revision <<< "$entry"
  else
    alias=${entry%%=*};model=${entry#*=};backend=${VLM_BACKEND:-qwen}
    precision=${VLM_PRECISION:-4bit};endpoint=${VLM_ENDPOINT:-http://localhost:8000/v1/chat/completions}
    api_key_env=${VLM_API_KEY_ENV:-VLM_API_KEY};reasoning_effort=${VLM_REASONING_EFFORT:-}
  fi
  [[ -n "$alias" && -n "$backend" && -n "$model" ]] || { echo "Invalid VLM spec: $entry" >&2; exit 2; }
  [[ "$backend" == qwen || "$backend" == hf || "$backend" == openai-compatible ]] || { echo "Invalid backend in: $entry" >&2; exit 2; }
  case "$precision" in bf16) four_bit=0;; 4bit) four_bit=1;; server) four_bit=0;; *) echo "Precision must be bf16, 4bit, or server: $entry" >&2;exit 2;; esac
  endpoint=${endpoint:-http://localhost:8000/v1/chat/completions}
  api_key_env=${api_key_env:-${VLM_API_KEY_ENV:-VLM_API_KEY}}
  revision=${revision:-${VLM_REVISION:-main}}
  rounds=3
  if [[ "$alias" == ui2coden ]]; then
    [[ "$run_ui2coden" == 1 && "$model" == zai-org/UI2Code_N ]] || { echo 'ui2coden group requires its own model and RUN_UI2CODEN=1' >&2; exit 2; }
    revision=$ui_revision
    rounds=0
  fi
  if [[ "$backend" != openai-compatible ]]; then
    revision=$(python - "$model" "$revision" "$output/$alias" <<'PYREV'
import sys
from framediff.vlm import pin_revision
print(pin_revision(*sys.argv[1:]))
PYREV
    )
  fi
  aliases+=("$alias")
  for seed in "${seed_entries[@]}"; do
    seed_output="$output/$alias/seeds/$seed"
    abstract=0
    [[ "$run_ablation" == 1 && "$alias" == "$representative" && "$seed" == "${seed_entries[0]}" ]] && abstract=1
    echo "Running $alias initial seed=$seed ($backend, $model, $precision), abstract ablation=$abstract"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WEB_DATASET="${WEB_DATASET:-design2code}" SELF_REVISION_PROTOCOL=design2code REVISION_ROUNDS="$rounds" \
      TARGET_IMAGE_SOURCE="${TARGET_IMAGE_SOURCE:-reference-render}" REFERENCE_MANIFEST="$shared_target_manifest" \
      INITIAL_TEMPERATURE="$initial_temperature" SEED="$seed" \
      ABSTRACT_ONLY="$abstract_only" ABSTRACT_VLM_REVISION=$abstract VLM_BACKEND="$backend" VLM_MODEL="$model" \
      VLM_ENDPOINT="$endpoint" VLM_API_KEY_ENV="$api_key_env" VLM_REASONING_EFFORT="$reasoning_effort" \
      VLM_FOUR_BIT="$four_bit" VLM_REVISION="$revision" \
      bash scripts/run_visual_web.sh "$root" "$seed_output" "$train"
    [[ -n "$shared_target_manifest" ]] || shared_target_manifest="$seed_output/prepare/prepared.jsonl"
    if [[ "$run_ui2coden" == 1 ]]; then
      GPU=${CUDA_VISIBLE_DEVICES:-0} ROUNDS=3 LIMIT=0 MODEL_REVISION="$ui_revision" EVALUATION_PYTHON="$evaluation_python" \
        bash baselines/ui2coden/run_4090.sh "$seed_output/prepare/prepared.jsonl" "$seed_output/ui2coden"
    fi
  done
done
aliases_csv=$(IFS=,; echo "${aliases[*]}")
suite_options=(--root "$output" --aliases "$aliases_csv" --representative "$representative" --out "$output/paper" --require-complete-metrics --own-initial-alias ui2coden --require-three-rounds --initial-seeds "$initial_seeds")
[[ "$run_ablation" == 0 ]] && suite_options+=(--skip-ablation)
[[ "$run_ui2coden" == 1 ]] && suite_options+=(--include-ui2coden --ui2coden-rounds 3)
python -m framediff.paper_suite "${suite_options[@]}"
echo "Completed ${#entries[@]} VLM(s). Paper tables: $output/paper/README.md"

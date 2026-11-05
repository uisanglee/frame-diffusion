#!/usr/bin/env bash
# Main multi-VLM comparison plus an optional representative-VLM ablation.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_paper_experiments.sh D2C_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2
  exit 2
fi
root=$1;output=$2;train=$3
source "$(dirname "${BASH_SOURCE[0]}")/experiment_log.sh"
experiment_log_init "$output"
experiment_banner START "Dataset experiment"
# Dataset protocol takes precedence over old shell exports. The Hard dataset
# also uses WEB_DATASET=design2code and is distinguished by its input/output root.
case "${WEB_DATASET:-design2code}" in
  webui) output_budget=16384 ;;
  design2code) output_budget=4096 ;;
  *) echo 'WEB_DATASET must be webui or design2code (including Design2Code-Hard)' >&2; exit 2 ;;
esac
if [[ "${MAX_NEW_TOKENS:-$output_budget}" != "$output_budget" || "${UI2CODEN_MAX_NEW_TOKENS:-$output_budget}" != "$output_budget" ]]; then
  echo "Using dataset output budget=$output_budget; overriding inherited token-budget variables."
fi
export MAX_NEW_TOKENS=$output_budget
export UI2CODEN_MAX_NEW_TOKENS=$MAX_NEW_TOKENS
echo "Dataset=${WEB_DATASET:-design2code}, Initial/Self Revision/UI2CodeN max_new_tokens=$MAX_NEW_TOKENS"
specs=${VLM_SPECS:-'llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|bf16,glm4v-9b-thinking|hf|zai-org/GLM-4.1V-9B-Thinking|bf16,websight-8b|websight|HuggingFaceM4/VLM_WebSight_finetuned|bf16,design2code-18b|design2code-sat|SALT-NLP/Design2Code-18B-v0|bf16,ui2coden|hf|zai-org/UI2Code_N|bf16'}
IFS=',' read -ra entries <<< "$specs"
# Fail before generating any pages if a selected legacy runtime is missing.
for selected_entry in "${entries[@]}"; do
  IFS='|' read -r selected_alias selected_backend _ <<< "$selected_entry"
  case "$selected_backend" in
    websight)
      export WEBSIGHT_PYTHON=${WEBSIGHT_PYTHON:-$PWD/baselines/websight/.venv/bin/python}
      [[ -x "$WEBSIGHT_PYTHON" ]] || { echo 'Run bash baselines/websight/setup.sh first (or set WEBSIGHT_PYTHON)' >&2; exit 2; } ;;
    design2code-sat)
      export DESIGN2CODE_PYTHON=${DESIGN2CODE_PYTHON:-$PWD/baselines/design2code/.venv/bin/python}
      [[ -x "$DESIGN2CODE_PYTHON" ]] || { echo 'Run bash baselines/design2code/setup.sh first (or set DESIGN2CODE_PYTHON)' >&2; exit 2; }
      IFS=',' read -ra sat_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
      [[ ${#sat_gpus[@]} == 2 && -n "${sat_gpus[0]}" && -n "${sat_gpus[1]}" && "${sat_gpus[0]}" != "${sat_gpus[1]}" ]] || {
        echo 'Design2Code-18B official model-parallel runner requires two distinct GPUs; set CUDA_VISIBLE_DEVICES=0,1' >&2; exit 2;
      } ;;
  esac
done
first=${entries[0]};first_alias=${first%%|*};[[ "$first_alias" == "$first" ]] && first_alias=${first%%=*}
representative=${ABLATION_VLM_ALIAS:-$first_alias}
revision_rounds=${REVISION_ROUNDS:-3}
[[ "$revision_rounds" =~ ^[1-9][0-9]*$ ]] || { echo "REVISION_ROUNDS must be positive" >&2; exit 2; }
run_ablation=${RUN_ABLATION:-0}
run_ui2coden=${RUN_UI2CODEN:-1}
run_tuide=${RUN_TUIDE:-1}
[[ "$run_tuide" == 0 || "$run_tuide" == 1 ]] || { echo 'RUN_TUIDE must be 0 or 1' >&2; exit 2; }
[[ "$run_tuide" == 1 || "$run_ablation" == 0 ]] || { echo 'RUN_TUIDE=0 requires RUN_ABLATION=0' >&2; exit 2; }
export RUN_TUIDE="$run_tuide"
[[ "$run_ablation" == 0 || "$run_ablation" == 1 ]] || { echo 'RUN_ABLATION must be 0 or 1' >&2; exit 2; }
[[ "$run_ui2coden" == 0 || "$run_ui2coden" == 1 ]] || { echo 'RUN_UI2CODEN must be 0 or 1' >&2; exit 2; }
[[ "${TARGET_IMAGE_SOURCE:-reference-render}" == reference-render ]] || { echo 'Paper runs require reference-render targets' >&2; exit 2; }
: "${OFFICIAL_REPO:?Set OFFICIAL_REPO to the official Design2Code repository}"
source_check=(--root "$root")
[[ -n "${PAGE_MANIFEST:-}" ]] && source_check=(--manifest "$PAGE_MANIFEST")
run_logged "Input and metric checks" python -m framediff.paper_matrix check "${source_check[@]}" --dataset "${WEB_DATASET:-design2code}" \
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
initial_seeds=${INITIAL_SEEDS:-42}
IFS=',' read -ra seed_entries <<< "$initial_seeds"
[[ ${#seed_entries[@]} -ge 1 ]] || { echo 'INITIAL_SEEDS must list at least one seed' >&2; exit 2; }
seen_seeds=','
for seed in "${seed_entries[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ && "$seen_seeds" != *",$seed,"* ]] || { echo "Invalid or duplicate initial seed: $seed" >&2; exit 2; }
  seen_seeds+="$seed,"
done
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
  [[ "$backend" == qwen || "$backend" == hf || "$backend" == openai-compatible || "$backend" == websight || "$backend" == design2code-sat ]] || { echo "Invalid backend in: $entry" >&2; exit 2; }
  temperature_request=${INITIAL_TEMPERATURE:-auto}
  if [[ "$backend" == openai-compatible && "$temperature_request" == auto ]]; then
    temperature_request=0
    [[ ${#seed_entries[@]} -le 1 ]] || temperature_request=0.7
  fi
  initial_temperature=$(python - "$model" "$temperature_request" "${#seed_entries[@]}" <<'PYTEMP'
import sys
from framediff.vlm import initial_temperature
value = initial_temperature(sys.argv[1], sys.argv[2])
if int(sys.argv[3]) > 1 and value == 0:
    raise SystemExit('This model defaults to greedy generation. Use one initial seed, or explicitly set INITIAL_TEMPERATURE above zero for a sampling experiment.')
print(value)
PYTEMP
  )
  case "$precision" in bf16) four_bit=0;; 4bit) four_bit=1;; server) four_bit=0;; *) echo "Precision must be bf16, 4bit, or server: $entry" >&2;exit 2;; esac
  endpoint=${endpoint:-http://localhost:8000/v1/chat/completions}
  api_key_env=${api_key_env:-${VLM_API_KEY_ENV:-VLM_API_KEY}}
  revision=${revision:-${VLM_REVISION:-main}}
  rounds=$revision_rounds
  self_revision_protocol=design2code
  if [[ "$backend" == websight || "$backend" == design2code-sat ]]; then
    rounds=0
    self_revision_protocol=none
  fi
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
    export EXPERIMENT_MODEL_LABEL="$alias" EXPERIMENT_SEED_LABEL="$seed"
    EXPERIMENT_ACTIVE_STAGE="Model experiment: $alias"
    experiment_banner START "Model experiment"
    echo "Running $alias initial seed=$seed temperature=$initial_temperature ($backend, $model, $precision), max_new_tokens=$MAX_NEW_TOKENS, abstract ablation=$abstract"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WEB_DATASET="${WEB_DATASET:-design2code}" SELF_REVISION_PROTOCOL="$self_revision_protocol" REVISION_ROUNDS="$rounds" VLM_RETRIES="${VLM_RETRIES:-0}" \
      TARGET_IMAGE_SOURCE="${TARGET_IMAGE_SOURCE:-reference-render}" REFERENCE_MANIFEST="$shared_target_manifest" \
      INITIAL_TEMPERATURE="$initial_temperature" SEED="$seed" \
      ABSTRACT_ONLY="$abstract_only" ABSTRACT_VLM_REVISION=$abstract VLM_BACKEND="$backend" VLM_MODEL="$model" \
      VLM_ENDPOINT="$endpoint" VLM_API_KEY_ENV="$api_key_env" VLM_REASONING_EFFORT="$reasoning_effort" \
      VLM_FOUR_BIT="$four_bit" VLM_REVISION="$revision" \
      bash scripts/run_visual_web.sh "$root" "$seed_output" "$train"
    [[ -n "$shared_target_manifest" ]] || shared_target_manifest="$seed_output/prepare/prepared.jsonl"
    if [[ "$run_ui2coden" == 1 ]]; then
      MAX_NEW_TOKENS="$MAX_NEW_TOKENS" MAX_CONTEXT="${UI2CODEN_MAX_CONTEXT:-0}" \
      GPU=${CUDA_VISIBLE_DEVICES:-0} ROUNDS="$revision_rounds" LIMIT=0 MODEL_REVISION="$ui_revision" EVALUATION_PYTHON="$evaluation_python" \
        bash baselines/ui2coden/run_4090.sh "$seed_output/prepare/prepared.jsonl" "$seed_output/ui2coden"
    fi
    experiment_banner END "Model experiment"
  done
done
aliases_csv=$(IFS=,; echo "${aliases[*]}")
suite_options=(--root "$output" --aliases "$aliases_csv" --representative "$representative" --out "$output/paper" --require-complete-metrics --own-initial-alias ui2coden --revision-rounds "$revision_rounds" --initial-seeds "$initial_seeds")
[[ "$run_ablation" == 0 ]] && suite_options+=(--skip-ablation)
[[ "$run_ui2coden" == 1 ]] && suite_options+=(--include-ui2coden --ui2coden-rounds "$revision_rounds")
[[ "$run_tuide" == 0 ]] && suite_options+=(--skip-tuide)
run_logged "Dataset tables" python -m framediff.paper_suite "${suite_options[@]}"
echo "Completed ${#entries[@]} VLM(s). Paper tables: $output/paper/README.md"

# Read-only failure analysis after metric evaluation; excluded from repair timing.
if [[ "$run_tuide" == 1 && "${PAPER_DIAGNOSTICS:-1}" == 1 ]]; then
  run_logged "Repair diagnostics" python -m framediff.repair_diagnostics --root "$output" --dataset "$(basename "$output")" \
    --detector-checkpoint "${DETECTOR_CHECKPOINT:-$train/detector/best.pt}" \
    --device "${DEVICE:-cuda}" --out "$output/diagnostics"
fi

experiment_banner END "Dataset experiment"

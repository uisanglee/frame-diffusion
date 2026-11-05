#!/usr/bin/env bash
# Display-only logging; does not affect cache identities.
experiment_banner() {
  printf '\n%s\n' '================================================================================'
  printf '[%s] %s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$1" "$2"
  printf 'Dataset: %s | Initial model: %s | Seed: %s | GPUs: %s\n' "$EXPERIMENT_DATASET_LABEL" "${EXPERIMENT_MODEL_LABEL:-${VLM_MODEL:-unspecified}}" "${EXPERIMENT_SEED_LABEL:-${SEED:-${INITIAL_SEEDS:-42}}}" "${CUDA_VISIBLE_DEVICES:-default}"
  printf 'Output: %s\n%s\n' "$EXPERIMENT_LOG_OUTPUT" '================================================================================'
}
experiment_log_init() {
  EXPERIMENT_LOG_OUTPUT=$1
  if [[ -z ${EXPERIMENT_DATASET_LABEL:-} ]]; then
    case "$1 ${PAGE_MANIFEST:-} ${WEB_DATASET:-}" in
      *design2code-hard*|*Design2Code-Hard*|*Design2Code-HARD*) EXPERIMENT_DATASET_LABEL=Design2Code-Hard ;;
      *webui*) EXPERIMENT_DATASET_LABEL=WebUI ;;
      *) EXPERIMENT_DATASET_LABEL=Design2Code ;;
    esac
  fi
  export EXPERIMENT_DATASET_LABEL
  EXPERIMENT_ACTIVE_STAGE=Setup
  trap 'experiment_log_exit "$?"' EXIT
}
experiment_log_exit() {
  local status=$1
  if (( status != 0 )); then
    experiment_banner "FAILED (exit=$status)" "$EXPERIMENT_ACTIVE_STAGE"
  fi
}
run_logged() {
  local label=$1 started=$SECONDS
  shift
  EXPERIMENT_ACTIVE_STAGE=$label
  experiment_banner START "$label"
  "$@"
  experiment_banner END "$label | elapsed=$((SECONDS-started))s (including cache checks)"
  EXPERIMENT_ACTIVE_STAGE='Between stages'
}

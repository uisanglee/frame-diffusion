#!/usr/bin/env bash
# GPT-4o and Gemini 3.5 Flash API main-table evaluation. Secrets stay in environment variables.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_paper_api_vlms.sh D2C_ROOT OUTPUT_ROOT TRAIN_RUN_DIR' >&2
  exit 2
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running}"
root=$1;output=$2;train=$3
export RUN_ABLATION=${RUN_ABLATION:-0}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
export VLM_SPECS=${VLM_SPECS:-'gpt4o|openai-compatible|gpt-4o|server|https://api.openai.com/v1/chat/completions|OPENAI_API_KEY,gemini35flash|openai-compatible|gemini-3.5-flash|server|https://generativelanguage.googleapis.com/v1beta/openai/chat/completions|GEMINI_API_KEY|low'}
bash scripts/run_paper_experiments.sh "$root" "$output" "$train"

pricing_date=${PRICING_AS_OF:-2026-10-07}
python -m framediff.api_cost --metrics "$output/gpt4o/evaluation/metrics.jsonl" --out "$output/paper/main/gpt4o" \
  --model gpt-4o --input-usd-per-million "${GPT4O_INPUT_USD_PER_M:-2.50}" \
  --output-usd-per-million "${GPT4O_OUTPUT_USD_PER_M:-10.00}" --pricing-as-of "$pricing_date"
python -m framediff.api_cost --metrics "$output/gemini35flash/evaluation/metrics.jsonl" --out "$output/paper/main/gemini35flash" \
  --model gemini-3.5-flash --input-usd-per-million "${GEMINI_INPUT_USD_PER_M:-1.50}" \
  --output-usd-per-million "${GEMINI_OUTPUT_USD_PER_M:-9.00}" --pricing-as-of "$pricing_date"
echo "API cost reports: $output/paper/main/{gpt4o,gemini35flash}/api-cost.md"

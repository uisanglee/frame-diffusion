#!/usr/bin/env bash
# Two fixed pilot datasets (10 pages each by default); VLM baselines only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -ra selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#selected_gpus[@]} == 2 && -n "${selected_gpus[0]}" && -n "${selected_gpus[1]}" && "${selected_gpus[0]}" != "${selected_gpus[1]}" ]] || {
  echo 'Select two different GPUs, for example CUDA_VISIBLE_DEVICES=0,1' >&2; exit 2;
}
export PYTHONUNBUFFERED=1
export VLM_DEVICE_MAP=balanced
export VLM_GPU_MEMORY_GIB=${VLM_GPU_MEMORY_GIB:-20}
export UI2CODEN_PRECISION=bf16
export UI2CODEN_MAX_NEW_TOKENS=4096 UI2CODEN_MAX_CONTEXT=0
export INITIAL_SEEDS=42 INITIAL_TEMPERATURE=auto MAX_NEW_TOKENS=4096
export PAGE_LIMIT=0 REVISION_ROUNDS=2 REPEATS=1
export RUN_TUIDE=0 RUN_UI2CODEN=1 RUN_ABLATION=0 PAPER_DIAGNOSTICS=0 TUIDE_SCALES=0
export OFFICIAL_REPO=${OFFICIAL_REPO:-$PWD/external/design2code-official}
export PAPER_REFERENCE_CACHE=${PAPER_REFERENCE_CACHE:-$PWD/.cache/paper-reference-renders}
export VLM_SPECS=${VLM_SPECS:-'llava-next-7b|hf|llava-hf/llava-v1.6-mistral-7b-hf|bf16,glm4v-9b-thinking|hf|zai-org/GLM-4.1V-9B-Thinking|bf16,websight-8b|websight|HuggingFaceM4/VLM_WebSight_finetuned|bf16,design2code-18b|design2code-sat|SALT-NLP/Design2Code-18B-v0|bf16,ui2coden|hf|zai-org/UI2Code_N|bf16'}
pilot_pages=${D2C_PILOT_PAGES:-10}
[[ "$pilot_pages" =~ ^[1-9][0-9]*$ ]] || {
  echo 'D2C_PILOT_PAGES must be a positive integer' >&2; exit 2;
}
output=${D2C_PILOT_OUTPUT:-runs/d2c-pilot${pilot_pages}-seed42-n2-bf16-2gpu-v1}
d2c=${D2C_ROOT:-$PWD/data/raw/design2code/testset_final}
hard=${D2C_HARD_ROOT:-$PWD/data/raw/design2code-hard}
# Select and validate BOTH datasets before any costly model generation.
python - "$d2c" "$hard" "$output" "$pilot_pages" <<'PY'
import json,sys
from pathlib import Path
from framediff.paper_matrix import check_sources
folder=Path(sys.argv[3])/'inputs'
folder.mkdir(parents=True,exist_ok=True)
count=int(sys.argv[4])
for name,root in [('design2code',sys.argv[1]),('design2code-hard',sys.argv[2])]:
    root=Path(root).resolve()
    if not (root/'rick.jpg').is_file():raise ValueError(f'Missing placeholder: {root}/rick.jpg')
    rows=sorted(check_sources(root=root),key=lambda r:str(r['id']))[:count]
    if len(rows)!=count:raise ValueError(f'{name}: need {count} pages')
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    path=folder/f'{name}.jsonl'
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'Pilot inputs changed: {path}; use a new D2C_PILOT_OUTPUT')
    path.write_text(content)
    print(f'{name}: {len(rows)} fixed pages -> {path}',flush=True)
PY
echo "VLM-only pilot: GPUs=$CUDA_VISIBLE_DEVICES, pages/dataset=$pilot_pages, BF16, seed=42, N=2, output=$output"
for dataset in design2code design2code-hard; do
  WEB_DATASET=design2code PAGE_MANIFEST="$output/inputs/$dataset.jsonl" \
    bash scripts/run_paper_experiments.sh . "$output/$dataset" unused-no-checkpoints
done
python -m framediff.paper_matrix export --root "$output" --out "$output/table" \
  --datasets design2code design2code-hard --skip-tuide --revision-rounds 2
echo "Completed. Combined table: $output/table/vlm-quality.tex"

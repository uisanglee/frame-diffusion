#!/usr/bin/env bash
# Procedural pretraining -> WebUI fine-tuning -> held-out controlled repair.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage (from repo root): bash scripts/run_visual_full.sh WEBUI_SPLIT_ROOT RUN_TAG' >&2
  exit 2
fi
webui_root=$1
tag=$2
if [[ ! -d "$webui_root" ]]; then echo "Missing WebUI root: $webui_root" >&2; exit 2; fi
if [[ ! "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then echo 'RUN_TAG must contain letters, numbers, _ or -' >&2; exit 2; fi
pre_data="data/$tag-pretrain"
pre_run="runs/$tag-pretrain"
ft_data="data/$tag-webui"
ft_run="runs/$tag-webui"
eval_run="runs/$tag-test"
export WEBUI_TRAIN=${WEBUI_TRAIN:-6000} WEBUI_VAL=${WEBUI_VAL:-2000} WEBUI_TEST=${WEBUI_TEST:-2000}
export BATCH_SIZE=${BATCH_SIZE:-2} ACCUMULATION=${ACCUMULATION:-4}
export VAL_SAMPLES=${VAL_SAMPLES:-2000} EVAL_EVERY=${EVAL_EVERY:-500}
# Validate dataset availability and freeze the split before spending time on training.
python -m framediff visual-import-webui --root "$webui_root" --out "$ft_data/source" \
  --view "${WEBUI_VIEW:-default_1280-720}" --train-count "$WEBUI_TRAIN" \
  --val-count "$WEBUI_VAL" --test-count "$WEBUI_TEST" --seed "${SEED:-42}" --resume

echo '[1/4] Procedural pretraining'
TRAIN_HTML_MANIFEST='' TRAIN_PAGES=${PRETRAIN_PAGES:-1000} \
DETECTOR_STEPS=${PRE_DETECTOR_STEPS:-10000} \
POLICY_STAGE1_STEPS=${PRE_POLICY_STAGE1_STEPS:-10000} \
POLICY_STAGE2_STEPS=${PRE_POLICY_STAGE2_STEPS:-5000} \
bash scripts/train_visual.sh "$pre_data" "$pre_run"

echo '[2/4] WebUI fine-tuning'
FINETUNE_FROM_PROCEDURAL=1 REUSE_DETECTOR=0 DETECTOR_SOURCE=rendered \
DETECTOR_STEPS=${FT_DETECTOR_STEPS:-20000} \
POLICY_STAGE1_STEPS=${FT_POLICY_STAGE1_STEPS:-30000} \
POLICY_STAGE2_STEPS=${FT_POLICY_STAGE2_STEPS:-15000} \
bash scripts/finetune_visual_webui.sh "$webui_root" "$ft_data" "$ft_run" "$pre_run"

echo '[3/4] Controlled test on usable held-out pages (no VLM generation)'
echo "Selected/excluded/usable counts: $ft_data/rendered/report.json"
PAGE_MANIFEST="$ft_data/rendered/pages-test.jsonl" WEB_DATASET=webui \
PAGE_LIMIT=0 REPEATS=${REPEATS:-3} REPAIR_STEPS=${REPAIR_STEPS:-20} \
ORACLE_ABLATION=1 SELF_REVISION_PROTOCOL=none \
bash scripts/run_visual_web.sh "$webui_root" "$eval_run" "$ft_run"

echo '[4/4] Figures'
python -m framediff visual-figures --run-root "$pre_run" --out "$pre_run/figures"
python -m framediff visual-figures --run-root "$ft_run" --evaluation "$eval_run/repair" --out "$ft_run/figures"
echo "Complete. Results: $eval_run/evaluation/report.md"
echo "Coverage: $ft_data/rendered/report.json"

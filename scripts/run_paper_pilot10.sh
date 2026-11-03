#!/usr/bin/env bash
# Thirty fixed inputs, all six initial VLMs, no training or dataset changes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OFFICIAL_REPO=${OFFICIAL_REPO:-$PWD/external/design2code-official}
train=${TRAIN_RUN:-$PWD/runs/webui-full-dom-screened-7k-v1/7000}
export DETECTOR_CHECKPOINT=${DETECTOR_CHECKPOINT:-$train/detector/best.pt}
export ABSTRACT_CHECKPOINT=${ABSTRACT_CHECKPOINT:-$train/tuide-s-abstract/replacement/abstract-stage2/best.pt}
export PAPER_REFERENCE_CACHE=${PAPER_REFERENCE_CACHE:-$PWD/.cache/paper-reference-renders}
export INITIAL_SEEDS=42 INITIAL_TEMPERATURE=auto MAX_NEW_TOKENS=16384
export PAGE_LIMIT=0 REPAIR_STEPS=20 REVISION_ROUNDS=2 REPEATS=1 RUN_UI2CODEN=1 RUN_ABLATION=0
output=${PILOT_OUTPUT:-runs/paper-pilot10-seed42-n2-v1}
webui=${WEBUI_TEST:-$PWD/data/webui-full-dom-screened-7k-v1/7000/css/pages-test.jsonl}
d2c=${D2C_ROOT:-$PWD/data/raw/design2code/testset_final}
hard=${D2C_HARD_ROOT:-$PWD/data/raw/design2code-hard}
python - "$webui" "$d2c" "$hard" "$output" <<'PY'
import json, sys
from pathlib import Path
from framediff.paper_matrix import check_sources
folder=Path(sys.argv[4])/'inputs'
folder.mkdir(parents=True,exist_ok=True)
for name, options in [('webui',dict(manifest=sys.argv[1],dataset='webui')),
                      ('design2code',dict(root=sys.argv[2])),
                      ('design2code-hard',dict(root=sys.argv[3]))]:
    rows=check_sources(**options)
    # Prefer desktop 1280x720; take one view per unique WebUI page.
    rows.sort(key=lambda r:(str(r.get('page_id',r['id'].split('/')[0])),
                           r.get('view')!='default_1280-720',str(r['id'])))
    selected=[]; seen=set()
    for row in rows:
        key=str(row.get('page_id',row['id'].split('/')[0])) if name=='webui' else str(row['id'])
        if key in seen:continue
        seen.add(key);selected.append(row)
        if len(selected)==10:break
    if len(selected)!=10:raise ValueError(f'{name}: need 10 unique pages')
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
    path=folder/f'{name}.jsonl'
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'Pilot inputs changed: {path}; use a new PILOT_OUTPUT')
    path.write_text(content)
    print(f'{name}: 10 unique pages, 10 renderings -> {path}',flush=True)
PY
for dataset in webui design2code design2code-hard; do
  kind=design2code
  [[ "$dataset" == webui ]] && kind=webui
  WEB_DATASET="$kind" PAGE_MANIFEST="$output/inputs/$dataset.jsonl" \
    bash scripts/run_paper_4090_vlms.sh . "$output/$dataset" "$train"
done
python -m framediff.paper_matrix export --root "$output" --out "$output/table"

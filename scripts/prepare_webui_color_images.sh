#!/usr/bin/env bash
# Rebuild labels and frozen corruption from raw source HTML, never old normalized pages.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
source_manifest=${SOURCE_MANIFEST:-data/webui-full-dom-screened-7k-v1/7000/source/manifest.jsonl}
data=${COLOR_DATA:-data/new}
size=${DATA_SIZE:-7000}
workers=${PREPARE_WORKERS:-4}
python -m framediff.webui_placeholders --manifest "$source_manifest" --out "$data/master/source"
python - "$data" "$size" <<'PY'
import sys
from pathlib import Path
from framediff.ir import read_jsonl,write_json,write_jsonl
from framediff.web_experiment import guard_run
from framediff.visual_data import validate_splits
root=Path(sys.argv[1]);size=int(sys.argv[2]);rows=list(read_jsonl(root/'master/source/manifest.jsonl'))
validate_splits(rows)
counts={s:len({r.get('page_id',r['id']) for r in rows if r['split']==s}) for s in ('train','val','test')}
if not all(counts.values()):raise SystemExit('Each split must contain usable pages')
guard_run(root,dict(kind='webui-five-color-images-v1',sizes=[size],split_counts={str(size):counts}),True)
folder=root/str(size)/'source';folder.mkdir(parents=True,exist_ok=True)
write_jsonl(folder/'manifest.jsonl',rows)
for s in counts:write_jsonl(folder/f'manifest-{s}.jsonl',[r for r in rows if r['split']==s])
write_json(root/'split-report.json',dict(common_test_scale=size,sizes={str(size):counts},
    protocol='Inherited source splits; actual counts after image availability/geometry exclusions'))
print('Actual unique pages by split:',counts)
PY
python -m framediff visual-build-data --manifest "$data/master/source/manifest.jsonl" \
  --out "$data/master/rendered" --workers "$workers" --clean-only \
  --abstract-size 384 --seed 42 --min-elements 3 --max-source-mae 1 --resume
python -m framediff visual-tree-normalize --rendered "$data/master/rendered" \
  --out "$data/master/labels" --workers "$workers" --latest-visual-labels --clean-targets-only \
  --max-css-owners 512 --observation-size 384 --seed 42 --resume
python -m framediff visual-tree-freeze --rendered "$data/master/labels" \
  --out "$data/master/css" --workers "$workers" --samples-per-page 1 \
  --max-noise 4 --observation-size 384 --seed 90210 --resume
python -m framediff visual-webui-scales-subset --data "$data"

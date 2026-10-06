#!/usr/bin/env bash
# Build v7 multi-positive supervision and shared policy inputs. Does not retrain
# the detector or regenerate the cached HTML/screenshots/corruption states.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ $# -gt 4 ]]; then
  echo 'Usage: bash scripts/prepare_numeric_policy_v7.sh [OLD_RENDERED] [DETECTOR_BEST_PT] [NEW_RENDERED] [SHARED_RUN]' >&2
  exit 2
fi

old_rendered=${1:-}
detector=${2:-runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt}
new_rendered=${3:-data/webui-10k-v7-multipositive/rendered}
shared_run=${4:-runs/numeric-policy-v7-shared-fresh}
python_bin=${PYTHON:-python}
cache_gpu=${CACHE_GPU:-3}

discover_rendered() {
  "$python_bin" - "$new_rendered" <<'PY'
import json
import sys
from pathlib import Path

excluded=Path(sys.argv[1]).resolve()
candidates=[]
for train in Path('data').glob('**/policy-train.jsonl'):
    folder=train.parent
    if folder.resolve()==excluded:
        continue
    required=[folder/f'{kind}-{split}.jsonl'
              for split in ('train','val','test')
              for kind in ('policy','pages','detector')]
    if not all(path.is_file() for path in required):
        continue
    rows=0
    has_corruption=False
    try:
        with train.open() as stream:
            for line in stream:
                rows+=1
                if not has_corruption:
                    row=json.loads(line)
                    has_corruption=bool(row.get('corruption_edit'))
    except (OSError,json.JSONDecodeError):
        continue
    if rows and has_corruption:
        candidates.append((rows,str(folder)))
if not candidates:
    raise SystemExit('Could not auto-discover a complete rendered corruption corpus under data/')
candidates.sort(reverse=True)
print(candidates[0][1])
PY
}

if [[ -z "$old_rendered" || ! -f "$old_rendered/policy-train.jsonl" ]]; then
  if [[ -n "$old_rendered" ]]; then
    echo "Supplied source is invalid; auto-discovering instead: $old_rendered" >&2
  else
    echo 'Auto-discovering the largest complete rendered corruption corpus under data/ ...' >&2
  fi
  old_rendered=$(discover_rendered)
fi
echo "Selected source rendered corpus: $old_rendered"

for split in train val test; do
  for kind in policy pages detector; do
    path="$old_rendered/$kind-$split.jsonl"
    [[ -f "$path" ]] || { echo "Missing source file: $path" >&2; exit 2; }
  done
done
[[ -f "$detector" ]] || { echo "Missing detector checkpoint: $detector" >&2; exit 2; }

grep -q 'joint_improvement' framediff/visual_train.py || {
  echo 'This checkout is not the v7 multi-positive implementation. Update the repository first.' >&2
  exit 2
}

echo '[1/3] Browser-verified multi-positive relabeling'
"$python_bin" -m framediff visual-relabel-best-reverse \
  --rendered "$old_rendered" \
  --out "$new_rendered" \
  --resume

"$python_bin" - "$new_rendered/policy-train.jsonl" <<'PY'
import json
import sys

path=sys.argv[1]
rows=positives=0
with open(path) as stream:
    for line in stream:
        row=json.loads(line)
        if not row.get('teacher_edits'):
            continue
        rows+=1
        if row.get('teacher_strategy')!='improvement-distribution-v1' or not row.get('improving_edits'):
            raise SystemExit(f'Invalid v7 supervision in {row.get("id")}')
        if any(item.get('gain',0)<=0 for item in row['improving_edits']):
            raise SystemExit(f'Non-positive improvement gain in {row.get("id")}')
        positives+=len(row['improving_edits'])
if not rows:
    raise SystemExit('No EDIT rows found after relabeling')
print({'validated_edit_rows':rows,'improving_actions':positives,
       'mean_improving_actions':positives/rows})
PY

echo '[2/3] Numeric-only complete-prefix subset'
"$python_bin" -m framediff visual-subset-numeric \
  --rendered "$new_rendered" \
  --out "$shared_run/subset"

echo "[3/3] Frozen detector target cache on physical GPU $cache_gpu"
for split in train val; do
  CUDA_VISIBLE_DEVICES=$cache_gpu "$python_bin" -m framediff visual-cache-targets \
    --data "$shared_run/subset/policy-$split.jsonl" \
    --checkpoint "$detector" \
    --out "$shared_run/predicted-$split" \
    --device cuda \
    --resume
done

for required in \
  "$shared_run/subset/policy-train.jsonl" \
  "$shared_run/subset/policy-val.jsonl" \
  "$shared_run/predicted-train/data.jsonl" \
  "$shared_run/predicted-val/data.jsonl"; do
  [[ -s "$required" ]] || { echo "Preparation produced no data: $required" >&2; exit 1; }
done

echo 'v7 policy data preparation complete.'
echo "  rendered: $new_rendered"
echo "  shared:   $shared_run"
echo "  detector: $detector"

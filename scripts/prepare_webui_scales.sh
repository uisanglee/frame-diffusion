#!/usr/bin/env bash
# Raw WebUI -> shared assets -> nested 8:1:1 dataset manifests.
# TUIDE_SAVE_DIAGNOSTICS=1 retains normalization/scroll screenshots; default 0.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/prepare_webui_scales.sh RAW_WEBUI_ROOT DATA_ROOT' >&2
  exit 2
fi
raw=$1
data=$2
if [[ "${BOX_LAYOUT:-0}" != 0 || "${RECOVER_NORMALIZATION:-0}" != 0 || -n "${REUSE_NORMALIZED:-}" ]]; then
  echo 'Retired normalization/xywh options: use safe size/margin preparation with a new labels directory.' >&2
  exit 2
fi
read -r -a sizes <<< "${WEBUI_SIZES:-7000 70000}"
source "$repo_root/scripts/webui_views.sh"
split_options=()
if [[ -n "${MASTER_COUNTS:-}" ]]; then
  read -r -a master_counts <<< "$MASTER_COUNTS"
  split_options+=(--master-counts "${master_counts[@]}")
fi
if [[ -n "${ALLOCATION_COUNTS:-}" ]]; then
  read -r -a allocation_counts <<< "$ALLOCATION_COUNTS"
  split_options+=(--allocation-counts "${allocation_counts[@]}")
fi
if [[ -n "${REUSE_SPLITS:-}" ]]; then
  # Reuse the exact published selections; do not scan raw HTML or repartition.
  python - "$REUSE_SPLITS" "$data" <<'PY'
import json
import sys
from pathlib import Path

source, target = (Path(p).resolve() for p in sys.argv[1:])
if source == target or source in target.parents or target in source.parents:
    raise SystemExit('REUSE_SPLITS requires a separate output root')
config = json.loads((source / 'config.json').read_text())
files = ['config.json', 'split-report.json']
directories = ['master/source'] + [f'{size}/source' for size in config['sizes']]
# Preflight every path before publishing anything. Existing new-label results
# remain usable only when their exact original selections still match.
for name in files:
    src, dst = source / name, target / name
    if not src.is_file():
        raise SystemExit(f'Missing split metadata: {src}')
    if dst.exists() and dst.read_bytes() != src.read_bytes():
        raise SystemExit(f'Split metadata differs: {dst}; choose a fresh output root')
for name in directories:
    src, dst = source / name, target / name
    if not (src / 'manifest.jsonl').is_file():
        raise SystemExit(f'Missing selection: {src}')
    if dst.exists() or dst.is_symlink():
        if not dst.is_dir():
            raise SystemExit(f'Invalid existing source directory: {dst}')
        for path in src.rglob('*'):
            if path.is_file():
                existing = dst / path.relative_to(src)
                if not existing.is_file() or existing.read_bytes() != path.read_bytes():
                    raise SystemExit(f'Existing selection differs: {existing}')
target.mkdir(parents=True, exist_ok=True)
for name in files:
    dst = target / name
    if not dst.exists():
        temp = dst.with_suffix('.tmp')
        temp.write_bytes((source / name).read_bytes())
        temp.replace(dst)
for name in directories:
    dst = target / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to((source / name).resolve(), target_is_directory=True)
print(f'Reusing exact splits from {source}; sizes={config["sizes"]}; raw scan skipped', flush=True)
PY
  export REUSE_RENDERED="${REUSE_RENDERED:-$REUSE_SPLITS/master/rendered}"
else
  python -m framediff visual-webui-scales-prepare --root "$raw" --out "$data" \
    --sizes "${sizes[@]}" --views "${views[@]}" --seed "${SEED:-42}" "${split_options[@]}" --resume
fi
# REUSE_RENDERED may point at a complete master/rendered directory from an
# identical source manifest.  This keeps the expensive HTML rendering cache
# while allowing labels/CSS data to be rebuilt in a fresh experiment root.
if [[ -n "${REUSE_RENDERED:-}" ]]; then
  python - "$data" "$REUSE_RENDERED" <<'PY'
import hashlib
import sys
from pathlib import Path

data = Path(sys.argv[1])
source = Path(sys.argv[2]).resolve()
target = data / "master/rendered"
manifest = data / "master/source/manifest.jsonl"
source_manifest = source.parent / "source/manifest.jsonl"

if not source.is_dir():
    raise SystemExit(f"REUSE_RENDERED is not a directory: {source}")
required = (
    "detector-train.jsonl", "detector-val.jsonl", "detector-test.jsonl",
    "policy-train.jsonl", "policy-val.jsonl", "policy-test.jsonl",
)
missing = [name for name in required if not (source / name).is_file()]
if missing:
    raise SystemExit(f"Incomplete rendered cache {source}; missing: {missing}")
if not source_manifest.is_file():
    raise SystemExit(f"Cannot verify rendered cache manifest: {source_manifest}")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if digest(manifest) != digest(source_manifest):
    raise SystemExit("Source manifests differ; refusing to reuse rendered cache")
if target.is_symlink():
    if target.resolve() != source:
        raise SystemExit(f"Rendered symlink points elsewhere: {target} -> {target.resolve()}")
elif target.exists():
    if target.resolve() != source:
        raise SystemExit(f"Rendered output already exists and is not the requested cache: {target}")
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
print(f"Reusing verified rendered cache: {source}")
PY
else
  # No historical detector checkpoint or offline trajectories required.
  python -m framediff visual-build-data --manifest "$data/master/source/manifest.jsonl" \
    --workers "${RENDER_WORKERS:-${PREPARE_WORKERS:-4}}" \
    --out "$data/master/rendered" --clean-only --lazy-train --abstract-size "${IMAGE_SIZE:-384}" \
    --seed "${SEED:-42}" --min-elements "${MIN_ELEMENTS:-3}" \
    --max-source-mae "${MAX_SOURCE_MAE:-1.0}" --resume
fi
if [[ "${REBUILD_HELDOUT:-0}" == 1 ]]; then
  [[ -n "${REUSE_SPLITS:-}" ]] || { echo 'REBUILD_HELDOUT requires REUSE_SPLITS' >&2; exit 2; }
  export TUIDE_SHARED_CACHE="${TUIDE_SHARED_CACHE:-$repo_root/data/webui-shared-cache}"
  python -m framediff visual-policy-relabel \
    --rendered "$REUSE_SPLITS/master/labels" --out "$data/master/labels" \
    --clean-heldout --workers "${PREPARE_WORKERS:-4}" \
    --observation-size "${IMAGE_SIZE:-384}" --resume
  python -m framediff visual-tree-freeze --rendered "$data/master/labels" \
    --out "$data/master/css" --workers "${PREPARE_WORKERS:-4}" \
    --samples-per-page "${FIXED_SAMPLES_PER_PAGE:-1}" --max-noise "${ONLINE_MAX_NOISE:-4}" \
    --observation-size "${IMAGE_SIZE:-384}" --seed "${TEST_SEED:-90210}" --resume
elif [[ "${RELABEL_POLICY_ONLY:-0}" == 1 ]]; then
  [[ -n "${REUSE_SPLITS:-}" ]] || { echo 'RELABEL_POLICY_ONLY requires REUSE_SPLITS' >&2; exit 2; }
  export TUIDE_SHARED_CACHE="${TUIDE_SHARED_CACHE:-$repo_root/data/webui-shared-cache}"
  # Frozen policy rows are relabeled independently from the original css rows.
  # Keep normalized labels for subset coverage/provenance without re-rendering
  # their unused seed corruptions. Never symlink the directory itself: subset
  # writes prepare-report.json into the destination directory.
  python - "$REUSE_SPLITS/master/labels" "$data/master/labels" <<'PY'
import shutil
import sys
import tempfile
from pathlib import Path

source, target = (Path(arg).resolve() for arg in sys.argv[1:])
if source == target or source in target.parents or target in source.parents:
    raise SystemExit('Reused labels need a separate output directory')
required = [f'{kind}-{split}.jsonl' for split in ('train', 'val', 'test')
            for kind in ('policy', 'pages')]
required += [f'rejections-{split}.jsonl' for split in ('train', 'val', 'test')]
for name in required:
    if not (source / name).is_file():
        raise SystemExit(f'Missing normalized-label manifest: {source / name}')
if target.exists() or target.is_symlink():
    if target.is_symlink() or not target.is_dir():
        raise SystemExit(f'Invalid existing label directory: {target}')
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise SystemExit(f'Partial existing label directory: {target}; missing {missing}')
    print(f'Keeping existing normalized labels: {target}', flush=True)
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        for name in required:
            (staging / name).symlink_to(source / name)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f'Reusing normalized-label manifests without relabel: {source}', flush=True)
PY
  python -m framediff visual-policy-relabel \
    --rendered "$REUSE_SPLITS/master/css" --out "$data/master/css" \
    --workers "${PREPARE_WORKERS:-4}" --observation-size "${IMAGE_SIZE:-384}" --resume
else
label_options=()
case "${LATEST_VISUAL_LABELS:-0}" in
  1) label_options+=(--latest-visual-labels) ;;
  0) ;;
  *) echo 'LATEST_VISUAL_LABELS must be 0 or 1' >&2; exit 2 ;;
esac
python -m framediff visual-tree-normalize --rendered "$data/master/rendered" \
  "${label_options[@]}" \
  --out "$data/master/labels" --workers "${PREPARE_WORKERS:-4}" \
  --seed "${SEED:-42}" --max-css-owners "${MAX_CSS_OWNERS:-512}" \
  --observation-size "${IMAGE_SIZE:-384}" \
  --max-pixel-mae "${NORMALIZE_MAX_PIXEL_MAE:-0.01}" \
  --max-box-error "${NORMALIZE_MAX_BOX_ERROR:-1}" --resume
python -m framediff visual-tree-freeze --rendered "$data/master/labels" \
  --out "$data/master/css" --workers "${PREPARE_WORKERS:-4}" \
  --samples-per-page "${FIXED_SAMPLES_PER_PAGE:-1}" --max-noise "${ONLINE_MAX_NOISE:-4}" \
  --observation-size "${IMAGE_SIZE:-384}" --seed "${TEST_SEED:-90210}" --resume
fi
python -m framediff visual-webui-scales-subset --data "$data"
echo "Prepared nested scale datasets in $data. Keep master/ assets."

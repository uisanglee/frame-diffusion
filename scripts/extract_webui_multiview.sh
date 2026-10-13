#!/usr/bin/env bash
# Extract only requested WebUI page views from the official split archives.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/extract_webui_multiview.sh DOWNLOAD_ROOT RAW_OUT' >&2
  exit 2
fi
downloads=$1
raw=$2
source "$repo_root/scripts/webui_views.sh"
includes=()
for view in "${views[@]}"; do includes+=("-ir!*$view-*"); done
view_contract=$(IFS=,; echo "${views[*]}")

extract_split() {
  local repo=$1 archive=$2 split=$3
  local source="$downloads/$repo/$archive" destination="$raw/$split"
  local marker="$destination/.extract-complete"
  [[ -f "$source" ]] || { echo "Missing archive: $source" >&2; return 2; }
  if [[ -f "$marker" && "$(<"$marker")" == "$view_contract" ]]; then
    echo "Reusing completed extraction: $destination"
    return
  fi
  mkdir -p "$destination"
  # -aoa makes a retry repair/overwrite files left by an interrupted extraction.
  7z x "$source" "-o$destination" -aoa "${includes[@]}"
  printf '%s\n' "$view_contract" > "$marker"
}

extract_split webui-70k train_split_web70k.zip.001 train
extract_split webui-val val_split_webui.zip.001 val
extract_split webui-test test_split_webui.zip.001 test
echo "EXTRACT_COMPLETE views=${views[*]}"

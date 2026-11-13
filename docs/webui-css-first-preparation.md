# CSS-first WebUI dataset refresh

This creates a new dataset version. It does **not** train models or call VLMs.
The input is the **previously published final source manifest**, not the original
7,000-page candidate selection. By default it reuses the surviving page/view IDs
and page splits from `data/new/7000/source/manifest.jsonl`.

## Server command

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
python -m pip install 'tinycss2>=1.2'
mkdir -p log

nohup setsid env PYTHONUNBUFFERED=1 PREPARE_WORKERS=4 \
  bash scripts/prepare_webui_css_color_images.sh \
  >> log/prepare-webui-css-color-v1.log 2>&1 < /dev/null &
echo $! > log/prepare-webui-css-color-v1.pid
```

Defaults:

| Setting | Value |
|---|---|
| Previous dataset (`PREVIOUS_COLOR_DATA`) | `data/new` |
| Final split input (`SOURCE_MANIFEST`) | `data/new/7000/source/manifest.jsonl` |
| New dataset (`CSS_COLOR_DATA`) | `data/new-css-v1` |
| Original image download reuse | `data/new/master/source/downloads` |
| CSS raw-download reuse (`CSS_DOWNLOAD_CACHE`) | `runs/webui-css-replay10-v1/css-cache/downloads` |
| Workers (`PREPARE_WORKERS`) | 4 CPU/browser workers; no GPU |

The default `7000` directory is a historical scale name, **not** a promise to
produce 7,000 pages. Only previously published views are selected. Missing
downloads or failed downstream checks can reduce the final counts further.

## Processing order

1. Read the old split, recover each row's `original_html`, and archive linked
   stylesheets and recursive `@import`s. Rebase resource URLs using the CSS URL
   after redirects. Inline the CSS while retaining cascade order, media, layer
   and supports conditions. JavaScript remains disabled. External fonts are
   not downloaded; this is the existing font policy, not full historical replay.
2. Let Chromium select image sources at the saved viewport **with CSS applied**.
   Reuse hash-checked original-image downloads, download missing active images,
   then replace them with the existing five-color placeholders. Intrinsic image
   dimensions and the DOM geometry preservation check remain in force.
3. Generate new target renders and Detector boxes, normalize HTML/CSS, and check
   visible editable CSS candidates. These cannot reuse the old CSS-blocked labels.
4. Generate fixed validation/test corruptions. Publish training clean targets
   and all final manifests with the **same page split as before**. All surviving
   resolutions of a page remain together. Failures never trigger redistribution
   or filling the dataset with different pages. Cross-split source leakage is
   still checked and stops publication rather than silently moving pages.

Unsuccessful required CSS downloads reject the view, with errors recorded;
partially loaded CSS is not silently accepted. Unsupported imported namespace
scopes are also rejected rather than flattened incorrectly. Alternate/disabled
stylesheets stay inactive. Current downloaded CSS may differ from the original
historical WebUI capture: inspect the new targets before committing to training.

## Resume and cache behavior

Rerun the same command. Completed new targets are hash-checked and reused without
rendering; downstream rendering, normalization and frozen-corruption caches use
their existing resume paths. Download caches use per-URL locks across workers.
Verified successful old downloads are copied on demand into the new version;
old cache files and old dataset artifacts are not modified.

Failures in the **old** download caches are not imported. Failures encountered
in the new download cache remain cached on ordinary resume, preventing repeated
404 requests for every viewport. If you change source content or want a new
download/recovery policy, use a new dataset version rather than mixing targets
into an already published dataset. A changed raw source or modified completed
target is an error, not a silently excluded page.

The wrapper prevents concurrent preparation of the same output with a Linux
`flock`. Keep the previous raw HTML and source files; the new records retain
provenance paths to them. Do not reuse previous rendered targets, Detector
predictions or TUIDE training caches as labels for the new images.

## Progress and outputs

```bash
tail -f log/prepare-webui-css-color-v1.log
cat data/new-css-v1/split-report.json
```

Progress includes processed views, retained views, failures, cache hits and ETA.
The final log prints `final_split_complete: true` and split counts.

- `master/source/pages/<hash>/css-applied.html`: CSS-first HTML before image replacement.
- `master/source/pages/<hash>/target.png`: new color-image target, before normalization.
- `master/source/pages/<hash>/css-report.json`: archived stylesheet provenance.
- `master/source/rejections.jsonl`: CSS/image preparation errors.
- `master/rendered/`, `master/labels/`: new rendering/normalization artifacts and reports.
- `7000/rendered/detector-{train,val,test}.jsonl`: Detector data.
- `7000/css/policy-{train,val,test}.jsonl`: TUIDE data.
- `7000/css/pages-{train,val,test}.jsonl`: page manifests for evaluation.
- `split-report.json`: original split counts, final counts, dropped page/view IDs
  and held-out corruption failures. Ratios may differ from 8:1:1 after exclusions.

Use a row's `target_image` / `screenshot` path to inspect the materialized target.
No model training or VLM regeneration is started automatically.

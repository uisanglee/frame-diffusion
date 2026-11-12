# Replay the existing WebUI targets with external CSS

This is a diagnostic comparison, not a replacement dataset or training run.
It reads the same ten targets from `runs/target-provenance-audit/report.json`.
Original HTML and preprocessing outputs remain untouched.

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
python -m pip install 'tinycss2>=1.2'
mkdir -p log
nohup env PYTHONUNBUFFERED=1 \
  python scripts/compare_webui_css.py \
    --audit runs/target-provenance-audit/report.json \
    --out runs/webui-css-replay10-v1 \
  > log/webui-css-replay10-v1.log 2>&1 < /dev/null &
echo $! > log/webui-css-replay10-v1.pid
tail -f log/webui-css-replay10-v1.log
```

No GPU or checkpoint is needed. The script runs one browser at a time. Download
and per-page progress are printed. The default timeout per stylesheet is 15
seconds, with at most 100 unique stylesheets per page, 8 MiB per stylesheet and
16 import levels. Failed downloads are cached and explicitly counted as partial
CSS restoration. Run the same command to reuse completed comparisons and CSS
downloads. Use `--retry-failed` to retry failed downloads and rerender comparisons;
successful downloads are retained.

## Controlled conditions

- Both conditions start from the saved **color-image source**, before size
  normalization. They use the same HTML parser, viewport and browser.
- A separate replay of the unmodified color source checks that HTML serialization
  did not introduce the observed effect. A mismatch is marked as confounded.
- Ordinary stylesheet links are downloaded, archived and served from a local
  route. Relative paths and CSS imports resolve against the document URL, existing
  `base` tag, and final URL after redirects. Import media/layer/supports conditions
  and link media attributes remain in place.
- Existing colored images, their intrinsic-size attributes and existing inline
  CSS are retained. Image/font URL resources introduced by downloaded CSS are
  suppressed, including data URIs. No new image or font files are downloaded.
  Existing embedded CSS assets remain as in the blocked condition. CSS gradients,
  borders and layout rules apply normally.
- JavaScript remains disabled. The browser cannot access live websites. Python
  downloads only the selected CSS dependencies; browser CSS requests are served
  from digest-checked local files.

The script saves raw CSS plus source/final URLs, timestamp and hashes. These are
the **currently available** stylesheets: a 2022 screenshot may have used different
CSS. A visible improvement does not prove pixel-exact recovery of that historical
capture. Font, image, JavaScript and browser-version differences remain controlled
or unresolved factors. An unavailable stylesheet is not evidence against a CSS
cause. A smaller original-image RGB MAE also is not sufficient proof of improved
layout, since the intentional image colours differ.

## Outputs

- `README.md`: per-page CSS successes/failures, changed DOM boxes, pixel change,
  and whether the comparison was complete or partial.
- `index.html`: original capture, saved color target, blocked control, CSS-restored
  rendering and previous evaluation target.
- `summary.json`: all comparisons, CSS failures and per-element geometry changes.
- `pages/*/`: HTML, screenshots, DOM geometry and reusable per-page results.
- `css-cache/downloads/`: archived raw CSS and download metadata.
- `css-cache/assets/`: rewritten, local CSS files used by the offline renderer.

`css_changes_render` with a clean serialization control attributes the new
rendering difference to the restored CSS under these conditions. It does not
automatically classify the page as correctly restored; inspect the gallery.
`no_observed_css_effect_partial_css` is inconclusive, not a successful negative
test. Restored HTML uses the diagnostic local CSS route; open the saved PNG/gallery
for viewing rather than expecting a standalone browser to resolve that reserved
origin directly.

To share the gallery and complete evidence:

```bash
tar -czf runs/webui-css-replay10-v1.tar.gz -C runs webui-css-replay10-v1
```

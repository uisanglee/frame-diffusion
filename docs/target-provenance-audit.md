# Target image provenance audit

This audit distinguishes loss of styling before normalization from a change
introduced by normalization or by the evaluation renderer. It reads saved files;
it does not generate VLM outputs, train, download assets, or redo preprocessing.

## Findings from the current implementation

The WebUI colour-image pipeline fetches image resources but does not archive or
load external stylesheets. `Browser._new_context()` disables page JavaScript and
aborts requests other than registered local assets. `HtmlBrowser.load()` uses
`set_content()` without reconstructing the original document's base URL. Inline
CSS and embedded `<style>` content can still apply, while ordinary external
`<link rel="stylesheet">` and `@import` dependencies cannot be relied on.

This happens **before** safe-size normalization. In particular:

1. `webui_placeholders.convert_html()` renders the original HTML under that
   offline policy, downloads selected images, and replaces them with colours.
   Its geometry check compares two versions under the same offline CSS policy.
2. `_prepare_view()` replaces the manifest's `screenshot` with this new
   `target.png`. It retains `original_html`, but the screenshot field no longer
   points at the original WebUI capture.
3. `visual_data` compares its render against that replaced screenshot. The
   wrapper's `--max-source-mae 1` is also permissive on a 0–1 scale.
4. `safe_sizes.convert()` checks preservation against its immediate input.
   An accepted report proves that input was preserved within the configured
   bounds, not that the original captured website was reproduced.

Therefore a page that depends on external layout CSS can already be unstyled
before normalization and still pass all of these checks. Missing styles can
also expose originally hidden menus, changing both the visible content and
layout. Replacing image pixels with the intentional red/green/yellow/blue/black
placeholders is a separate change and is not itself a target-rendering failure.

## Evidence checked on 2026-11-13

- The supplied comparison gallery contains visibly unstyled WebUI targets,
  including `1655890736260/default_1280-720` and
  `1655896586203/default_1280-720`. Their raw/intermediate HTML files are on the
  server, not in the gallery, so the gallery alone cannot locate the exact stage.
  All 10 WebUI pages reference the same target across the four initial VLMs.
  The target problem therefore precedes those models' individual generations.
- The earlier supplied preprocessing audit measured 105 views from 20 pages:
  maximum normalized RGB MAE was `0.000023888548474945534`, and maximum DOM box
  change was `0.09375` CSS pixels. These measurements compared pre-normalization
  rendered HTML with normalized HTML, not the original WebUI capture. The two
  gallery IDs above and `1655906481949` were not in that sample.
- A controlled local browser probe reproduced the mechanism: three horizontal
  cards became vertical when their external stylesheet was blocked. Safe-size
  normalization accepted the unstyled version with zero pixel and box change.
  Embedding the same known CSS restored the horizontal layout. This verifies
  the code path, not the cause of every dataset page.
- The server SSH connection was reset before authentication. No claim of having
  inspected the current server's per-page intermediate files is made here.

## Compare the saved experiment pages

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
mkdir -p log
python scripts/audit_target_provenance.py \
  --run runs/paper-color-pilot10-seed42-n2-v1 \
  --data data/new \
  --count 10 \
  --out runs/target-provenance-audit \
  > log/target-provenance-audit.log 2>&1
```

Read `runs/target-provenance-audit/README.md`, `report.json`, and `index.html`.
The gallery orders available images as original capture, colour replacement,
tagged input to normalization, normalized output, and evaluation target.
Different reference variants are not silently combined. Missing files and
dimension mismatches remain unavailable comparisons; no images are resized to
manufacture a match. The original captured image can differ intentionally in
image colours, so a pixel difference alone is not evidence of a CSS failure.

Use the saved normalization report together with its input/output hashes to
attribute changes. If tagged input and normalized output are identical, the
normalizer did not introduce a visible change in that viewport. If normalized
output differs from the evaluation target, examine rendering/resource/cache
differences at that later stage. External CSS references plus the offline policy
identify an unresolved dependency, but proving that a particular stylesheet
caused a particular difference requires the stylesheet contents or a controlled
replay with archived CSS.

Do not overwrite training data or delete caches to run this audit. A later fix
should archive and resolve required layout styles before taking the baseline,
then use a versioned dataset and consistent renders. Simply permitting live
network access cannot guarantee reproduction of historical WebUI captures.

# WebUI color-image protocol and migration

The current preparation entry point is `scripts/prepare_webui_css_color_images.sh`.
Follow [CSS-first preparation](webui-css-first-preparation.md) for commands,
outputs, failure handling and resume behavior, and
[mixed geometry corruption](mixed-corruption.md) for training and held-out data.
The previous CSS-blocked preparation and screened-page selection wrappers have
been retired.

## Current processing order

Archive and apply the page's CSS first, then select active images at each saved
viewport and replace them with the five-color placeholders. Generate target
screenshots and Detector labels from this result, normalize HTML/CSS, and publish
Train, Validation and Test with the previous page split preserved. Training uses
clean targets with online corruption; Validation and Test contain fixed clean,
mild, medium, strong and multi-element corruption conditions.

The placeholder sequence is `rick1.jpg` (red), `rick2.jpg` (green), `rick3.jpg`
(yellow), `rick4.jpg` (blue), and `rick5.jpg` (black), repeated as needed.
The image replacement preserves original intrinsic dimensions and checks layout
geometry. A repeated color is an asset rule, not a unique element identifier.
Design2Code and Design2Code-Hard continue to use their separate `rick.jpg` rule.

## Preserve existing data and download caches

The default input remains the previously published
`data/new/7000/source/manifest.jsonl`. Original HTML provenance and successful
image downloads under `data/new/master/source/downloads` are reused; existing CSS
downloads can also be reused. Keep these inputs and caches. The updated dataset
is written to `data/new-css-v1`, without overwriting the old dataset or changing
its page splits.

Old CSS-blocked screenshots, normalized HTML and Detector labels cannot serve as
labels for the CSS-corrected targets. They are regenerated in the new version.
Rerunning the current preparation command reuses completed new-version artifacts
and compatible inspection/download caches. If only the corruption mixture needs
to be refreshed after normalization, use
`scripts/refresh_webui_mixed_corruption.sh`; it reuses the normalized targets.

No data directories, experiment results or model checkpoints are removed by this
migration. Downloaded CSS reflects the available server version, so inspect the
new targets before training.

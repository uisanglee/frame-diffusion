# WebUI five-colour image experiment

This is a new controlled benchmark protocol, not native WebUI screenshot fidelity.
Each HTML image element, in DOM order, uses red/green/yellow/blue/black, repeated
as `rick1.jpg` through `rick5.jpg`. The source page's current-viewport image is
fully decoded before replacement; intrinsic dimensions are preserved. Width/height
CSS, flow, and aspect-ratio behaviour remain in effect. The browser resolves these
reserved names to small solid SVGs with the recorded intrinsic dimensions, avoiding
large per-element bitmaps. Five 1x1 JPEG files are also exported for reference.
Generated HTML must explicitly set both CSS dimensions. An unmatched filename is
not automatically repaired.

Prepare from the existing **raw source manifest**, never normalized HTML:

```bash
cd /home/uisang/2027/frame-diffusion
mkdir -p log
nohup env PYTHONUNBUFFERED=1 PREPARE_WORKERS=4 \
  bash scripts/prepare_webui_color_images.sh \
  > log/prepare-webui-color-images.log 2>&1 < /dev/null &
```

Defaults: source `data/webui-full-dom-screened-7k-v1/7000/source/manifest.jsonl`,
output `data/webui-color-images-v1`. Override `SOURCE_MANIFEST`, `COLOR_DATA`,
`DATA_SIZE` (the output scale name) if needed. Source splits are inherited.
Failed downloads of required active images, unknown intrinsic sizes and >1px changes to any measured DOM
box are rejected and recorded in `master/source/rejections.jsonl`. Actual counts
may be below 7000; there is no silent resampling. Source images and downloads
are cached. Missing-image failures are not replaced by invented dimensions.
The conversion step is sequential and resumes per view; subsequent render,
normalization and corruption preparation use PREPARE_WORKERS. No GPU is needed.

CSS backgrounds declared in inline styles or style blocks also get solid colours
with intrinsic size preservation, after the img-element sequence. External CSS,
fonts and JavaScript remain blocked; their restoration is not part of this
protocol. Inline vector drawings are not rewritten. Animated images are frozen
by their solid replacement. Picture/srcset is resolved for the current viewport;
unused candidates and inactive CSS background URLs are not downloaded. Active
offscreen images are still required because their intrinsic dimensions can affect
document layout. Lazy image loading is made eager for deterministic preparation.
Image buttons (`input type=image`) use dimensions decoded from their source bytes,
since they do not expose the img element's naturalWidth/naturalHeight properties.

Both the detector RGB targets and TUIDE clean targets are generated from the same
converted source. All splits, including Train, are rendered and labeled eagerly.
Normalization and visible CSS candidate inspection finish before training starts.
Inspected clean targets are saved without temporary seed corruptions; only val/test
receive fixed corruptions in the freeze stage. Train receives new online corruptions
during training. Candidate-cache paths are retained in the manifests so TUIDE and
screenshot runs (including different model sizes) reuse the same verified entries.
HTML/code/viewport changes still invalidate the appropriate cache entries.
Old labels/normalization caches and
old model weights must not be treated as training runs under this protocol.

Preparation failures are recorded in each stage's rejection report; the split
manifests contain the accepted views. `7000` is the scale name, not a guarantee of
7000 retained pages. Inspect `7000/coverage.json` for final counts. If preparation
was started with the earlier lazy version of this script, use a fresh COLOR_DATA
directory; do not mix its deferred Train manifests with the eager pipeline.

Train in a **new run directory**, using the existing training entry point:

```bash
CUDA_VISIBLE_DEVICES=0 STAGE=all POLICY_MODES=abstract POLICY_SCALE=s \
  ONLINE_WORKERS=4 POLICY_STAGE1_STEPS=15000 POLICY_STAGE2_STEPS=5000 \
  bash scripts/run_webui_scale.sh 7000 \
  data/webui-color-images-v1 runs/webui-color-images-v1
```

Use `data/webui-color-images-v1/7000/css/pages-test.jsonl` as WEBUI_TEST and
`runs/webui-color-images-v1/7000` as TRAIN_RUN for VLM experiments, with the new
Detector and TUIDE checkpoints. Use a fresh experiment output directory. Initial
and self-revision prompts automatically detect the reference HTML protocol marker;
UI2CodeN polishing uses the same colour/file mapping. Reference renders and
browser execution embed these reserved files offline. Design2Code's single
`rick.jpg` protocol is unchanged. Do not compare old and new runs as identical
benchmark settings. Existing paper wrapper scripts with hardcoded defaults must
be pointed explicitly at the new manifests/checkpoints.

The image mapping is stored per view in `master/source/pages/*/images.json`.
The supplied asset set is complete for accepted source image elements, but this
does not guarantee that a VLM uses the correct colours or includes every image.
Repeated colours are not element identities. Evaluate geometry using the usual
DOM matching, not colour equality.

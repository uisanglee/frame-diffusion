# Image-only transfer pilot (10 held-out WebUI pages)

Run from the repository root on the server:

```bash
mkdir -p log
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  bash scripts/run_image_assets_pilot10.sh \
  > log/webui-image-assets-pilot10-gpu0.log 2>&1 < /dev/null &
```

Defaults use the screened 7k test manifests and TUIDE-s stage-2 / detector
checkpoints in `runs/webui-full-dom-screened-7k-v1/7000`. Override `TRAIN_RUN`,
`ABSTRACT_CHECKPOINT`, `DETECTOR_CHECKPOINT`, `PILOT_DATA`, or `PILOT_OUT` if needed.
No retraining, VLM generation, or edits to the original dataset are performed.

The pilot selects one existing frozen corruption per unique held-out page. For
both target and corrupted HTML it prepares two paired conditions:

- `blocked`: the existing offline rendering policy.
- `images`: external image URLs present in the HTML (img/picture src/srcset and
  inline CSS image URLs) are downloaded, validated, and embedded as data URIs.

The same frozen CSS changes are used in both conditions. Both HTML versions pass
through identical serialization. Target and initial boxes are measured again in
each condition, and current geometry/class features are refreshed. Stable DOM
identities are preserved for evaluation. Target geometry is never passed into
policy action selection; target abstraction comes from the frozen detector.

External stylesheets, fonts, page JavaScript, and lazy-loading behavior remain
disabled. This tests image-only transfer on the existing normalized HTML. It does
NOT restore the original website or assess full external-CSS/font transfer.
Some images may be irretrievable or have changed since WebUI was collected.
Downloaded URLs, byte hashes and failures are saved in `screening.jsonl` and
`downloads/`. Failed downloads are cached too; use a fresh PILOT_OUT for a clean
repeat. The output directory is standalone and never replaces training caches.

At most 50 distinct page candidates are inspected, up to 32 image references per
page, with 10-second per-request socket/deadline checks and a 10 MiB size limit.
This bounds routine requests but is not a strict whole-page wall-clock deadline.
Pages must show a nonzero target pixel change after embedding to qualify. Failed
or visually unchanged candidates are recorded, not silently counted as transfer
successes. Partial asset restoration is explicitly recorded. If fewer than 10
qualify, the experiment reports the actual N; no samples are duplicated. This is
an asset-retrievable subset, not an unbiased benchmark estimate.

Both conditions use 20 repair steps, one worker, one inference item, one repeat.
Pixel-based early stopping is disabled to avoid changes in image content altering
the stopping rule; existing runtime/error guards can still stop a rollout.
Preparation and image downloads are outside the measured repair time.

Outputs in `runs/webui-image-assets-pilot10`:

- `index.html`: target/initial/repaired images for both conditions, by page.
- `comparison.md`: initial/final identity-matched Box IoU, improvement %, failures.
- `paired-results.jsonl`: page-level paired outcomes and difference in IoU gains.
- `screening.jsonl`: selection coverage, downloads, exclusions, visible changes.
- `blocked/repair` and `images/repair`: standard fixed-test rollout artifacts,
  HTML, PNG, traces, and summaries. Their coverage files describe the small
  exported pilot, not the entire original WebUI test population.

Move the entire output directory to your local machine to open `index.html`.
The pilot reuses completed preparation/rollouts when restarted. Inspect
`complete.json` for the actual selected count. Keep settings fixed on resume.

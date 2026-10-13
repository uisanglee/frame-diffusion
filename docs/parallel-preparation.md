# Parallel stylesheet preparation

Set `PREPARE_WORKERS=4` when running `scripts/prepare_stylesheet_policy.sh`
with the same four source, detector, data-output and cache-output arguments
used for your existing run. The default is 1. Both `visual-tree-normalize`
and `visual-tree-freeze` also accept `--workers 4` directly.

`visual-build-data --workers 4 --resume` also renders initial HTML in separate
page processes. `prepare_webui_scales.sh` passes `PREPARE_WORKERS` to this stage;
`RENDER_WORKERS` can override it independently. Existing `pages/*/records.json`
files remain reusable: worker count is excluded from the configuration guard.
Missing assets or unreadable completion records trigger page regeneration.
`progress.json` and console output include `cache_hits`. Successful pages are
checkpointed even before the aggregate manifests are written.

Each worker owns its browser and processes a separate page. Normalization,
candidate inspection within normalization, and fixed val/test corruption run
in parallel. Detector caching remains on `CACHE_GPU`; its inference is not
parallelized by this option. The freeze stage still reads the target inspection
cache sequentially before dispatching pages.

The script already supplies `--resume`. Successful page records and candidate
caches are reused under their existing validity checks. Interrupted pages and
failed pages without a completed record are retried. Keep the same input and
output paths. Workers are an execution setting, so changing their count does
not invalidate a run. Incompatible data/normalization settings still fail the
existing configuration guard.

Page indices/seeds and manifest order do not depend on worker completion order.
Only the coordinator writes aggregate manifests. Page completion records and
manifests are replaced atomically; unfinished `.tmp` files are overwritten on
retry. At most twice the worker count is submitted at once. Graceful interruption
may wait for in-flight pages to finish; completed records survive interruption.
Do not launch two preparation commands against the same output directory.

Start at 4 workers and measure before increasing to 8. Progress reports count
results collected in input order, so a slow early page can temporarily hold the
displayed count while later pages are already checkpointed.

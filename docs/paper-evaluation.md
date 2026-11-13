# Final evaluation reporting

`web-evaluate` now writes `paper-report.md`, `paper-summary.json`, and
`paper-summary.csv` next to its existing reports. No training or dataset rebuild
is required. The new files report final rendered geometry IoU, center/size error,
pixel MAE, and official Design2Code Block, Position and CLIP metrics when
`--official-repo` is enabled. Text and Color scores remain in the upstream raw
evaluation output but are excluded from the paper tables and paired comparisons:
text/content and color correction are outside the repair task. CLIP and pixel MAE
are supplementary whole-image checks, not direct measures of geometry recovery.
Geometry scores are diagnostic Hungarian-matched geometry, not official scores.

Repeated trials are averaged within each page. Each metric includes its evaluated
page count and a 95% page-bootstrap confidence interval (2000 resamples, seed 42).
Paired gains compare each method with the same pages' initial result; abstract
methods are also compared with screenshot-policy. Positive gains always mean
better. Improved/worsened/unchanged rates use a 1e-8 tolerance on page means.
Pipeline failures keep their evaluated fallback results. A page with a missing
repeat metric is excluded from that metric, never assigned a zero. Failure rates
remain visible. Different pages from one domain are not independent clusters in
this interval; the result is explicitly a page-level interval.

Cost reporting separates full pipeline seconds from repair seconds (including
target preprocessing). Visual repair counters include browser executions,
screenshots and actions when available. Missing counters are N/A. Startup and
final scoring are excluded from repair timing. Abstract feedback still performs
browser layout/reflow; screenshot counts must not be interpreted as reflow counts.

Rebuild tables from a completed evaluation without model loading or rendering:

```bash
python -m framediff.paper_report \
  --metrics runs/YOUR_EVALUATION/evaluation/metrics.jsonl \
  --out runs/YOUR_EVALUATION/evaluation
```

Replace the example directory with the directory containing your metrics.jsonl.
These are final image/geometry results, distinct from symbolic declaration-distance
diagnostics logged during training. No exact teacher-action match metric is added.

### Evaluation reuse and failed Initials

`web-evaluate` now retains failed-Initial rows for coverage without rendering or
computing Position, CLIP, or Box IoU. These rows carry
`evaluation_skipped: initial_generation_failed`; scores are absent, not zero.
Repairs that fail after a valid Initial still evaluate the retained HTML.
On resume, legacy failed-Initial score caches are converted to skipped rows;
valid existing evaluation caches remain reusable.

Within a page, identical embedded HTML shares quality scores and the rendered PNG
across methods. Method-specific failures, token counts, and repair times remain
separate. Hits record `evaluation_cache_hit: true` and zero evaluation browser
executions. The cache is page-local, so target, viewport, and scoring settings
cannot cross-contaminate other pages.

The official scorer also caches OCR-free extracted blocks by HTML content,
viewport, browser instance, and upstream source hash. Its LRU holds at most 16
entries; upstream matching receives a deep copy to prevent mutation of cached
blocks. This cache lives only within the evaluation process. It does not persist
font-dependent results across machines or change official metric formulas.
Different predicted HTML still requires fresh scoring (including CLIP); standalone
target CLIP feature caching is not implemented.

## Evaluation timing diagnostics

`web-evaluate` prints flushed JSON `evaluation_stage` events (`started`,
`completed`, `failed`, or `cache_hit`) with page ID and method. Each newly
computed row in `metrics.jsonl` also contains `evaluation_timing`:

- `stage_seconds`: exclusive wall time (nested stages are subtracted).
- `stage_inclusive_seconds`: wall time including nested work; do not sum these.
- `stage_calls`: invocation counts.

Stages separate `render_dom_snapshot`, `pixel_comparison`, `dom_box_matching`,
`official_render`, `block_html_preprocess`, `block_pixel_extraction`,
`block_matching`, `clip_image_preprocess` (masking/inpainting and resizing), and
`clip` (remaining CLIP preprocessing, transfer, encoding and similarity).
The inclusive `clip` duration covers the full CLIP metric. Upstream `.item()`
waits for the resulting GPU similarity, so this is not only kernel launch time.
`official_other` accounts for remaining upstream scoring work.
Reference preparation and official initialization (including CLIP model loading)
are logged separately from per-method scoring.

Existing evaluation caches remain reusable and are not recomputed to obtain
historical timings. In-memory score cache hits do not copy the original compute
timings into the new row. Restart the evaluator with the updated code to profile
new evaluations; an already running process does not acquire these changes.

Official evaluation also uses bounded process-local caches for rendered PNGs,
CLIP masking/resizing, CLIP image features, and ordered text-pair similarities.
PNG/masked-image caches each retain at most 16 entries / 32 MiB; the feature
cache retains at most 32 entries / 1 MiB. Cache hits are logged. Features are
cloned before upstream in-place normalization, and feature reuse is disabled
under autograd, training, or autocast. Mask cache keys include exact image bytes
and ordered boxes, not just the page ID. Browser changes clear image/block caches.

For recognized upstream OCR-free functions, exact vectorized pixel operations
replace per-pixel Python loops and repeated PNG decoding; color tolerances,
integer overflow behavior, coordinate order, and averaging remain unchanged.
Unknown upstream pixel/text implementations fall back to their original code.
`official.json` records which fast paths are enabled. Existing on-disk score
caches and their signatures remain valid; no dataset preparation or VLM
regeneration is needed. These optimizations do not reduce resolution, truncate
blocks, change CLIP precision, or cap the official merging search.

To aggregate actual work from a resumed evaluation log (also while it runs):

```bash
python -m framediff.evaluation_report \
  --log log/d2c-pilot10-seed42-n2-static-v5-gpu01.log \
  --out runs/d2c-pilot10-seed42-n2-static-v5/timing-report
```

The command prints a report and writes `evaluation-timing.json` and
`evaluation-timing.md`. It ranks exclusive stage time and reports percentages,
call counts, mean/max duration, failures, and CLIP's combined share. Group totals
and detailed stage totals describe the same work, not additive totals. Initial
CLIP/model loading is separate from per-page CLIP scoring. It reads stage events
only: method-summary totals and historical timings in cached score files are
never added again. Page-score, identical-HTML method-score, and internal-operation
cache hits are reported separately as access counts (not unique page counts).

Multiple distinct process logs may follow `--log`; do not supply mirrored copies
of the same execution. Appended restarts count actual repeated work. Old logs
without timing events cannot provide past timings. Pending/interrupted calls and
malformed partial lines are reported; their missing durations cannot be inferred.
The denominator is measured stage work, not full experiment elapsed time or GPU
busy time. Report generation does not modify experiment results or caches.

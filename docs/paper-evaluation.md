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

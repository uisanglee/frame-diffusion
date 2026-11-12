# Geometry-based protection and stopping

Enable `--preserve-matched-geometry` in `visual-evaluate` or
`visual-test-rollout`. It requires the full-visible DOM abstract CSS-owner
replacement policy. The option disables the RGB pixel-goal check even if its
usual default is enabled. It is mutually exclusive with
`--preserve-matched-regions`. Old checkpoints are used unchanged and old
unprotected/pixel caches are not overwritten; use a new output directory.

## Matching and protection

Target elements come from the usual detector once (threshold unchanged), or
from the explicitly requested oracle ablation. Current visible primitives come
from the existing DOM observation collector. Matching never uses reference DOM
IDs, ground-truth CSS values, or evaluation Box IoU. It uses same-class box IoU
only to establish a conservative correspondence:

- Target and current primitive must be each other's best same-class IoU match.
- IoU must be at least 0.5.
- The score must exceed the second-best candidate on both sides by at least 0.1.
- Each of x, y, width and height must agree within the axis tolerance.

Axis tolerance is `max(geometry_tolerance_px, viewport_axis *
geometry_tolerance_fraction)`, default `max(2 px, 0.002 * axis)`. At 1280x720
this is 2.56 px horizontally and 2 px vertically. These are explicit heuristic
defaults, not benchmark-derived or trained thresholds. Change them only with
a separate output directory. Geometry matching uses visible clipped xyxy
primitive boxes; the tolerance check converts them to xywh.

A DOM owner is protected only when **all** its visible primitives satisfy the
criteria. A shared CSS rule is excluded only if all matched DOM elements are
protected. Missing detections, unmatched current primitives and ambiguous
correspondences are unknown, not correct. Font/color values are not compared;
font changes that actually change text bounding boxes can still count as
geometric differences. Detector category errors can prevent correspondence.

The original DOM xywh of a newly protected element is anchored for the rest of
the rollout. After each proposal, the actual DOM boxes are measured. If any
protected box moves/resizes by more than 0.25 px (serialization tolerance),
or disappears, the candidate is rolled back. Persistent anchors prevent
successive sub-tolerance edits from accumulating drift. A rollback blocks that
CSS owner for the current state; accepted edits reset this temporary block.
Identical-value proposals are skipped. Blocking an entire owner is conservative
and can suppress another useful action on it.

## Stopping

`geometry_goal` requires every current AND target visible primitive to have an
unambiguous aligned one-to-one match. It is checked before policy inference and
also after the last allowed accepted edit. It means agreement with the available
target representation, not proof that the detector captured the entire real UI.
If no eligible editable owner remains, stop with
`no_unprotected_editable_owners`; this is **not** counted as a geometry goal.
Normal step/time limits and scroll preservation still apply. A step budget
counts proposals, including no-ops and rollbacks, not accepted edits alone.

This is inference protection, not a learned STOP action and not a guarantee of
monotonic whole-page quality. No pixels are used for guard/goal decisions, but
ordinary capture/scroll verification may still take screenshots. Existing
policy observations and final evaluation metrics are unchanged.

## Reuse the previous 20 samples

Run `bash scripts/run_geometry_protection_audit.sh`. Defaults:

- input: `runs/tuide-s-structure-severity-v1/inputs` (no new preparation)
- checkpoints: `runs/webui-color-images-v1/7000/detector/best.pt` and
  `tuide-s-abstract/replacement/abstract-stage2/best.pt` under the same root
- output: `runs/tuide-s-geometry-protection-v1`
- 4 severity levels, budgets 1/5/20, oracle and detector conditions
- original comparison: `runs/tuide-s-structure-severity-v1`
- pixel comparison: `runs/tuide-s-matched-region-v1`

`README.md` has the usual severity results; `comparison.md` and
`comparison.json` compare all three existing runs. They are regenerated from
saved results after each budget. Missing groups and differing trial cohorts,
initial scores or checkpoint/manifest fingerprints are reported. The
comparison does not rerun the old experiments. Restarting reuses completed
geometry evaluation caches.

Environment overrides: `SCALE`, `TRAIN_RUN`, `SEVERITY_INPUTS`,
`SEVERITY_BASELINE`, `PIXEL_BASELINE`, `GEOMETRY_OUTPUT`,
`ABSTRACT_CHECKPOINT`, `DETECTOR_CHECKPOINT`, `SEVERITY_BUDGETS`,
`SEVERITY_WORKERS`, `GEOMETRY_TOLERANCE_PX`,
`GEOMETRY_TOLERANCE_FRACTION`.

Trace files contain protected owner IDs, target/current primitive indices,
IoU correspondence confidence, xywh errors, unknown/unmatched counts,
geometry-goal decisions and violated protected DOM IDs. Rollout metrics retain
`preservation_rollbacks`, `preservation_seconds`, `noop_skips`, and
`geometry_goal_reached`.

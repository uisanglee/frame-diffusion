# Protect already matching DOM regions

Opt in using `--preserve-matched-regions` on `visual-evaluate` or
`visual-test-rollout`. Requires the CSS-owner replacement policy. Existing
checkpoints are unchanged; this is an inference guard, not a learned STOP head.
Disabled by default so the previous baseline and caches remain comparable.
Use a separate output directory when enabling it.

Before each proposal, compare the target screenshot and current RGB screenshot
inside each current DOM box, expanded by eight CSS pixels and clipped to the
viewport. Every pixel/channel must match within `--pixel-goal-tolerance`
(default 0.005). Exclude an owner only if all its matching DOM elements have
matching regions. This supports shared CSS rules without assuming detector
IDs correspond to current DOM IDs. No reference HTML, target DOM, Box IoU or
teacher values enter these decisions.

After an edit, take a screenshot. If it disturbs any region that matched before
the edit, restore the last accepted HTML and block that owner for this state.
This protects matching siblings from layout reflow. Also skip identical-value
proposals. Reconsider blocked owners after a successful state change. Stop
when no grammar-valid unprotected owner remains. Budgets count proposal
iterations, so a skipped/rolled-back proposal still consumes a step. This is
conservative: blocking an owner can miss a useful alternative edit on that
owner. The guard does not promise global quality improvement.

Limitations: matching pixels are evidence, not proof of correct geometry.
Blank/occluded regions can match accidentally, and different fonts/assets or
antialiasing can prevent matching even when geometry is correct. The check is
restricted to the visible viewport. Parent/child box overlap can make it
conservative. It does not solve the learned policy's generalization problem.
It adds screenshots, comparison and possible rollback overhead; these are
included in rollout wall time and browser counts. `preservation_seconds`
tracks guard computations/verification (initial screenshot remains in normal
render timing); it is not an additional exclusive timing total.

Traces record `protected_ids`, `accepted`, `rejection`, and `violated_ids`.
Statistics include `preservation_rollbacks`, `noop_skips`, and
`no_unprotected_editable_owners` as a possible stop reason. The stop reason
means there are no remaining eligible proposals under the guard, not that the
whole UI was proven correct.

Run `bash scripts/run_matched_region_audit.sh` to reuse the previous 20-page
severity experiment inputs without generating new corruptions. Defaults use
`runs/tuide-s-structure-severity-v1/inputs` and save guarded evaluations to
`runs/tuide-s-matched-region-v1`. The same budgets, oracle/detector conditions
and disabled global pixel early stop are retained for a controlled comparison.
Compare its README.md with the original severity report. Already evaluated
pages in the guarded output are reused on restart.

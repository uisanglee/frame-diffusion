# Same-DOM severity diagnosis

Run `bash scripts/run_structure_severity_audit.sh` from the project environment.
Defaults: held-out `data/new/7000/css/policy-test.jsonl`, 20 unique pages,
one deterministic viewport per page, seed 42, existing TUIDE-s Stage2 and
Detector checkpoints under `runs/webui-color-images-v1/7000`.
No training, VLM calls, downloads, or replacement of existing preprocessing.

Each clean target HTML is re-rendered locally to obtain a consistent target PNG
and target boxes. A seeded selection of up to four existing, numeric px CSS
fields on owners with positive-size boxes is shared across all conditions.
The experiment does not replace the DOM, selectors, content or non-editable
styles. It changes width/height/margins only. Percentage/auto properties are
not selected. The changed fields may belong to the same element.

| Condition | Width/height multiplier | Margin displacement |
|---|---|---|
| clean | 1 | 0 |
| mild | 4^(+/-0.1) | +/-2.5% of viewport axis |
| medium | 4^(+/-0.5) | +/-12.5% of viewport axis |
| strong | 4^(+/-1) | +/-25% of viewport axis |

The direction is fixed per field. Explicit padding/border minimum sizes remain
respected. Finite CSS serialization precision is tolerated, but unselected
fields must remain unchanged. Every condition has a detached-document
DOM/non-editable-CSS signature check and a teacher restoring the original
values; its rendered pixels must match the clean target. A failing page is
reported and excluded from **all** severity groups. No replacement pages are
silently selected. Invisible/inert changes are retained and measured.

The same Stage2 weights run with detector-derived and oracle target
abstractions at budgets 1, 5 and 20. Each budget starts from the same saved
corruption (not the previous budget output). Pixel-goal early stopping is
disabled, including for clean controls; production scroll rollback remains.
Actual actions, failures and rollback are in rollout results/traces.
The oracle is for diagnosis only. Checkpoint training-overlap and geometry
contract checks remain in the existing frozen-test evaluator.

Outputs under `runs/tuide-s-structure-severity-v1`:
- `inputs/coverage.json`: selected/retained counts and preparation errors.
- `inputs/severity.jsonl`: sites, edits, actual RGB/abstraction MAE, aligned IoU,
  and teacher-restoration checks.
- `inputs/pages/*/{target,clean,mild,medium,strong}.{html,png}`: inspectable inputs.
- `evaluations/<level>/budget-<N>/results.jsonl` and page traces: repair outcomes.
- `README.md`, `summary.json`: aggregate comparison, non-monotone severity warnings.

Re-running the command reuses completed generated pages and existing fixed-test
evaluation caches. The evaluator may still load the small checkpoints for its
validation; no completed page is deliberately re-inferred. Changed experiment
settings/checkpoints require a new output directory. Partial reports are
updated after each condition/budget.

Overrides: `SCALE=m`, `SEVERITY_SAMPLES=10`, `SEVERITY_SITES=12`,
`SEVERITY_BUDGETS="1 20"`, `SEVERITY_WORKERS=1`, `SEVERITY_OUTPUT=...`,
`ABSTRACT_CHECKPOINT=...`, `DETECTOR_CHECKPOINT=...`, `SEVERITY_DATA=...`.
Use separate output directories for Stage1 vs Stage2 or different site counts.

## Interpretation

- Strong measured visual corruption hurts oracle repair: magnitude/error-state
  generalization is a plausible problem even with the original DOM.
- Only detector repair degrades: investigate detector target representation.
- Clean pages degrade: policy stopping/action selection needs attention.
- Strong corruption is repaired but VLM output is not: this supports examining
  different CSS/DOM patterns, missing elements, and reachable edit space.

This is a controlled test of **magnitude sufficiency**, not proof that structure
alone causes VLM failures. The selected synthetic pages and VLM pages differ,
and a large nominal scale can produce a small visual change due to clipping or
CSS constraints. Check measured abstraction distances and saved screenshots;
if needed run more sites in a new directory. Aligned Box IoU here uses known
same-DOM IDs and must not be numerically equated with cross-DOM Hungarian IoU
in the VLM table. Existing VLM results are untouched.

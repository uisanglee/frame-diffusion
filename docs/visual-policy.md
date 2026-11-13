# Visual layout repair

TUIDE refines an existing webpage with structured CSS edits. Current training
and evaluation use the declaration-tree policy, browser-rendered geometry and
the same prepared page identity across repair methods. This page replaces the
older procedural-data training guide.

## Current workflow

- [CSS-first WebUI preparation](webui-css-first-preparation.md): CSS archive,
  image replacement, target/Detector labels, safe normalization and page splits.
- [Mixed corruption](mixed-corruption.md): clean, mild, medium, strong and
  multi-element training states, geometry logs and fixed validation/test states.
- [Policy training](numeric-policy.md): S/M/L oracle Stage1, frozen-Detector
  Stage2, screenshot baseline and compatible cache reuse.
- [TUIDE scales](tuide-scales.md): controlled evaluation and model scales.
- [Paper experiments](paper-experiment-suite.md): shared VLM initial
  implementations, repair baselines, metrics and tables.

The abstract policy conditions on target/current layout abstractions and current
HTML/CSS. The target Detector remains frozen during policy training and inference.
The screenshot baseline uses target/current RGB. Both require browser layout
updates after CSS edits. See [stylesheet-policy.md](stylesheet-policy.md) for
editable declarations and [geometry-preservation.md](geometry-preservation.md)
for protection of already aligned elements.

## Diagnostics

Keep the same pages and checkpoints when investigating repair failures:

- [Repair diagnostics](repair-diagnostics.md) compares Detector, target DOM and
  initial DOM coverage.
- [Policy cause audit](policy-cause-audit.md) compares training-state repair and
  Stage1/Stage2 behavior.
- [Same-DOM severity](structure-severity-audit.md) changes CSS magnitudes without
  changing HTML structure.
- [First worsening edit](vlm-first-edit-audit.md) checks feasible alternatives
  and whether Stage1 makes the same error on a VLM initial page.

## Legacy compatibility

Older visual policy and serialized-data readers remain so existing results and
checkpoints can be inspected. Retired procedural pretraining and small WebUI
fine-tuning launchers are no longer supported entry points. Datasets, caches,
weights and results are not removed by retiring those launchers. Changed
preprocessing or corruption contracts require a new run; resuming incompatible
old training is not a checkpoint migration.

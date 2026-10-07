# Synthetic Action Composition Benchmark

This held-out diagnostic isolates CSS repair by category and composition. It
does not train a detector/policy or replace the WebUI and Design2Code evaluations.

| Category | Corrupted existing declaration |
|---|---|
| W | width |
| H | height |
| X | margin-left or margin-right |
| Y | margin-top or margin-bottom |

All 15 nonempty subsets are evaluated. X/Y alternate left/right and top/bottom
across scenes. Isolated cards anchor to the corresponding side so each margin
visibly moves the selected element. Categories are CSS
intervention labels, not guaranteed pure geometric movements. In flow layouts a
width or margin edit can move several elements.

Each subset is crossed with same-element/distributed-element corruption and
isolated-grid/flow-flex layout. The default `--samples-per-cell 24` produces 1,440
cases from 48 base scenes generated with 24 seeds. Variants reuse the clean scene
and should be paired across methods. Do not treat 1,440 cases as 1,440 independent
pages; any confidence interval should cluster by scene_seed (including both
layout variants). One-category same/distributed cases are
intentionally duplicates and should not count as independent replications.
Inline/stylesheet owner types alternate by sample; corruption magnitudes cycle
through 8/24/48 CSS pixels, with both signs. Use 24 or a multiple of 24 samples per
cell for balanced owner/severity/sign counts. Seeds and assets are fixed on disk.

```bash
# Build once, reused for all S/M/L and RGB checkpoints. No GPU or VLM required.
python -m framediff visual-composition-build \
  --out data/composition-v1 --samples-per-cell 24 --resume

# Verify generation and scoring before evaluating a model.
python -m framediff visual-composition-evaluate \
  --data data/composition-v1/manifest.jsonl \
  --out runs/composition-sanity --sanity --steps 4 --device cpu

# Use actual checkpoint file paths; checkpoints are not trained by this script.
CUDA_VISIBLE_DEVICES=3 SAMPLES_PER_CELL=24 \
bash scripts/run_composition_benchmark.sh \
  /path/to/abstract/best.pt /path/to/detector/best.pt /path/to/screenshot/best.pt
```

The third positional argument (RGB policy) is optional. `COMPOSITION_DATA`,
`COMPOSITION_OUT`, `REPAIR_STEPS`, and `DEVICE` can override paths/budget/device.
Use a distinct output directory per model scale. A changed checkpoint, seed or
budget requires a new output directory. `--resume` reuses completed cases after
configuration and input asset checks.

For one checkpoint only:

```bash
CUDA_VISIBLE_DEVICES=3 python -m framediff visual-composition-evaluate \
  --data data/composition-v1/manifest.jsonl \
  --checkpoint /path/to/abstract/best.pt \
  --detector-checkpoint /path/to/detector/best.pt \
  --steps 20 --device cuda --out runs/composition-s-abstract --resume
```

Ordinary abstract evaluation runs the frozen detector once per rollout; `--oracle`
explicitly substitutes DOM target masks. RGB uses the same initial HTML and
target screenshot. All use the existing rollout decoder, no target CSS or boxes
for selecting actions and no target-based early stop. The target is used only
after rollout to compute metrics. No exact single-teacher action accuracy is
reported. `--sanity` executes known corrections and must never be presented as a
learned-model result. It is not a latency baseline.

Outputs: `report.md`, `summary.json`, `results.jsonl`, plus per-case `trace.json`,
`final.html` and `final.png`. Summary breakdowns include all 15 combinations,
category count, placement, layout, owner kind, magnitude and full cells.

* `improving_action_rate`: fraction of applied edits reducing the number of
  unequal CSS declaration children, allowing any improving action/order.
* `geometry_improving_action_rate`: fraction reducing mean absolute x/y/w/h
  error over matched DOM elements, measured separately from symbolic improvement.
* `symbolic_full_recovery`: all supported declarations restored (strict CSSOM
  value/priority equality). Equivalent CSS spellings can fail this diagnostic.
* `geometric_full_recovery`: all observed DOM x/y/w/h values within 1 CSS pixel
  of target by default; configurable via `--geometry-tolerance`.
* `category_recovery`, `recovered_W/H/X/Y`: corrupted declarations restored.
* `collateral_declaration_rate`: initially correct declarations changed.
* `collateral_geometry_rate`: initially correct element boxes moved outside the
  recovery tolerance (viewport excluded).
* Initial/final geometry IoU, center/size errors, first-hit step, ever-recovered
  rate and fixed-budget final recovery expose both progress and later regression.
* Repair `seconds` excludes final screenshots and diagnostic action replay;
  startup is excluded. Failures retain initial state and remain in recovery
  denominators; nullable metrics include their own `_n` counts.

These scenes are deliberately small and use px values. Positive results establish
controlled composition capability, not generalization to arbitrary web CSS or
VLM-generated HTML. Keep this test seed out of training and model selection.

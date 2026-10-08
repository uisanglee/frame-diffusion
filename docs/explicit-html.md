# Parent-preserving HTML repair and boundary abstractions

TUIDE never detaches/reparents DOM elements during preparation, training, or
rollout. The flat mode and visual-restore-hierarchy command are removed. There is
no hierarchy sidecar, virtual-parent clipping, or final restoration pass.

## Existing data and models

- Use original HTML or existing non-flat rendered corpora. Old assets marked
  data-tuide-flat / tuide-hierarchy, or records with hierarchy metadata, are
  rejected. Return to original sources; deleting their markers does not restore
  their parents. Original data files are not modified by this change.
- Existing non-flat HTML, screenshots, box labels, and detector box caches remain
  usable. New semantic tensors are assembled from boxes at training time, not
  read from old colorized abstraction PNGs.
- Detector weights remain usable: its box/class output format has not changed.
  Detector quality must still be evaluated; boundaries cannot recover boxes it
  never detected.
- Train a NEW abstract policy in a NEW run directory. Four input planes became
  eight; the paired current/target/absolute-difference encoder has 24 input
  channels rather than 12. Old replacement-policy and CSS-diffusion abstract
  checkpoints are rejected explicitly. RGB remains 9 paired channels.
- Older numeric-head v1 models retain their explicitly versioned legacy
  representation for historical comparisons, not as the new TUIDE model.
- Use a new evaluation output directory; caches use the DOM-boundary contract.

## Abstraction v2

semantic-occupancy-boundary-pair-v2 consists of:

1. Four class occupancy planes: text, image, control, painted region.
2. Four corresponding boundary planes: the union of each individual box's
   antialiased, one-output-pixel rectangular ring.

Rings are generated before union. A small same-class card inside a larger card
therefore remains visible in boundary planes even when occupancy is unchanged.
Targets from detector boxes and current DOM boxes use the same rasterizer,
including in CSS diffusion. This is not an instance-ID representation: coincident
identical boxes and sub-resolution differences are not guaranteed distinguishable.
No promise of perfect geometry follows merely from a low abstraction error.

Clipping follows the CURRENT real DOM ancestors. RGB uses ordinary browser
screenshots. Abstract visible rectangles use ancestor overflow bounds while full
box geometry remains available for actions. Rounded/complex clipping, opacity,
and stacking are not represented exactly by box abstractions.

## Explicit style normalization

`prepare_stylesheet_policy.sh` runs `visual-tree-normalize` on a copy of the
clean HTML. The targeted-inline-v4 normalizer reads the original page once and
adds editable inline values only to tagged nodes that appear as visible
abstraction boxes. It retains their original box sizing, DOM structure and CSS.
Browser probes test whether min/max or flex sizing blocks a size edit; only
observed blockers are neutralized. Unsupported logical declarations such as
`margin-inline` are retained, with only their conflicting physical fields
excluded from actions. Transformed, inline, table and embedded nodes remain
read-only. An independent reload checks geometry, clipping and global/local
pixels against the original; changed pages are rejected.
Accepted normalized HTML and screenshots are saved once and reused on resume.
Geometry and global/per-element pixel checks reject changes to the clean view.
This changes the controlled learning environment; it does not establish
performance on arbitrary unnormalized VLM HTML.

Each normalized page produces a fresh seed corruption. Old corruption labels,
CSS owners and screenshot caches are not copied across the normalization.
Fixed val/test states are generated from these new clean targets. Train noise
is online and never runs normalization again. Detector weights are reused;
target prediction caches are reused only when image hashes/provenance match.
The new data, cache and policy output paths prevent resuming an old run silently.
Direct real-web evaluation does not automatically normalize VLM-generated HTML.

For a separately chosen explicit-style experiment:

    python -m framediff visual-explicit-html \
      --html path/to/initial.html --out runs/explicit-preview \
      --width 1280 --height 720 --mode boxes

The supported modes are sizes and boxes. Both retain existing element parents.
sizes materializes selected CSS dimensions but retains flow/padding. boxes additionally
uses absolute positions under the SAME parent and bakes removed padding offsets
into positions; it does not preserve original responsive layout rules.
Single-line direct text may be wrapped locally without moving it outside its
original UI component. Controls' descendant text remains suppressed by the
control abstraction rule.

In the separate boxes mode, computed styles are captured before edits, then author stylesheets are removed.
Verification compares element geometry and global/per-element image differences.
Only accepted pages produce normalized.html; originals are never overwritten.
In boxes mode, generated pseudo-elements, transforms, fragmented text, and unsupported embedded
content are rejected. Sizes mode preserves these instead. External resources must be bundled; browser requests remain
blocked. This is a single-viewport conversion, not responsive reconstruction.

For explicit boxes, width/height cannot be smaller than border+retained-padding
sums. Replacement decoding masks invalid prefixes before selecting an action,
execution rechecks bounds, and symbolic validation uses the same bounds. Border
styles are unchanged. Normal, non-normalized HTML retains its existing action
semantics and browser layout constraints.

On rollout failure, logs/time/execution counts and the last available HTML are
retained with failed=true. There is no restoration stage or restoration cost.

## Size and margin actions

Replacement policies edit six fields: width, height and four margins.
Row-gap and column-gap are excluded from corruption, labels, decoding and
finite-state diffusion slots. Existing gap CSS remains fixed, along with
display, grid tracks and DOM parents. A width edit does not remove gap.

Online corruption samples existing supported size/margin declarations.
Candidate inspection probes up to four deterministic values per
owner/property and caches edits witnessed to change both layout and the clipped
occupancy/boundary tensor. Pages without a witnessed candidate are excluded.
This is finite probing, not a proof that no conceivable CSS value could work.
It does not invent overrides. Gap-only pages have no supported mutation sites.
Inspect online_teacher_property_distribution,
symbolic_by_predicted_property and freeze-report property counts for coverage.

The action/state schema changed. Do not resume old policy weights or reuse
eight-field labels. Reuse the original non-flat HTML/images and detector weights,
but reparse CSS labels and regenerate fixed validation/test corruptions:

```bash
CACHE_GPU=4 bash scripts/prepare_stylesheet_policy.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/train_tuide_scale.sh s
```

Defaults use data/webui-css-owners-v10-targeted-inline and runs/css-owners-v10-targeted-inline-shared;
training uses runs/tuide-targeted-inline-s-abstract (POLICY_MODES=screenshot selects RGB).
Old output directories are not deleted. Target detector predictions may be
reused by the cache command when their provenance matches. Train corruptions
are generated online, while held-out size/margin examples are fixed during preparation.

The first eligibility inspection is slower: it executes actual candidate edits.
Entries under .online-target-cache include successful candidates and exclusions,
are atomically written under a process lock, and are shared by model scales and
RGB/abstract runs. HTML, viewport, IDs/owners, representation code or observation
resolution changes invalidate entries. Transient browser errors are not cached.
Even rows with target_declaration_state are inspected. Older syntax-only caches
are not reused. No source HTML or existing detector weights are changed.

Every online forward edit is rechecked in its current context; unchanged
abstractions are reverted. After a reverse-path prefix, the current abstraction
must still differ from the clean target, and the chosen teacher edit must itself
change the abstraction. RGB and abstract use the same screening for matched
corruptions. The threshold is max absolute tensor difference >1e-6, not an RGB
pixel metric or a guaranteed perceptual improvement. Hidden control-internal
edits can therefore be excluded even when full screenshots differ.

These checks apply to replacement-policy online corruption and frozen held-out
states. The separate finite-state CSS diffusion experiment retains its specified
transition kernel: rejection filtering would change that probabilistic model.

Before processing the full server corpus, smoke-check real pages in a separate
directory (20 per split, not training-ready benchmark data):

```bash
python -m framediff visual-tree-normalize \
  --rendered data/webui-10k-v5-improvement-distribution/rendered \
  --out data/webui-targeted-inline-smoke --limit-pages 20 --resume
```

Inspect prepare-report.json, progress-{split}.json and rejections-{split}.jsonl.
Failures include their stage and are saved immediately, along with per-page
page-status.json; no need to wait for the split to finish. Per-page report.json
lists normalized and preserved elements and rendering errors. Normalization
uses one batch application instead of repeated rollback trials. Do not loosen visual QA to
increase the acceptance rate. Server WebUI coverage must be measured; local
regression fixtures do not establish that coverage. Existing v7 artifacts are
preserved but are not reused as v10 normalization results.

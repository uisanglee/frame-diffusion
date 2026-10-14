# Inline and stylesheet declaration policy

## Default: safe sizes with the original six-field CE policy

`prepare_stylesheet_policy.sh` now uses `flow-preserved-safe-size-margin-v1`.
Its default outputs are `data/webui-safe-size-margin-v1` (labels in the `-labels`
sibling) and `runs/webui-safe-size-margin-v1-shared`. `train_tuide_scale.sh`
uses these same defaults and new `runs/tuide-safe-size-margin-*` policy outputs.
Pass explicit source and detector paths as usual. Retired x/y and normalization-error recovery entry points have been removed.

The action schema remains width, height, margin-left/right/top/bottom. The policy
and token CE are unchanged; oracle stage 1 and frozen-detector stage 2 remain.
Clean labels are the accepted clean program's six CSS values, not x/y coordinates
or an undo journal. Online corruption and fixed held-out corruption retain the
existing visibility probes and six-field sampling contract.

The normalizer measures the original layout, tries measured px assignments and
only relevant min/max neutralization, and verifies viewport plus scroll pixels.
It never changes position, display, DOM hierarchy, flex behavior or grid rules.
If a combined assignment fails, individual fields are verified against the
original and only safe ones retained. Rejected fields remain original and are
excluded from actions. This does NOT promise arbitrary requested CSS sizes will
equal final rendered sizes: normal layout constraints still participate.
Reports include rejected fields, partial normalization and scroll checks.
The bounded field/scroll verification may leave unsupported fields untouched or
reject unverifiable pages; these are never treated as extra corruption labels.

New checkpoints record `safe_sizes=true`, enabling the same conversion on initial
HTML during rollout. Already prepared fixed test HTML is not converted twice.
Use new output directories; original rendering and detector weights can be reused,
but changed manifests need new conditioning-cache manifests. Existing trained
weights are not silently resumed with a different data/preprocessing contract.

## Parallel preparation and evaluation scope

`PREPARE_WORKERS` controls normalization, held-out candidate inspection, and
fixed val/test corruption generation. Each stage uses bounded page workers,
keeps input order and index-based seeds, and reuses each worker's Chromium
with isolated page contexts. Worker count may change on resume. Successful
per-page records are reused; incomplete pages are retried. Original rendering
uses `RENDER_WORKERS`. Lazy train pages are materialized by online workers.

These switches do not parallelize final test rollouts. Training validation CE
uses batches under no-grad; symbolic decoding and final browser rollout are
separate costs. Fixed test rollout is still sequential over pages, with its
existing per-page resume cache. No end-to-end speedup factor is claimed.

The `css-owner-tree-v2` policy edits six declarations (width, height and four
physical margins) on DOM inline styles **and embedded `<style>` rules**. An owner
is either a DOM ID or a style-block index plus nested CSS rule index path. Selector
text alone is not an identity: duplicate selectors are distinct owners.

For example, `.card {width:240px}` and `.container .card {width:300px}` are
separate positions. A mutation can change the second rule to `width:400px`, and
the teacher restores `300px`. All elements sharing that rule reflow together.
Rule memory includes current declarations, selector/path/conditions, matched
DOM memberships and their current box union. Target CSS is never model input.

## Training and inference

The v3 corruption sampler uniformly selects a **present, supported owner/property
pair** and replaces its value in place. It never inserts an absent declaration,
deletes a declaration, or changes its priority. For example, the actual
`.card {width:300px}` rule becomes `.card {width:240px}`; no inline override is
created. There are no hand-picked inline/rule or SET/REMOVE quotas. Each candidate is applied in Chromium and
accepted only when both a matched element's computed property and tracked DOM
geometry change. Shadowed/inactive rules therefore do not become no-effect
training corruption. Rejected candidates are reverted. This is a browser effect
probe, not a complete winning-declaration resolver or a target-distance search.

Actual current and clean declaration states define a shuffled exact repair path.
Training selects a random intermediate state and learns the next replacement
using ordinary token CE. This follows the program-path distinction in
[tree-diffusion mutator.py](https://github.com/revalo/tree-diffusion/blob/e8b29f27d6bd5e9fbf4679da49603df946b95eb1/td/samplers/mutator.py),
but the restricted CSS grammar, real fixed target corpus and browser effect
filter are our adaptations. It is not the complete TinySVG sampler.

This defines a value-repair training task: reverse labels are SET, so the expected
`online_teacher_remove_rate` is zero. This is deliberate task scope, not evidence
that the model learned REMOVE. REMOVE remains executable for existing unwanted
declarations, but v3 value-only training does not supervise that capability.
Pages without supported effective values can be excluded; inspect coverage.
Logs include `online_teacher_rule_rate`,
`online_teacher_remove_rate` and `online_probe_rejections`. Validation reports
token CE and sample counts separately for inline/SET, inline/REMOVE, rule/SET and
rule/REMOVE in `validation_by_teacher`. Inspect `freeze-report.json` for coverage;
a corpus with few editable embedded rules can still have little rule coverage.

At inference the same owner inventory and current context feed the decoder.
New stylesheet checkpoints record `existing_values_only=true`: only currently
present supported fields may be selected, and SET preserves priority. This
prevents creating inline overrides or promoting shadowed rules with !important.
It is not a complete cascade winner mask; effect probes still filter training
mutations. Old checkpoints retain their original decoding behavior.
Rule edits are serialized back into `<style>` text, so `page.content()`, saved
HTML and subsequent browser loads retain changes. RGB policy captures current
screenshots; abstract policy queries current DOM geometry and builds masks.

## Reuse existing targets, prepare fixed held-out corruption

Run from the repository root in the existing Python environment:

```bash
CACHE_GPU=4 bash scripts/prepare_stylesheet_policy.sh
```

Defaults reuse `data/webui-10k-v5-improvement-distribution` (or its `rendered/`
subdirectory) and the existing detector at
`runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt`.

The script writes new paths:

- `data/webui-css-owners-v3-labels`: cached HTML parsed with stylesheet owners
  (the default command reuses `data/webui-css-owners-v2-labels` if complete).
- `data/webui-css-owners-v3`: training target references plus **new, fixed val/test
  corrupted HTML/screenshots**, one state per held-out page by default.
- `runs/css-owners-v3-shared`: manifests pointing to detector predictions.

Original target screenshots and source HTML are reused. No train screenshot
corpus is regenerated: fresh current train states are produced online. Held-out
current screenshots do need to be generated once, because old held-out states
mostly test inline override removal. The frozen-data signature includes the
corruption version, so an old directory cannot silently be resumed.
`SOURCE_LABELS` explicitly selects an existing parsed stylesheet corpus.
`FIXED_SAMPLES_PER_PAGE` increases fixed
states per page (not the number of independent pages). Each prepared page is
resumable and deterministic by sample index. Rejections are recorded explicitly.

Existing detector outputs under `runs/css-owners-v2-shared` are reused by image
hash after verifying checkpoint hash, threshold and abstraction size. Set
`REUSE_TARGET_CACHE` to another compatible cache root if needed. Missing target
predictions are inferred with the frozen detector; the detector is never trained.
Keep referenced old cache assets on disk.

The preparation script accepts `[SOURCE_CORPUS] [DETECTOR] [DATA_OUT] [CACHE_OUT]`.
`MAX_CSS_OWNERS` defaults to 512 including DOM and rule owners; overflowing pages
are reported as rejected, never truncated. Use the same limit during training.

## GPU 1 / GPU 3

After preparation completes, and after previous jobs release the GPUs:

```bash
mkdir -p log
ONLINE_WORKERS=2 nohup bash scripts/train_stylesheet_policy_gpu1.sh \
  > log/css-owners-v3-screenshot.log 2>&1 &
ONLINE_WORKERS=2 nohup bash scripts/train_stylesheet_policy_gpu3.sh \
  > log/css-owners-v3-abstract.log 2>&1 &
```

Each script runs stage 1 then stage 2. Maximum steps remain 30,000/15,000;
learning rates are 1e-4/1e-5, batch size 2, accumulation 4, validation every 500,
early-stop patience 10. Existing environment overrides still apply. Both stages
use new online corruption; abstract stage 2 switches to detector target masks.

Outputs are `runs/css-owners-online-v3-screenshot` and
`runs/css-owners-online-v3-abstract`. Start new policy runs because corruption and
decoding constraints changed; do not resume v2 runs. Rerunning the same v3 commands
resumes matching `last.pt` checkpoints. Detector weights and target screenshots
are reused. Historical checkpoints remain loadable for comparison.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=1 \
PAGE_MANIFEST=data/webui-css-owners-v3/pages-test.jsonl \
RAW_CHECKPOINT=runs/css-owners-online-v3-screenshot/replacement/raw-stage2/best.pt \
ABSTRACT_CHECKPOINT=runs/css-owners-online-v3-abstract/replacement/abstract-stage2/best.pt \
DETECTOR_CHECKPOINT=runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
PAGE_LIMIT=0 REPEATS=3 REPAIR_STEPS=20 ORACLE_ABLATION=1 \
bash scripts/run_visual_web.sh data/raw/webui/test runs/eval-css-owners-v3 runs/css-owners-v3-shared
```

Report geometry/pixel changes and execution time on the same frozen initial
states. The number of independent pages is in `freeze-report.json`, not the
page × repeat count. Old inline-only and new mixed-corruption test scores are
different tasks and should not be presented as a direct policy improvement.

For the held-out controlled WebUI test, prefer `visual-test-rollout`. Its
`coverage.json` and `report.md` recover the complete provenance chain:
originally selected test pages, upstream corpus failures, normalization
rejections, fixed-corruption rejections, evaluation-limit exclusions, rollout
failures, and per-method end-to-end page success. `summary.json` remains the
quality/cost summary on the actually evaluated fixed corruptions; do not report
it without the accompanying coverage table.

## Supported scope

Embedded ordinary style rules and nested `@media`, `@supports`, and `@layer`
blocks are editable. External `<link>` stylesheets still participate in layout
but are not edited. Embedded `@import`, CSS nesting inside style rules and
overlapping logical/all declarations are rejected explicitly. Keyframe and
pseudo-element-only rules are not owner candidates. Selectors must match at least
one tracked DOM element. No selectors/rules/DOM nodes are inserted or removed.
Unsupported unchanged values remain visible to the encoder; `calc()`/`var()`
replacement targets are outside the finite grammar and are not sampled.

Probe filtering adds CPU browser cost. Local tests cover parsing, specificity,
shared rule reflow, nested paths, persistence, cache reuse, online training and
resume. Large-dataset throughput and GPU convergence require server experiments.

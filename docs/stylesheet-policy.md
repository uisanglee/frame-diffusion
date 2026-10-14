# Inline and stylesheet declaration policy

## Default: safe sizes with the original six-field CE policy

`prepare_stylesheet_policy.sh` now uses `flow-preserved-safe-size-margin-v1`.
Its default outputs are `data/webui-safe-size-margin-v1` (labels in the `-labels`
sibling) and `runs/webui-safe-size-margin-v1-shared`. `train_tuide_scale.sh`
uses these same defaults and new `runs/tuide-safe-size-margin-*` policy outputs.
Pass explicit source and detector paths as usual. For scaling, leave
`BOX_LAYOUT=0` and `RECOVER_NORMALIZATION=0`.

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

## Parent-preserving pixel boxes (new, independent protocol)

Use `bash scripts/prepare_box_layout.sh RENDERED NEW_ROOT` to prepare
`NEW_ROOT/labels` and fixed held-out examples in `NEW_ROOT/css`.
`RENDERED` must contain existing `policy-{train,val,test}.jsonl` and
`pages-{train,val,test}.jsonl` plus their referenced assets. `PREPARE_WORKERS`
controls workers. Re-running resumes verified page records and retries failures.
Original rendered HTML/images are reused read-only; old normalized outputs and
old policy checkpoints must not be used as this protocol's clean programs.

The `parent-relative-xywh-v1` protocol measures all geometry before writing any
styles. DOM parent-child relationships remain unchanged. Supporting HTML boxes
are also frozen to avoid unselected siblings reflowing, but only tagged visual
targets are editable. Targets use parent-containing-block-relative `left/top`
(action names `x/y`) and border-box `width/height`, all in px. Padding/borders
are retained; size actions respect their minimum physical size. Moving a parent
moves its children. Changing a parent's size does not rescale its frozen children.
This replaces automatic flow with fixed-viewport positioning, not responsive CSS.

The serialized result is reloaded and checked for pixel, target-box, clipping,
scroll-region and DOM-parent preservation. Reports and before/after PNGs are in
`labels/normalized-pages/SPLIT/HASH/`. Unsupported geometry or a failed preservation
check is a reported conversion failure, **not** extra training corruption.
Inline formatting islands and table subtrees retain their original layout and
are excluded from individual box actions (`preserved_noneditable_targets` in
the report); their editable ancestors can still move them. Fixed elements keep
viewport-relative fixed positioning, unlike ordinary parent-relative boxes.
BODY geometry and zero-size containing blocks are handled explicitly.
Transforms and sticky positioning are currently unsupported. This mode
cannot be combined with `--recover-normalization`.

Preservation now also compares rendered pixels at tiled document scroll positions
and separate nested-scroller positions, plus scroll extents. It does not claim
all combinations of nested offsets are tested. More than 128 required frames is
an explicit verification failure, not an accepted unchecked page. Verification
PNG pairs are retained in each page's `scroll-before/` and `scroll-after/` folders.
Use a new output root after this change: the cache contract includes converter
and scroll-guard code hashes, so old weaker verification results cannot resume
silently as new results.

Online corruption changes one or more target coordinates/sizes by sampled
±8/16/32 px steps (sizes clamped above their border/padding floor). The clean
measured box program provides exact restoration labels. CE predicts owner,
one of `x/y/width/height`, and a replacement px value. There is no REMOVE,
margin action, original-CSS undo journal, or visual teacher search in this path.
Validation/test use fixed corruptions of the same clean measured program.

Before using `train_tuide_scale.sh`, generate new `visual-cache-targets` manifests
for `NEW_ROOT/css/policy-{train,val}.jsonl`, with the existing detector checkpoint,
into a new shared run directory. Matching old predictions can be reused with
`--reuse-cache`; the detector does not need retraining. Then pass explicit
`s|m|l NEW_ROOT/css DETECTOR NEW_SHARED_RUN NEW_POLICY_RUN` arguments to
`train_tuide_scale.sh`. Stage 1/2 keep their oracle/detector conditioning and CE;
the four-field contract is detected from the manifests. Never use the legacy
default data/run paths for this experiment.

Use `visual-test-rollout --data NEW_ROOT/css/policy-test.jsonl --pages
NEW_ROOT/css/pages-test.jsonl` with the new policy checkpoint(s). It exports
target/initial/final HTML and screenshots and evaluates against the original
preserved render. Symbolic distance here means differing clean box coordinates,
not original CSS syntax. Ground-truth box values are supervision/evaluation only,
never rollout action selection inputs. Scaling preparation exposes `BOX_LAYOUT=1`
with a new output root; existing source rendering caches can still be reused.

## Optional normalization-error recovery

`scripts/prepare_normalization_recovery.sh RENDERED OLD_LABELS NEW_ROOT`
creates an independent recovery corpus in `NEW_ROOT/labels` and `NEW_ROOT/css`.
The target is the cached original HTML **server render before normalization**,
not the raw WebUI screenshot. Existing rendered images are referenced, not rebuilt;
verified old normalized HTML is reused read-only. All original clean rows are
considered, including pages absent from the old accepted manifest. Rerunning the
same command resumes completed page caches and retries failures. Keep the source
rendered assets and use a new output root, not an old training dataset directory.

Normalization preservation failures alone no longer exclude pages in this mode.
The `original-declaration-normalization-corruption-v2` contract records original
values, declaration absence and priorities in `normalization-journal.json`.
Labels restore those recorded values directly; there is no teacher search or
per-step visual-improvement acceptance test. Online sampling includes normalization
alone and normalization plus random corruption. A repair path may contain a
temporarily neutral or worsening edit: the label still restores an original
declaration. Parent-first supervision orders these direct labels by owner depth.

Normalization changes only supported width/height/physical-margin fields in this
mode. Non-action declarations such as min/max constraints remain original.
Unsupported original values (e.g. calc expressions) are preserved and their fields
excluded, not approximated with guessed labels. Original percentages, auto,
absence (REMOVE), and priority restoration are supported. Restoring all differing
declarations therefore has a known target, rather than a fitted approximation.
The journal is supervision metadata, not a policy input or an inference lookup.

Unstable original renders and technical browser errors still have separate
rejection records. A page without any supervised edit remains in held-out rollout
manifests; it is omitted only from CE and counted in `ce_supervision_coverage`.
No claim is made that every restored CSS declaration improves pixels individually.

Use `NEW_ROOT/css` as the data argument to existing `train_tuide_scale.sh` and
`visual-test-rollout` workflows, with **new policy output directories**. Regenerate
`visual-cache-targets` manifests for the new rows; `--reuse-cache OLD_PREDICTED_SPLIT`
can reuse matching detector/image predictions without retraining the detector.
Stage 1 and stage 2 remain token-CE training with oracle and detector conditioning,
respectively. Rollout evaluates original target images/boxes and exports both
`target.html` and `normalized-initial.html`. Symbolic distance is explicitly marked
as distance to original declarations, not pixel distance. Inference receives no
original target CSS. The existing rendering-preserving mode remains unchanged.

The scaling preparation script exposes the same option with
`RECOVER_NORMALIZATION=1` and optional `REUSE_NORMALIZED`;
use a new scaling output root when changing this data contract.
The former `--recovery-fit-steps` argument is accepted but ignored for compatibility.
Search-based v1 manifests must not be resumed as v2 training data; reuse their
verified normalization assets in a new output root instead.

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

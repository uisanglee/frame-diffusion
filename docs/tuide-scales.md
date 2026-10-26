# TUIDE policy capacity comparison

All scales share the frozen detector, semantic channels, CSS grammar, image size,
512 CSS-owner limit, fixed val/test corruptions, and prepared target cache.
Only hidden width, tree/decoder depth, and attention head count change.

The maintained policy trainer is `visual-tree-train` (`tree_policy.py`) for both
abstract and screenshot modes; `visual_train.py` handles detectors and frozen
target predictions. Retired delta-policy training (`visual-train-policy`) and
numeric-prefix preprocessing (`visual-subset-numeric`) have been removed.
Historical policy checkpoint loading/evaluation and existing caches are retained.

| Preset | Hidden | Layers | Heads | Abstract policy parameters |
|---|---:|---:|---:|---:|
| s | 128 | 3 | 4 | 4,766,449 |
| m | 256 | 6 | 8 | 15,532,325 |
| l | 512 | 8 | 8 | 67,087,477 |

Counts include the policy's spatial image encoder. The frozen detector adds
41,314,536 parameters separately. Counts assume the current stylesheet grammar
and 512 owners. Larger policies do not add supported CSS actions.

## Launch on physical GPUs 1, 3 and 4

Run from the repository with the Python environment activated. These commands
use the existing v3 corpus and detector cache; no preparation command is required.
Training still renders fresh online corrupted states.

Online CPU workers prepare pages independently. If one sample times out, only
its worker and browser descendants are restarted; other queued/completed results
are retained and consumed in their original index order. `ONLINE_TIMEOUT`
(900 seconds in `run_webui_scale.sh`) counts active sample work, including worker
startup, but excludes waiting for another process's shared cache lock. Such waits
have a separate `TUIDE_CACHE_LOCK_TIMEOUT` limit (default 3600 seconds). Timeout
logs identify `working` versus `cache_wait`. Increase `ONLINE_WORKERS` only when
CPU/RAM have headroom; `ONLINE_PREFETCH` bounds the pending result count.

If a retry times out again in `working` on the same page/view, the skip log reports
`page` and `excluded_view: true`. Records in the shared cache's
`online-timeout-exclusions/` persist across abstract/screenshot runs and resumes.
Subsequent selections skip that view before browser preparation; in-flight jobs
are retained. Other viewports remain eligible, and source/preparation changes
invalidate the exclusion. Lock waits, unidentified startup stalls and timeouts on
different views do not exclude a page. Delete a record to retry its view. Historical
timeout logs lack page identity and cannot be imported automatically.

Existing successful shared page caches remain compatible. Deterministic
`no_editable_css`/`no_visible_candidate` failures are cached under the same
source/settings key; transient browser and inspection errors are retried. A
source or preparation change invalidates these failures along with page caches.
Full-DOM observations use one browser query, with the same boxes, candidate
screening and teacher labels. No Train prewarm or held-out regeneration is needed.

```bash
mkdir -p log
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/train_tuide_scale.sh s > log/tuide-s.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/train_tuide_scale.sh m > log/tuide-m.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup bash scripts/train_tuide_scale.sh l > log/tuide-l.log 2>&1 &
```

Defaults: batch size 2, accumulation 4, seed 42, stage 1 up to 30,000 steps
at LR 1e-4, stage 2 up to 15,000 steps at LR 1e-5, validation every 500 steps,
2,000 validation samples, early-stop patience 10. Each stage loads its own
`last.pt` if present; stage 2 otherwise initializes from that scale's stage 1 best.
Use a new output directory for a different scale or changed run settings.
This training early-stop patience is based on validation loss; it is separate
from the inference-time pixel goal described below.

Validation keeps token CE for best-checkpoint selection and early stopping.
The existing fixed 128-state diagnostic subset additionally reports
`action_target_accuracy` (the predicted owner/property is one of the currently
faulty sites) and `exact_repair_action_rate` (one action removes a declaration
error). These reuse the same single prediction, with no additional render or
detector pass. Numeric spellings in the same unit, such as `480px`/`480.0px`,
compare equally; unitless length zero also equals `0px`. Units, percentages,
keywords and cascade priority remain distinct. Any faulty site is a correct
target, not just the single recorded teacher's site.

`action_metric_contract=css-action-diagnostics-numeric-v1` identifies this new
comparison. Existing `symbolic_*` metrics also use this normalization; historical
logs used literal strings. `symbolic_error_count_unchanged_rate` explicitly names
the legacy `symbolic_no_change_action_rate`: unchanged error count can include
partial CSS/visual improvement. Per-property `n` counts predicted sites, not
ground-truth corruption frequencies; `target_accuracy` and `exact_repair_rate`
use that property's prediction count. Invalid/unavailable predictions cannot
count as exact repairs; a valid site with an execution-invalid value can still
count as correct site selection. No new preprocessing or checkpoint is needed.

Override paths explicitly when necessary:

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/train_tuide_scale.sh m \
  data/webui-css-owners-v3 \
  runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
  runs/css-owners-v3-shared \
  runs/tuide-m-abstract
```

Final checkpoints:

```text
runs/tuide-s-abstract/replacement/abstract-stage2/best.pt
runs/tuide-m-abstract/replacement/abstract-stage2/best.pt
runs/tuide-l-abstract/replacement/abstract-stage2/best.pt
```

The default effective batch is 8. If L exceeds available VRAM, start a new run
with `BATCH_SIZE=1 ACCUMULATION=8`; keep effective batch and example budget
comparable across scales and report actual training steps/time. Equal maximum
steps with early stopping does not guarantee equal training compute. Select the
main scale on validation performance, not the held-out test set.

## RGB comparison and evaluation

The integrated scaling runner needs only S screenshot plus S/M/L abstract
checkpoints. It prepares initial HTML once, evaluates S with its matched RGB
baseline, then evaluates M and L in abstract-only mode in separate processes.
M/L screenshot training is optional.

```bash
CUDA_VISIBLE_DEVICES=1 TUIDE_SCALES=1 PAGE_LIMIT=20 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_visual_web.sh "$D2C_ROOT" runs/eval-tuide-scales \
  runs/webui-10k-v3-nospacing-fresh-webui
```

For already prepared evaluation pages, skip VLM preparation entirely:

```bash
CUDA_VISIBLE_DEVICES=1 PAGE_LIMIT=0 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_tuide_scales.sh \
  runs/existing-eval/prepare/prepared.jsonl runs/eval-tuide-scales
```

The combined report is `comparison/paper-report.md` under the scaling output
(under `scales/` when called through `run_visual_web.sh`). It contains shared
initial/self-revision rows where available, screenshot-policy (S), and TUIDE-S/M/L.
`policy-parameters.json` reports measured counts. Repeats are averaged per page.
Override paths with `RAW_CHECKPOINT`, `DETECTOR_CHECKPOINT`, and
`TUIDE_S_CHECKPOINT`, `TUIDE_M_CHECKPOINT`, `TUIDE_L_CHECKPOINT` as needed.
Capacity runs share the same prepared JSONL, seed, steps and detector. An S RGB
versus M/L abstract comparison changes both modality and capacity; use S RGB
versus S abstract for the isolated abstraction ablation.

### Inference-time pixel goal

Both screenshot and TUIDE abstract rollouts stop early by default when at least
99% of full-resolution pixels in the current browser render match the target
screenshot within a normalized RGB tolerance of 0.005 **in every channel**.
The check runs before the first policy action and after subsequent actions,
including the final action permitted by `REPAIR_STEPS` (20 by default).
`REPAIR_STEPS` remains a maximum, not a guaranteed number of edits. There is no
image-error-worsening rollback; the scroll-preservation guard is separate.
Evaluation results record `goal_reached`, `stop_reason`,
`pixel_match_fraction`, and the number/time of pixel-goal checks. This is a
pixel-count rule, not TinySVG's Gaussian-score rule in the public code.

In its normal, non-scaling evaluation path, `run_visual_web.sh` accepts
`PIXEL_GOAL_FRACTION` (default `0.99`) and `PIXEL_GOAL_TOLERANCE` (default
`0.005`). For example, to evaluate with a 99.5% match requirement, add
`PIXEL_GOAL_FRACTION=0.995` to the command's environment. For a fixed-step
comparison, set `PIXEL_GOAL_FRACTION=-1`. Direct
`visual-evaluate` and `visual-test-rollout` calls use the corresponding
`--pixel-goal-fraction` and `--pixel-goal-tolerance` flags. The
`run_tuide_scales.sh` path (also used when `TUIDE_SCALES=1`) uses the CLI
defaults and does not forward these environment overrides. Run comparisons
with the same pixel-goal settings. Using `--resume` with changed settings in
an existing evaluation directory is not supported.

Train a matched RGB policy using `POLICY_MODES=screenshot`:

```bash
CUDA_VISIBLE_DEVICES=1 POLICY_MODES=screenshot bash scripts/train_tuide_scale.sh m
```

`visual-evaluate` loads architecture dimensions from each checkpoint. Paired
RGB/abstract mode requires matched dimensions; `--abstract-only` evaluates any
single scale without an RGB checkpoint. For an optional matched M ablation:

```bash
CUDA_VISIBLE_DEVICES=1 \
RAW_CHECKPOINT=runs/tuide-m-screenshot/replacement/raw-stage2/best.pt \
ABSTRACT_CHECKPOINT=runs/tuide-m-abstract/replacement/abstract-stage2/best.pt \
DETECTOR_CHECKPOINT=runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
PAGE_LIMIT=20 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_visual_web.sh "$D2C_ROOT" runs/eval-tuide-m \
  runs/webui-10k-v3-nospacing-fresh-webui
```

For controlled capacity comparisons reuse the exact same prepared evaluation
records and initial HTML for S/M/L via `visual-evaluate --data ...`. Do not
regenerate initial HTML independently for each scale. Report final rollout
geometry, repair time, failures, training compute and policy parameter counts.

## Direct CLI

### Input-representation ablation

`ABSTRACTION_REPRESENTATION` in `train_tuide_scale.sh` (or
`--abstraction-representation` in `visual-tree-train`) selects:

| Value | One observation | Target/current/absolute-difference input |
| --- | --- | --- |
| `abstract-rgb` | Fixed class-colored filled boxes, ImageNet-normalized RGB | 9 channels |
| `occupancy` | Four class-specific filled masks | 12 channels |
| `boundary` | Four class-specific per-box boundary masks | 12 channels |
| `occupancy-boundary` (default) | Both mask sets | 24 channels |

Boundary-only is not an edge detector on the original screenshot: it retains
each detected/DOM box boundary, including internal boundaries of overlapping
same-class boxes. RGB uses the same boxes with deterministic drawing order,
not the original webpage RGB. It can hide overlapping elements; masks preserve
class-wise occupancy/edges. All variants use the same ResNet-18 architecture
except its input convolution, followed by the same repair decoder. Thus this
tests the representation, not a wholly different encoder architecture.

Reuse the same prepared policy data, detector checkpoint and frozen predicted
target cache. No relabeling or image regeneration is needed for this ablation.
Use a separate output directory per representation. For example, after setting
`POLICY_DATA` to the existing `.../7000/css`, `DETECTOR` to its detector `best.pt`,
and `SHARED_RUN` to the existing directory containing `predicted-train/data.jsonl`
and `predicted-val/data.jsonl`, run all four sequentially:

```bash
for rep in abstract-rgb occupancy boundary occupancy-boundary; do
  env CUDA_VISIBLE_DEVICES=2 POLICY_MODES=abstract \
    ABSTRACTION_REPRESENTATION="$rep" BATCH_SIZE=4 ACCUMULATION=2 \
    SEED=42 POLICY_STAGE1_STEPS=30000 POLICY_STAGE2_STEPS=15000 \
    EARLY_STOP_PATIENCE=0 \
    bash scripts/train_tuide_scale.sh s "$POLICY_DATA" "$DETECTOR" "$SHARED_RUN" \
      "runs/tuide-input-ablation/7000/$rep" || break
done
```

Stage 2 initializes from the same representation's stage 1. Existing `last.pt`
resumes only a matching configuration; do not initialize a different
representation from an already trained policy. Default old checkpoints remain
loadable as occupancy+boundary. Evaluation automatically reads the representation
from the checkpoint; it needs no representation override. Use the same fixed
test states, detector threshold, repair-step budget and seeds for every run.
Inference stopping uses the same full-resolution target-versus-current RGB
pixel goal for every input representation, and teacher generation is unchanged.
Equal optimizer-step budgets do not imply equal FLOPs: report wall time and
parameter counts too. Use multiple seeds for the final comparison.

`visual-tree-train --policy-scale s|m|l` resolves the preset before run/checkpoint
validation and overrides explicit `--hidden`, `--layers`, `--heads` values.
Without a preset the existing defaults/custom dimensions remain supported.
Only resolved dimensions are used for resume identity, preserving compatibility
with old S runs when all other configuration and input signatures match.

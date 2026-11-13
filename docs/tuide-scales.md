# TUIDE policy capacity comparison

Use the CSS-first targets and the mixed corruption recipe described in
[dataset preparation](webui-css-first-preparation.md) and
[mixed corruption](mixed-corruption.md). Historical screened-subset and inline-only
launch instructions have been retired. The current policy trainer is
`visual-tree-train` (`tree_policy.py`); Detector training and frozen predictions
use `visual_train.py`.

All scales share the detector, CSS grammar, image size, 512-owner limit,
prepared target cache and fixed validation/test states. Only width, depth and
attention heads change.

| Preset | Hidden | Layers | Heads | Abstract policy parameters |
|---|---:|---:|---:|---:|
| s | 128 | 3 | 4 | 4,766,449 |
| m | 256 | 6 | 8 | 15,532,325 |
| l | 512 | 8 | 8 | 67,087,477 |

Counts include the policy image encoder for the stylesheet grammar and 512 owners.
The frozen detector's 41,314,536 parameters are separate. Larger models do not
add supported CSS actions. Measure parameter counts for changed configurations.

## Data and run paths

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
mkdir -p log
export TRAIN_DATA="$PWD/data/new-css-v1/7000/css"
export TRAIN_RUN="$PWD/runs/webui-css-mixed-v1/7000"
export DETECTOR_CHECKPOINT="$TRAIN_RUN/detector/best.pt"
```

These are the scale wrapper defaults. Positional data/detector/shared/output
arguments still override them. Use a new training run when moving from CSS-blocked
or legacy corruption data to CSS-first mixed/STOP training. Existing checkpoints
remain loadable for evaluation; this does not make them equivalent to new training.

## Detector and Stage1

```bash
CUDA_VISIBLE_DEVICES=2 STAGE=detector \
  bash scripts/run_webui_scale.sh 7000 data/new-css-v1 runs/webui-css-mixed-v1

CUDA_VISIBLE_DEVICES=0 POLICY_STAGES=1 bash scripts/train_tuide_scale.sh s
```

Stage1 uses target DOM abstractions and needs no trained detector. Replace `s`
with `m` or `l` for other capacities. `POLICY_MODES=screenshot POLICY_STAGES=1`
trains a screenshot policy; screenshot mode has no Stage2.

## Predicted targets and Stage2

After the matching Detector finishes, build the shared predictions once:

```bash
for split in train val; do
  CUDA_VISIBLE_DEVICES=2 python -m framediff visual-cache-targets \
    --data "$TRAIN_DATA/policy-$split.jsonl" \
    --checkpoint "$DETECTOR_CHECKPOINT" \
    --out "$TRAIN_RUN/shared/predicted-$split" \
    --device cuda --size 384 --resume
done
```

Run each scale from its own completed Stage1:

```bash
nohup setsid env CUDA_VISIBLE_DEVICES=0 POLICY_STAGES=2 \
  bash scripts/train_tuide_scale.sh s > log/tuide-s-stage2.log 2>&1 < /dev/null &
nohup setsid env CUDA_VISIBLE_DEVICES=1 POLICY_STAGES=2 \
  bash scripts/train_tuide_scale.sh m > log/tuide-m-stage2.log 2>&1 < /dev/null &
nohup setsid env CUDA_VISIBLE_DEVICES=2 POLICY_STAGES=2 \
  bash scripts/train_tuide_scale.sh l > log/tuide-l-stage2.log 2>&1 < /dev/null &
```

Each stage resumes its matching `last.pt` if present. Stage2 otherwise starts from
that scale/representation's Stage1 `best.pt`. Keep the same data and configuration
when resuming. Checkpoint paths are:

```text
$TRAIN_RUN/tuide-{s,m,l}-abstract/replacement/abstract-stage1/best.pt
$TRAIN_RUN/tuide-{s,m,l}-abstract/replacement/abstract-stage2/best.pt
$TRAIN_RUN/tuide-s-screenshot/replacement/raw-stage1/best.pt
```

Defaults: batch 2, accumulation 4, seed 42; Stage1 up to 30,000 steps at LR 1e-4,
Stage2 up to 15,000 at LR 1e-5; validation every 500 steps, 2,000 validation
samples, early-stop patience 10. For lower memory use batch 1 / accumulation 8 in
a new run. Select capacity on validation, not test. Equal maximum steps do not
imply equal compute, especially with early stopping.

## Validation and corruption logs

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


The mixed recipe adds clean STOP examples and per-profile geometry/loss logs.
See [mixed corruption](mixed-corruption.md) for exact bins, coverage and filenames.
Training early stopping based on CE is separate from repair rollout stopping.

## Saved Initial evaluation

Use exactly the same prepared Initial HTML, seed, detector, repair budget and
cohort across scales. The integrated runner evaluates S abstract with its matched
S screenshot baseline, and M/L in abstract-only mode:

```bash
CUDA_VISIBLE_DEVICES=0 REPAIR_STEPS=20 REPEATS=1 \
  bash scripts/run_tuide_scales.sh \
    runs/existing-eval/prepare/prepared.jsonl runs/eval-tuide-scales
```

Set `OFFICIAL_REPO` for official Position/CLIP scoring. The output includes
`comparison/paper-report.md` and measured parameter counts. Override
`RAW_CHECKPOINT`, `DETECTOR_CHECKPOINT`, `TUIDE_S_CHECKPOINT`,
`TUIDE_M_CHECKPOINT`, `TUIDE_L_CHECKPOINT` when using an existing run.

Geometry protection is **opt-in**, using `--preserve-matched-geometry` on direct
`visual-evaluate`/`visual-test-rollout` calls. It disables the RGB goal and uses
position/size agreement. The scale runner retains its existing CLI stopping
behavior; it does not silently enable geometry protection on old experiments.
Use [the geometry comparison script](geometry-preservation.md) and separate output
paths to compare protected versus existing rollouts. A step budget counts
proposals, including rejected/no-op proposals, rather than accepted edits alone.

## Input-representation ablation


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


Reuse the same data, fixed states, detector and predicted-target caches, with a
new output for each representation. Stage2 must initialize from that representation's
Stage1. The checkpoint stores the representation for evaluation. Compare S RGB
with S abstract to isolate representation; S RGB versus M/L changes capacity too.
`scripts/train_webui_color_ablation.sh` provides screenshot, abstract-rgb and
occupancy variants. Set `TRAIN_DATA`/`TRAIN_RUN` to the matching dataset version.

## Online workers and cache reuse

Online CPU workers prepare pages independently. If one sample times out, only
its worker and browser descendants are restarted; other queued/completed results
are retained and consumed in their original index order. `ONLINE_TIMEOUT`
(180 seconds in `run_webui_scale.sh`) counts active sample work, including worker
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
With diagnostics off, newly materialized train caches store the clean state and
inspected candidates without generating a disposable seed corruption. Online
corruptions are still generated for each training sample. Existing seed-bearing
caches remain usable. Asset hashes are reused within each process while file
identity, size, modification time and change time are unchanged. Ancestor clips
and rectangles are memoized within one DOM observation and recomputed after edits.
Resume with the same data, shared cache and run directories after restarting the
training process to load this code; no new training option is required.


The cache reuse statements above apply to unchanged source/settings. Moving to
CSS-first targets or a new corruption recipe still requires the new manifests
and matching labels described in the preparation guide.

## Training resource lifecycle and stopping


Detector, abstract and screenshot training now handle SIGTERM/SIGINT by unwinding
worker contexts and cleaning up their browser descendants. Signal the training
**Python PID**, not only its launching shell. Find the exact run first:

```bash
ps -u "$USER" -o pid,ppid,args | grep -E '[p]ython.*framediff.*visual-(tree-train|train-detector)'
# Replace 12345 with the Python PID whose --out matches the desired run.
kill -TERM 12345
```

Wait for cleanup; do not immediately follow with SIGKILL. Resume with the same
training command from the latest saved checkpoint. Termination does not save a
partially completed optimizer step. SIGKILL, driver hangs and power loss cannot
be handled by this cleanup. Already running jobs must be restarted to load it.

Shared Chromium processes are recycled between contexts after 100 contexts by
default (`TUIDE_BROWSER_MAX_CONTEXTS=100`, positive integer). A corruption/repair
probe is never interrupted by this limit. This bounds browser reuse duration,
not RSS: one unusually heavy page can still require substantial memory. Cache
formats, data signatures, model settings and checkpoint formats are unchanged.
Worker target manifests remain process-local copies; increasing workers still
increases RAM use. Keep `ONLINE_WORKERS=4 ONLINE_PREFETCH=8` for these runs.

Stage 2 lazy target inference now processes at most **two cache misses per detector
forward pass**, independent of `BATCH_SIZE`/`ACCUMULATION`. Existing predictions
and checkpoints remain readable; precision and detector resize settings are
unchanged. This reduces inference peaks but keeps the detector resident on the
policy GPU. It does not precompute all targets or move detection to another GPU.

Online workers now have lightweight owner processes. On Linux each owner enables
child-subreaper mode so detached browser descendants remain reclaimable after a
rendering worker crashes. Owners receive no target manifest copy. A bounded
receiver thread keeps partial IPC responses from blocking timeout checks; retired
receivers are stopped when their workers are reclaimed. Linux-only orphan cleanup
can be checked on the training server with:

```bash
python -m pytest -q tests/test_worker_ipc_recovery.py tests/test_training_shutdown.py
```

These changes require restarting the training process with the updated code.
They are not a diagnosis or cure for GPU hardware faults or physical power loss.

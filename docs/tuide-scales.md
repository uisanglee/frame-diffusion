# TUIDE policy capacity comparison

All scales share the frozen detector, semantic channels, CSS grammar, image size,
512 CSS-owner limit, fixed val/test corruptions, and prepared target cache.
Only hidden width, tree/decoder depth, and attention head count change.

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

Train a matched RGB policy using `POLICY_MODES=screenshot`:

```bash
CUDA_VISIBLE_DEVICES=1 POLICY_MODES=screenshot bash scripts/train_tuide_scale.sh m
```

Existing `visual-evaluate` loads architecture dimensions from each checkpoint.
It requires matched RGB/abstract dimensions, so M abstract must be paired with
M screenshot, and L with L. For example, after both M policies finish:

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

`visual-tree-train --policy-scale s|m|l` resolves the preset before run/checkpoint
validation and overrides explicit `--hidden`, `--layers`, `--heads` values.
Without a preset the existing defaults/custom dimensions remain supported.
Only resolved dimensions are used for resume identity, preserving compatibility
with old S runs when all other configuration and input signatures match.

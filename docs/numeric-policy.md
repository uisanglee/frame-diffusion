# TUIDE declaration-policy training

`scripts/train_tuide_scale.sh` is the current S/M/L entry point. It calls
`scripts/train_numeric_policy.sh` with stylesheet editing, online mixed
corruption and prepared-data reuse enabled. The inline-only preparation wrapper
is retired; use [CSS-first preprocessing](webui-css-first-preparation.md).

## Inputs and stages

The policy edits existing width, height and four physical margin declarations
on supported inline and embedded stylesheet owners. It learns exact CSS
replacement paths. Mixed training includes clean states with STOP supervision;
see [mixed-corruption.md](mixed-corruption.md) for sampling and geometry logs.

| Mode / stage | Target conditioning | Training inputs |
|---|---|---|
| Abstract Stage1 | Ground-truth DOM abstraction | Published CSS manifests |
| Abstract Stage2 | Frozen Detector abstraction | Predicted-target manifests |
| Screenshot Stage1 | Target and current RGB | Published CSS manifests |

Screenshot mode has no Stage2. Model scale, page identities and preprocessing
must match when initializing Stage2 from Stage1. Validation/test states stay fixed.

## Prepare conditioning caches

Run from the repository root after preprocessing and Detector training:

```bash
export TRAIN_DATA="${TRAIN_DATA:-data/new-css-v1/7000/css}"
export TRAIN_RUN="${TRAIN_RUN:-runs/webui-css-mixed-v1/7000}"
export DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-$TRAIN_RUN/detector/best.pt}"
for split in train val; do
  python -m framediff visual-cache-targets \
    --data "$TRAIN_DATA/policy-$split.jsonl" \
    --checkpoint "$DETECTOR_CHECKPOINT" \
    --out "$TRAIN_RUN/shared/predicted-$split" \
    --device cuda --size 384 --resume
done
```

The command reuses compatible predictions. Changed target images, Detector
weights or manifests must not silently reuse incompatible labels. Stage1 can
run before these caches or the Detector checkpoint exist.

## Train or resume

```bash
POLICY_STAGES=1 bash scripts/train_tuide_scale.sh s
POLICY_STAGES=2 bash scripts/train_tuide_scale.sh s
```

Replace `s` with `m` or `l`. Set `CUDA_VISIBLE_DEVICES` for each process.
`POLICY_STAGES=all` runs both abstract stages. A screenshot baseline uses
`POLICY_MODES=screenshot POLICY_STAGES=1`. Representation ablations are exposed
by `scripts/train_webui_color_ablation.sh`.

Defaults are `TRAIN_DATA=data/new-css-v1/7000/css` and
`TRAIN_RUN=runs/webui-css-mixed-v1/7000`. Positional inputs take precedence:

```bash
bash scripts/train_tuide_scale.sh s \
  "$TRAIN_DATA" "$DETECTOR_CHECKPOINT" \
  "$TRAIN_RUN/shared" "$TRAIN_RUN/tuide-s-abstract"
```

`POLICY_RUN_DIR` overrides the default output when the fifth positional argument
is absent. Stage outputs are `replacement/abstract-stage{1,2}` or
`replacement/raw-stage1`. An existing `last.pt` resumes only matching data and
training contracts. Stage2 otherwise initializes from Stage1 `best.pt` in the
same policy run. Do not resume CSS-blocked or single-noise runs as CSS-first mixed
runs; use a new output directory.

Prepared HTML, target images and inspection caches are reused. Online workers
still render newly corrupted current states as required training work. Worker
count and queue depth can change without restarting the sample stream; see
[parallel-preparation.md](parallel-preparation.md).

## Evaluation and compatibility

Use [tuide-scales.md](tuide-scales.md) for controlled evaluation and
[paper-experiment-suite.md](paper-experiment-suite.md) for VLM-generated initial
implementations. Target DOM access is an explicitly privileged oracle ablation;
ordinary abstract repair uses Detector targets.

Legacy checkpoint/data readers remain for existing results. Their presence does
not make old preprocessing commands the current protocol. Keep referenced cache
and dataset files when reproducing old results; this cleanup does not migrate
weights.

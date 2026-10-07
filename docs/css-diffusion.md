# Optional finite-state CSS diffusion

This is an independent experiment inspired by LayoutDiffusion's discrete
transition/posterior approach, not a reproduction of its noise schedule or an
alternative setting for existing replacement-policy checkpoints. Existing
TUIDE training commands, data, detector and output directories are unchanged.
Reference: [LayoutDiffusion official implementation](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutDiffusion).

## State and objective

The state contains every supported existing numeric width, height and four
margin declaration. Inline and embedded stylesheet owners are supported.
Owner identity, unit, priority and declaration presence remain fixed. No REMOVE,
insertion, padding, gap, flex changes or external stylesheet editing is added.

Global grids (not ground-truth-centered grids) encode px and % values:

| Property | px range | % range |
|---|---|---|
| width/height | 0..2048 | 0..200 |
| margins | -1024..1024 | -100..100 |

Default 129 bins means 16px or 1.5625 percentage-point spacing. This is a coarse
baseline, not subpixel repair. Use a larger odd `--bins` for finer precision,
at the cost of quadratic kernel storage and more computation. Keywords, other
units and out-of-range values remain untouched. Unitless zero becomes 0px.
`record.json` records per-declaration quantization errors; new clean HTML and
screenshots are rendered after quantization so training labels and images agree.
An evaluation therefore measures recovery of these quantized targets, not exact
restoration of the original unquantized website.

Q_t is a row-stochastic lazy Gaussian transition on bins. Sigma grows linearly
from .35 to 2 bins across 200 steps; `move_rate=.2` mixes in an identity matrix.
A 1e-6 uniform component in the moving distribution gives full support. Every
slot transitions independently, so a timestep may change several declarations.
This schedule is a new CSS experiment, not LayoutDiffusion's original schedule.
The terminal distribution is not assumed to be uniform. Unconditional generation
from uniform noise is not implemented.

Training samples a page, a uniform t in 1..T and x_t from cumulative Q matrices.
The browser renders that state, and the image/CSS/owner encoder predicts clean
bin distributions conditioned on the target image, current image, current CSS
and t. All-slot predictions replace the autoregressive edit-string decoder.
The target reverse distribution is exactly

    q(j | xt=k, x0=i) = Qbar[t-1][i,j] Q[t][j,k] / Qbar[t][i,k].

The predicted reverse distribution is the mixture of these posteriors weighted
by p_theta(x0=i). Loss is posterior KL + auxiliary_weight * clean-bin CE,
averaged over slots and pages (uniform timestep average, not an unnormalized
sequence ELBO). At t=1 the posterior target is a point mass and KL is NLL.
There is no oracle action search, geometry rejection sampling or learned STOP.
Invisible/shadowed declarations may remain unidentifiable from images; this
limitation is not removed by defining an exact stochastic process.

## Prepare once in a separate directory

From the activated project environment:

```bash
python -m framediff visual-css-diffusion-prepare \
  --source data/webui-css-owners-v3 \
  --out data/webui-css-diffusion-v1 \
  --bins 129
```

This reads policy train/val/test files, deduplicates clean pages and preserves
source splits. It does not regenerate old corruption trajectories. Quantized
clean pages must be rendered once; inspect `summary.json` for exclusions. Use
`--resume` to reuse completed page records after interruption. For a smoke test
use `--limit 2` and a different output directory.

To cache a frozen detector's prediction of each NEW quantized target, add
`--detector-checkpoint runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt
--device cuda`. The detector is never trained. Do not reuse the original-image
detector cache, whose images may differ after quantization.

## Train separately

```bash
CUDA_VISIBLE_DEVICES=3 POLICY_SCALE=s POLICY_MODES=abstract \
bash scripts/train_css_diffusion.sh \
  data/webui-css-diffusion-v1 runs/css-diffusion-v1-s-abstract-oracle
```

Use a free GPU, not a GPU already occupied by existing training. `POLICY_MODES=screenshot`
trains the RGB comparison. `TARGET_SOURCE=detector` uses the frozen cache instead
of oracle masks and requires preparation with the detector option. These are
independent runs, not automatic stage1/stage2 training. S/M/L select encoder
capacity. Existing edit-policy checkpoints cannot initialize this new head.

`TRAIN_STEPS` counts optimizer updates; `DIFFUSION_STEPS` defines the corruption
chain. To resume, repeat the same command/settings with `RESUME=1`. Only
`TRAIN_STEPS` may be increased; optimizer, noise stream and random state resume
from `last.pt`. `best.pt` minimizes fixed-noise validation loss. Checkpoints are
saved on validation and the final step. This separate runner does not early-stop.

Training currently renders samples synchronously in the main process; browser
cost may dominate. No speed equivalence to the existing prefetched trainer is
claimed. Runtime browser failures fail visibly rather than silently resampling
a different corruption distribution.

## Controlled held-out rollout

```bash
CUDA_VISIBLE_DEVICES=3 python -m framediff visual-css-diffusion-evaluate \
  --data data/webui-css-diffusion-v1/test.jsonl \
  --checkpoint runs/css-diffusion-v1-s-abstract-oracle/best.pt \
  --out runs/css-diffusion-v1-eval-t200 \
  --start-t 200 --abstract-goal-threshold 0.01 --limit 20 --device cuda
```

The evaluator draws a known x_t, samples each reverse timestep and refreshes the
current rendering. Default start-t is 200, with an upper limit of 200 reverse
updates (and it cannot exceed the trained diffusion chain length). Before the
first update and after every update, it measures the semantic union error:
`sum(abs(target_mask-current_mask)) / sum(max(target_mask,current_mask))`.
This equals one minus soft, micro-aggregated semantic IoU and excludes empty
background from the denominator. Error <= 0.01 stops immediately by default;
the threshold is provisional and should be calibrated on validation data.
Empty target masks never trigger success. Set `--abstract-goal-threshold -1`
for a fixed-step comparison. The stopping target follows the checkpoint's oracle
or detector conditioning setting, including for the RGB model; oracle stopping
is therefore a controlled-benchmark feature, not deployable screenshot-only
inference. The same target mask is reused throughout the rollout.

Target CSS is used only to construct the controlled corruption
and diagnostics, not fed to the denoiser. The full numeric state is updated per
reverse step. Outputs include initial/final geometry IoU, repair time, traces,
initial/repaired PNG and repaired HTML. Results also record actual reverse steps,
stop reason, goal-reached rate, abstraction errors and every changed owner/property
value per step. No requirement forces a slot to change, nor prevents simultaneous
width/height/margin changes. Try multiple `--start-t` values with new
output paths; `--limit 0` uses all test pages. This is not yet wired into the
VLM/Design2Code paper comparison pipeline; these checkpoints have a separate
contract and must not be supplied to its edit-policy loader.

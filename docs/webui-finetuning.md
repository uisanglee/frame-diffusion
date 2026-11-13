# WebUI training workflow

The current workflow trains on CSS-first WebUI targets and mixed geometry
corruption. The previous procedural-pretraining and small WebUI fine-tuning
wrappers have been retired. Their data, checkpoints and results are not deleted.

1. Prepare or resume the dataset with
   [CSS-first WebUI preparation](webui-css-first-preparation.md). Inspect coverage
   and target screenshots before training. Existing page splits are preserved.
2. Train the Detector from `7000/rendered/detector-{train,val}.jsonl` using
   `scripts/run_webui_scale.sh` with `STAGE=detector`.
3. Train TUIDE Stage1 from `7000/css` with `POLICY_STAGES=1` using
   `scripts/train_tuide_scale.sh`. Stage1 does not require a trained Detector.
4. Build frozen Detector target caches, then train abstract Stage2 with
   `POLICY_STAGES=2`. See [numeric-policy.md](numeric-policy.md) for commands.
5. Evaluate held-out controlled pages and stored VLM initial implementations.
   See [tuide-scales.md](tuide-scales.md) and
   [paper-experiment-suite.md](paper-experiment-suite.md).

Training creates fresh current states online. Validation and test use the fixed
clean, mild, medium, strong and multi-element conditions from preprocessing;
see [mixed-corruption.md](mixed-corruption.md). Screenshot training uses one RGB
stage. Abstract Stage1 uses oracle DOM targets; Stage2 uses frozen Detector targets.

New CSS-first targets require matching Detector labels and policy conditioning
data. A different corruption contract requires a new training output. Resume
only the same experiment and keep files referenced by existing manifests.

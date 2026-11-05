# Screenshot-specialist models in the paper matrix

The default roster is LLaVA-NeXT-7B, GLM-4.1V-9B, WebSight VLM-8B,
Design2Code-18B-v0, and UI2CodeN. Qwen, Gemma, and Pixtral loaders and individual
runners remain available. New model combinations should use a new output root.
Existing source datasets, trained TUIDE weights, reference renders and prior
model outputs are not deleted.

## Input protocols and shared output budget

| Model | Input / preprocessing | Published generation settings |
|---|---|---|
| WebSight | Image-only special-token prefix; white alpha background; bilinear 960 x 960; published normalization | BF16, image-token bad-word IDs, `max_length=4096`; inherited model generation defaults (no temperature override) |
| Design2Code-18B-v0 | Single target image; empty query; official EVA/cross-image processors and Vicuna tokenizer; text processor length 2048 | BF16, `max_length=4096`, `temperature=0.5`, `top_p=1.0`, `top_k=1`, `repetition_penalty=1.1` |

Sources:
- https://huggingface.co/HuggingFaceM4/VLM_WebSight_finetuned
- https://huggingface.co/SALT-NLP/Design2Code-18B-v0
- https://github.com/NoviScl/Design2Code/tree/main/CogVLM

The table above records the **published examples**, whose 4096 cap counts the
entire sequence. Our comparison instead uses dataset-specific output budgets:
WebUI uses **16,384**, while Design2Code and Design2Code-Hard use **4,096**.
Within a dataset, every model and method uses the same output budget, including
Initial, supported Self Revision, and UI2CodeN polishing. The central paper runner
sets both `MAX_NEW_TOKENS` and `UI2CODEN_MAX_NEW_TOKENS` from the dataset;
older global values in the launching shell do not override this policy.
This is an explicit adaptation of those examples: 4,096 **output** tokens are
different from a 4,096-token total sequence cap. HTML
completion or the model's EOS can stop generation before the cap. Context
limits still apply to input plus output; raising the output budget does not
extend a checkpoint's supported context or guarantee that it fits GPU memory.

Actual settings, checkpoint revision, token counts, inference time and memory
are stored with every generation. Generic image downscaling is bypassed so each
model receives its official preprocessing exactly once. The new output-budget
configuration prevents reuse of drafts generated under a different budget or
the old total-4096 setting; use a new output root for the revised experiment. Reference-render and
model-download caches remain reusable.

Neither official example supplies a two-image/current-HTML Self Revision
protocol. These two groups therefore contain Initial, UI2CodeN polishing, and
TUIDE (when enabled), without a fabricated Self Revision result. The VLM-only
pilot omits TUIDE for every model. The target screenshot is shared with the
other methods, but these specialist models receive **no reference-text list,
viewport text, or asset instructions**. This input difference must be reported;
they are not silently treated as text-augmented chat models. Generated resource
references remain subject to the common rendering/resource policy.

## Installation and a small run

The legacy dependencies are isolated from the main `.venv` and UI2CodeN.
Environment package pins are compatibility choices, not hyperparameters claimed
by the papers. Do not install SAT or the old Transformers into the main venv.

```bash
cd /home/uisang/2027/frame-diffusion
bash baselines/websight/setup.sh
bash baselines/design2code/setup.sh
```

WebSight setup now loads its actual processor/tokenizer, checks the official
960-pixel image transform and imports the custom model class without downloading
weights. Repeat this lightweight check with:

```bash
baselines/websight/.venv/bin/python baselines/websight/check_setup.py
```

The processor uses built-in `IdeficsProcessor` (`trust_remote_code=False`); only
VMistral model/config loading enables remote code. The published preprocessor
`auto_map` contains bare class names, so forcing remote processor loading fails
in Transformers 4.37.2. `protobuf` is required by the published tokenizer's
`legacy=False` setting and is installed by setup. If FlashAttention2 is absent,
the loader explicitly selects the remote model's own eager text/vision attention
paths and logs that fallback; sampling settings, weights and image transforms
are unchanged. This setup check does not verify full GPU inference.

If needed, choose available Python interpreters with `WEBSIGHT_SETUP_PYTHON`
and `DESIGN2CODE_BOOTSTRAP_PYTHON`; see each setup script. The standard runtime paths
are `baselines/websight/.venv/bin/python` and
`baselines/design2code/.venv/bin/python`. Override with `WEBSIGHT_PYTHON` and
`DESIGN2CODE_PYTHON`. `DESIGN2CODE_REPO` selects the official repository with its
vendored CogVLM (not the generic CogVLM checkout).

```bash
mkdir -p log
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  D2C_PILOT_PAGES=3 \
  D2C_PILOT_OUTPUT=runs/d2c-pilot3-seed42-n2-official5-output4k-v1 \
  bash scripts/run_d2c_pilot10_bf16_2gpu.sh \
  > log/d2c-pilot3-seed42-n2-official5-v1.log 2>&1 < /dev/null &
```

This uses three fixed pages per Design2Code dataset, all five initial models,
two supported self-revision/polishing rounds, common metrics, and table export.
It never runs the different models concurrently on the same GPU pair.

For a first-load check before the matrix, select an existing target PNG:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python -m framediff vlm \
  --backend design2code-sat --model SALT-NLP/Design2Code-18B-v0 \
  --task generate-html --image data/raw/design2code/testset_final/10018.png \
  --out runs/specialist-smoke/design2code.html
CUDA_VISIBLE_DEVICES=0 python -m framediff vlm \
  --backend websight --model HuggingFaceM4/VLM_WebSight_finetuned \
  --task generate-html --image data/raw/design2code/testset_final/10018.png \
  --out runs/specialist-smoke/websight.html
```

The one-image commands also save raw output and model metadata. They are load /
generation checks, not an evaluation or a claim that generated HTML is valid.
Main matrix model revisions are pinned on first use and retained across resumes.
The source models' large checkpoint downloads happen only when used.

## Validation boundary

Local automated tests cover transforms, official settings, worker dispatch,
HTML preparation, cache reuse, lifecycle cleanup and table/roster logic. This
workspace has no RTX 4090 pair: actual checkpoint loading, GPU peak memory and
full inference compatibility must be checked on the server. An unavailable or
incompatible worker aborts the run instead of creating a batch of failed pages.

Design2Code's published example is single-GPU. The two-GPU adapter merges the
trained rank-8 LoRA deltas into base weights on CPU before SAT partitioning,
because SAT 0.4.10 does not correctly partition the unmerged LoRA factors.
This preserves the learned update mathematically, but BF16 rounding can differ
from the unmerged single-GPU path. This adaptation is recorded in metadata;
it is not claimed to be a bitwise-identical reproduction. See
`baselines/design2code/README.md` for CPU RAM and checkpoint storage requirements.

WebSight startup (including first weight download) has a separate `WEBSIGHT_STARTUP_TIMEOUT=86400` second limit. `WEBSIGHT_TIMEOUT=1800` continues to limit each generation request. Increase the startup limit for slow downloads; resuming preserves the Hugging Face download cache and completed experiment results.

Design2Code-18B downloads its checkpoint and extracts the ZIP in the parent process before starting the SAT worker. `DESIGN2CODE_TIMEOUT` does not limit this download/extraction; it limits worker readiness and generation waits. Stage logs distinguish these operations. Network request timeouts can still occur independently.

The default paper roster is LLaVA-NeXT-7B, GLM-4.1V-9B-Thinking, WebSight VLM-8B, and UI2CodeN. Design2Code-18B remains available only through explicit selection. WebSight has no self-revision stage; its saved Initial can be polished by UI2CodeN or repaired by TUIDE when enabled. Its first download/model startup may take up to 24 hours by default; per-request inference retains the 30-minute limit.

WebSight now uses both first visible GPUs when at least two are visible. Embeddings, the vision encoder, projection/resampler and output head stay on GPU 0 for image-token fusion; complete decoder layers are split (14/18 for the published 32-layer model), with final normalization on GPU 1. Accelerate dispatch hooks handle layer transfers. No CPU/disk offload or quantization is used. With one visible GPU it uses that GPU. Startup logs and generation metadata record the device map and per-device peak memory. This requires a real two-GPU smoke test on the server; local tests do not establish CUDA inference correctness or 16K memory capacity.

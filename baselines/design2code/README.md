# Design2Code-18B-v0 SAT runtime

The official checkpoint uses the **Design2Code repository's vendored CogVLM**, including
`FineTuneTestCogAgentModel`. The generic Transformers loader cannot load this SAT ZIP.

```bash
# From the project root; existing official evaluation checkout is reused.
# If it does not exist, clone it first:
# git clone https://github.com/NoviScl/Design2Code.git external/design2code-official
bash baselines/design2code/setup.sh
export DESIGN2CODE_REPO="$PWD/external/design2code-official"
export DESIGN2CODE_PYTHON="$PWD/baselines/design2code/.venv/bin/python"
```

The setup script requires Linux and Python 3.10 or 3.11 (`DESIGN2CODE_BOOTSTRAP_PYTHON`
can select its executable). It builds a separate environment without changing the main
`.venv`. SAT 0.4.10 is the version specified by Design2Code. Other pins in
`constraints.txt` are compatibility choices for that legacy runtime, not published
experimental hyperparameters. Dependency installation and two-GPU generation must
still be smoke-tested on the server; they were not GPU-tested on the development Mac.

The setup also installs and verifies `en_core_web_sm==3.7.1`, compatible with
spaCy 3.7.5. CogVLM eagerly imports its grounding parser, which loads this English
pipeline even for HTML generation. If an older setup stopped with spaCy E050,
rerun the updated setup; it reuses the existing environment.

Use exactly two visible CUDA GPUs, e.g. `CUDA_VISIBLE_DEVICES=0,1`. A persistent
`torchrun` worker loads weights once for the preparation process. Initial download is
29.1 GB ZIP plus space for unpacked weights. Checkpoint unpacking is cached under
`.cache/design2code-sat/checkpoints`; do not confuse it with generated results.
Loading also needs substantial host RAM: SAT loads the full checkpoint on CPU to
partition it. GPU memory is not the only resource constraint.

The official settings are: BF16, empty query, one target screenshot, checkpoint-defined
EVA and cross-image preprocessing, text-processor length 2048, total sequence
`max_length=4096`, `temperature=0.5`, `top_p=1.0`, `top_k=1`, and
`repetition_penalty=1.1`. The generic 16K output setting and generic chat prompts do not
apply. This model has no official target+current-image self-revision interface; the
experiment reports that method as unsupported rather than disguising a new initial
sample as revision. UI2CodeN and TUIDE can repair its saved initial HTML.

For two-GPU loading, trained LoRA deltas are merged on CPU before SAT partitions the
weights. This is necessary because SAT 0.4.10's generic splitter does not partition
LoRA factor tensors correctly. The merge preserves the original SAT parallel-linear
classes and uses the official merge formula, with FP32 delta accumulation followed by
casting to the checkpoint dtype. No LoRA weights are dropped. Small floating-point
rounding differences versus unmerged single-GPU evaluation are possible. This adaptation
is recorded in result metadata.

The immutable Hub revision, CogVLM source hash, exact generation settings, token counts,
per-GPU peak allocation, and generation time are recorded. Worker startup/load time is
outside per-page generation timing. Logs live under `.cache/design2code-sat/*.log`.
`DESIGN2CODE_TIMEOUT` defaults to 1800 seconds for startup or a single generation.
Workers terminate with their owner process, including when the parent is killed.

Sources:
- https://huggingface.co/SALT-NLP/Design2Code-18B-v0
- https://github.com/NoviScl/Design2Code/tree/main/CogVLM
- https://github.com/NoviScl/Design2Code/blob/main/CogVLM/finetune_demo/inference_design2code.py
- https://github.com/zai-org/CogVLM/blob/main/basic_demo/cli_demo_sat.py
- Published SwissArmyTransformer 0.4.10: `sat/model/base_model.py`,
  `sat/mpu/operation.py`, `sat/model/finetune/lora2.py`.

## Dependency audit

Audited the official repository at commit
`7a575e4c33f417c4be5c64072b8f5798de0d0f99`: all Python imports in CogVLM,
then the inference import closure in `utils/models`, `utils/utils`, and
`finetune_demo/inference_design2code.py`. The vendored requirements omit
`jsonlines` and `datasets` even though importing `utils.utils` eagerly imports
the dataset module. `requirements.txt` here explicitly supplies these and the
other direct inference dependencies instead of relying on transitive installs.
`datasets==2.18.0` and `pyarrow==15.0.2` keep the legacy dataset/Arrow APIs compatible.
Root Design2Code requirements are not installed wholesale into SAT: they also
contain evaluation/API dependencies and a different torch version.

The final setup audit scans imports in the actual server checkout, flags new
unmapped modules, imports all required packages, loads the spaCy pipeline, and
checks the exact worker symbols. Errors are collected together. This does not
load the checkpoint or certify CUDA kernel execution. Apex has an explicit
LayerNorm fallback in official `eva_clip_L_hf.py`; bitsandbytes is not used by
our BF16, non-quantized path. Neither warning is suppressed or confused with a
missing required import. Demo-only dependencies remain covered by the vendored
requirements; interactive Gradio demos are not an evaluation entry point.

To rerun only the audit without installing or generating anything:

```bash
baselines/design2code/.venv/bin/python baselines/design2code/check_dependencies.py \
  --repo external/design2code-official/CogVLM
```

# Architecture

## Data flow

```text
reference screenshot / target boxes
               │
               ├─ optional frozen VLM ──> identity-matched target frames
               │
initial LayoutIR ──> proxy executor ──> current frames
       │                                  │
       └──── tree + frame deltas ─────────┤
                                          ▼
                                  edit denoiser
                                          │
                                legal edit proposals
                                          │
                         proxy-scored beam search loop
                                          │
                                          ▼
                              repaired executable IR
                                          │
                          final HTML + Chromium validation
```

## Module boundaries

| Module | Responsibility | Must not do |
|---|---|---|
| `ir.py` | Schema validation, legal edits, proxy execution, HTML compilation | Load model weights or datasets |
| `data.py` | Synthetic trees, corruption, split and leakage rules | Evaluate on training groups |
| `benchmarks.py` | Design2Code preparation, batched VLM inference, dataset combination | Confuse Hungarian geometry with official visual metrics |
| `adapters.py` | WebUI import and surrogate fitting with fidelity metadata | Call fitted boxes original CSS |
| `model.py` | Feature encoding and edit-denoiser network | Access privileged clean trees |
| `train.py` | Sampling, losses, checkpoint and resume | Select checkpoints on test results |
| `search.py` | Legal proposal generation and budgeted repair | Mutate node topology silently |
| `evaluate.py` | Metrics, artifacts, bootstrap summaries | Use predicted targets as ground truth |
| `browser.py` | Isolated final geometry verification | Enable page JS or external network |
| `vlm.py` | Optional AR generation and frame extraction adapter | Act as the sole evaluator |
| `suite.py` | Sequential seed/ablation orchestration | Run multiple 24 GB jobs concurrently |

## Compatibility contracts

- LayoutIR has an explicit integer `version`; incompatible schema changes require a new version.
- Supervised pairs require identical node ID order and parent topology.
- Checkpoint resume requires identical model config and training objective.
- Evaluation rejects training-group overlap unless explicitly overridden.
- Predicted target frames require separate identity-aligned `evaluation_observations` or
  identity-free `reference_observations`; input targets are never reused as truth.
- Proxy/browser parity is a tested invariant of the supported DSL.

## Extension sequence

For a new layout property, update `FIELDS`, limits/defaults, validator, executor, compiler,
feature encoding, legal action masking, corruption, deterministic baseline, and parity tests together.
For topology edits, introduce an explicit action/schema version rather than overloading property edits.

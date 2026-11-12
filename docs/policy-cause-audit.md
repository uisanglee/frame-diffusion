# TUIDE policy cause audit

Run `bash scripts/run_policy_cause_audit.sh` from the repository environment.
Defaults: 20 unique validation pages (one fixed corruption per page), 8 freshly
sampled online corruptions using those pages, TUIDE-s Stage1/Stage2 and the
existing detector. No model training, dataset rebuilding, or VLM generation.

The audit checks:

* available original/normalized page pairs, screenshots and tagged geometry;
* saved declaration states against live browser CSS owners;
* teacher tokenizer round trips, symbolic improvement, and complete teacher
  repair replay in Chromium;
* all differing-site target values (fixed validation and fresh online samples
  reported separately; these are NOT the historical training frequency);
* Stage1/oracle, Stage2/oracle and Stage2/detector on identical sampled states;
* legality of each correct edit under the actual inference masks;
* per-token top-five logits before/after masking, instrumented/production
  decoder agreement, same-declaration predictions;
* actual CSS application, geometry changes, and same-DOM aligned IoU before/after
  (a diagnostic metric, not the paper's cross-DOM Box IoU);
* repeated actions/unchanged observations in the existing VLM budget traces.

Defaults are in `scripts/run_policy_cause_audit.sh`. Override `SCALE=m` or `l`,
`AUDIT_SAMPLES`, `AUDIT_ONLINE_SAMPLES`, `TRAIN_RUN`, `AUDIT_OUTPUT`, or explicit
checkpoint variables. Additional CLI arguments are forwarded, including
`--device cpu`, `--trace-root PATH`. `--data-only` skips detector/policy loading;
use a separate output directory for this mode because run provenance is guarded.

Outputs: README.md, error.log (combined portable diagnostic report),
checkpoint_comparison.json, predictions.jsonl, teacher_checks.jsonl,
preprocessing.jsonl, value_distribution.json, trace_checks.jsonl, errors.jsonl,
and per-sample screenshots/HTML/box JSON. Source manifests/checkpoints are not
modified. Small online diagnostic samples and inspection cache are written in
this audit directory. Re-running repeats the audit checks, not training.

Missing source HTML is unavailable, not a passing preprocessing test. Errors are
reported with stage and sample ID; compare matching IDs when a condition fails.
No claim of a unique root cause is made automatically. The script does not
reproduce historical online batches or rerun React applications: existing
application-transfer failures remain visible in the imported rollout traces.

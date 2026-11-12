# First worsening VLM repair edit

Run `bash scripts/run_vlm_first_edit_audit.sh`. Uses the three known failing
initials and their existing detector/oracle budget-20 traces. No VLM calls or
training. Outputs go to `runs/vlm-first-edit-audit-v1`.

1. Reproduce production safe-size normalization with a zero-step rollout.
2. Replay accepted historical actions, verifying pre-action boxes (1px tolerance)
   and CSS-owner identities. Compute reference DOM Hungarian Box IoU after each
   action using the existing metric and default evaluation node cap (2048).
3. Stop at the first per-step IoU decrease, not necessarily the first action.
4. On that exact pre-failure state, compare Stage1 and Stage2 with oracle and
   detector targets. These four scores describe static captures before rollout
   end guards/application transfer; they are not full pipeline scores.
5. Test up to 120 grammar-legal numeric CSS alternatives across owners/fields.
   Proposals use current values, nearby offsets/scales, and reference dimensions;
   they do not assume matching DOM IDs across implementations. No DOM changes.
6. Verify improving candidates under scroll preservation and, for React inputs,
   application patch transfer. Save an improving witness if found.

`improving_edit_found` proves only that a better allowed one-edit action exists
at that state. It does not prove complete recoverability. A finite search finding
none is inconclusive. This reference-guided search is diagnostic, never a fair
learned-policy baseline. Historical end-of-rollout rollback flags are recorded.

Set `AUDIT_CANDIDATES`, `SOURCE_RUN`, `TRACE_RUN`, `AUDIT_OUTPUT`, `TRAIN_RUN` or
checkpoint overrides in the launcher. `--cases` accepts a JSON list of triples
(dataset, initial model, page ID). Fixed inputs/checkpoints/traces are fingerprinted;
completed page/target-condition results are reused on an identical rerun. Failed
conditions are retried. Preserve the source experiment and diagnostic folders.

Deliver `error.log` (README + results + errors). Per-condition folders contain
trajectory.json, before-wrong-edit.html, policy comparison renders, search-trials.json,
individual candidate HTML/renders, and a verified witness when available.

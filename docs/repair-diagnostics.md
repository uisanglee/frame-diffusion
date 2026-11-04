# Detector and initial-layout diagnostics

`run_paper_experiments.sh` automatically runs observational diagnostics after each
complete dataset experiment. `PAPER_DIAGNOSTICS=0` disables this additional pass.
It does not change repair actions, checkpoints, datasets, or existing scores.
Diagnostic rendering/inference is excluded from repair latency.

For existing results, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m framediff.repair_diagnostics \
  --root runs/paper-pilot10-seed42-n2-bf16-2gpu-viewport-v2/webui \
  --dataset webui \
  --detector-checkpoint runs/webui-full-dom-screened-7k-v1/7000/detector/best.pt \
  --device cuda \
  --out runs/paper-pilot10-seed42-n2-bf16-2gpu-viewport-v2/webui/diagnostics
```

Replace `webui` with `design2code` or `design2code-hard` for the other datasets.
Use the actual experiment root if different. No VLM generation is repeated.

## Matching and outputs

Reference and initial HTML use the same full-visible-DOM classification and
visibility collector used for detector labels, rather than raw DOM node counts.
Detector output is boxes/classes, not a DOM tree. Matching is class-aware,
one-to-one, maximum-cardinality matching at IoU >= 0.5. The detector confidence
threshold defaults to 0.4; report it alongside Precision/Recall.

- `pages.jsonl`: each model/seed/page's boxes, match indices, unmatched counts,
  collector limitations (`label_audit`), and associated repair outcomes.
- `summary.json`: micro counts, detector class/view breakdown, diagnostic errors,
  and per-model/per-method outcome associations.
- `detector-table.md`: Precision, Recall, FP, FN, and target-rendering count.
- `repair-associations.md`: groups with no unmatched boxes, unmatched predictions
  only, unmatched targets only, or both; reports Box IoU change and improved/
  worsened percentages. N counts scored repair outcomes including repetitions.

Pair directions (prediction -> target):

| Pair | Unmatched predictions | Unmatched targets |
|---|---|---|
| detector_vs_reference | Detector FP | Detector FN |
| initial_vs_reference | Initial boxes unmatched to reference | Reference boxes unmatched to initial |
| initial_vs_detector | Initial boxes unmatched to detector | Detector boxes unmatched to initial |

For the latter two pairs, unmatched does NOT prove semantic absence: a shifted
or resized element can fail the geometric match. Detector errors can also be
localization errors, class confusion, or duplicate predictions. Detailed match
records retain these distinctions. Correlation with repair outcomes is not a
causal experiment; compare oracle-target and detector-target repairs on the
same initial pages to isolate detector influence.

Detector targets are deduplicated across initial models/seeds. Initial pairs
and repair outcomes remain separate. Initial-generation failures contribute to
detector evaluation, but not initial comparisons or repair associations. Missing
metrics are not counted as repair failures. A failed repair's retained valid
implementation is scored using the existing metric records.

Reference rerenders must exactly match the saved target PNG before new labels
are accepted; otherwise a diagnostic error is logged and excluded, not turned
into an FP/FN. Always report coverage/error counts. Label/prediction caches avoid
repeat rendering and detector inference; delete the diagnostics cache if the
browser/runtime environment changes. Source, image, checkpoint, threshold, and
collector changes invalidate relevant keys automatically.

## Full WebUI detector test

The pilot table covers only its selected target renderings. To report the full
held-out WebUI detector test, use the existing evaluator (all test renderings):

```bash
CUDA_VISIBLE_DEVICES=0 python -m framediff visual-evaluate-detector \
  --data data/webui-full-dom-screened-7k-v1/7000/rendered/detector-test.jsonl \
  --checkpoint runs/webui-full-dom-screened-7k-v1/7000/detector/best.pt \
  --out runs/webui-full-dom-screened-7k-v1/7000/detector-test.json \
  --device cuda --threshold 0.4
```

Use the detector dataset's actual `rendered/detector-test.jsonl` path if stored
separately. Report `precision_iou50`, `recall_iou50`, `n`, and `unique_pages`;
these are reference-box detection results, independent of VLM generation.

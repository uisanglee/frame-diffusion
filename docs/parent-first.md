# Optional parent-first TUIDE

Enable `PARENT_FIRST=1` with `scripts/train_tuide_scale.sh` (CLI:
`visual-tree-train --parent-first`). Default behavior is unchanged. Use a new
policy output directory; no detector retraining or corpus regeneration is needed.
Requires online corruption and stylesheet owners. Both stages record the option
in the checkpoint, and all callers of visual `rollout`, including fixed WebUI
test and real HTML evaluation, automatically use it when loading that checkpoint.

Training keeps the current DOM and online corruption. At each sampled state,
candidate clean-CSS replacements are ordered by current DOM ancestor depth;
only a replacement that strictly reduces semantic abstraction MAE is a teacher.
Invisible/non-improving parent repairs are skipped. This is supervised repair,
not RL, and is not a proof of the globally shortest repair path.
Fixed validation states are relabeled in memory using the same criterion;
source data is never overwritten. `parent-first-validation.json` records retained
counts and excluded IDs, so CE values with different validation subsets must not
be compared as if they shared a denominator. Test pages are never relabeled or
filtered by target CSS. Stage 2 still uses detector predictions as conditioning;
supervised teachers use the clean training target, not noisy detector geometry.

At test time the policy proposes one grammar-valid edit per owner, shallowest
first. Current DOM/CSS owners determine the order; no target DOM or target CSS is
consulted. A shared rule's depth is its shallowest matched current node. An edit
is committed only if target-observation MAE strictly decreases (semantic MAE for
abstract, RGB observation MAE for screenshot). Otherwise it is rolled back and
the next owner is considered. After an accepted edit, ancestors are reconsidered
because layout dependencies may have changed. DOM edges are never edited.

This checks *effective parent proposals*, not whether a parent is provably
correct. One greedy proposal may miss another useful edit on the same owner.
Unsupported/unselected DOM containers cannot be edited. No assumption that
detector boxes map one-to-one to DOM parents is needed.

`REPAIR_STEPS` / rollout steps cap accepted edits. The separate inference budget
`PARENT_FIRST_MAX_PROBES` (default 200, saved in checkpoint) caps proposal attempts;
the existing time budget also applies. Report attempts, rollback count and total
runtime alongside quality; this option performs more rendering than baseline.
Existing final below-fold safety checks can still revert the entire result.

Example after the shared v12 preparation has completed:

```bash
PARENT_FIRST=1 CUDA_VISIBLE_DEVICES=3 POLICY_MODES=abstract \
bash scripts/train_tuide_scale.sh s \
  data/webui-css-owners-v12-stable \
  runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
  runs/css-owners-v12-stable-shared \
  runs/tuide-v12-s-abstract-parent-first
```

Use that run's stage-2 `best.pt` in a fresh test output directory. Keep the same
held-out corruptions for baseline comparison:

```bash
CUDA_VISIBLE_DEVICES=3 python -m framediff visual-test-rollout \
  --data data/webui-css-owners-v12-stable/policy-test.jsonl \
  --pages data/webui-css-owners-v12-stable/pages-test.jsonl \
  --abstract-checkpoint runs/tuide-v12-s-abstract-parent-first/replacement/abstract-stage2/best.pt \
  --detector-checkpoint runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
  --out runs/test-v12-s-abstract-parent-first \
  --device cuda --steps 20 --limit 0 --repeats 1
```

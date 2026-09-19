# AMAUN — Assumptions Register

Every assumption made to proceed, its justification, and what breaks if it is wrong.
Assumptions are *not* decisions: decisions need a human (`docs/decision_register.md`).

| # | Assumption | Justification | Blast radius if wrong |
|---|---|---|---|
| A-01 | A `dummy` tier (tiny locally-built random Qwen2) is legitimate for tests | `idea.md` §14 gates real work behind passing smoke tests; those tests must run offline and in seconds. The tier is hard-blocked from reporting. | none scientific; only test coverage |
| A-02 | `Qwen2.5-*-Instruct` (not base) is the right checkpoint family | All six tasks are prompt/response formatted; an instruct backbone is the honest starting point and matches brief §3 | changes every absolute number; relative comparisons survive |
| A-03 | Loss is computed on response tokens only | Standard for instruction tuning; scoring prompt tokens would optimise the wrong objective | would change all adapters; large |
| A-04 | `primary_score` replaces `Acc` for generative tasks (D-009) | "Accuracy" is undefined for summarization/dialogue/long-form | changes CL metric values, not their ordering logic |
| A-05 | SUPPORT is carved from the training pool | Required so that support access can never imply test access (`idea.md` §16.5, brief §5) | leakage if violated — CRITICAL |
| A-06 | FWT reference `b_j` is the frozen backbone's zero-shot score | The only adapter-free reference available; alternatives (random) are meaningless for LMs | shifts FWT by a constant; ordering preserved |
| A-07 | Forgetting uses `max(0, ·)` per `idea.md` §4.5 | Taken verbatim from the source spec | asymmetric with BWT (which is signed); both are reported so the difference is visible |
| A-08 | Softmax-over-`m` α generalises `idea.md`'s sigmoid pair exactly (D-004) | `softmax([a₁,a₂])₁ = sigmoid(a₁−a₂)`; logits at 0 give 0.5 | none — provably equivalent at `m=2` |
| A-09 | Teacher logits may be cached (optionally top-p truncated) | Memory bound at 7B; truncation is recorded and its effect measurable | biases `L_KD` slightly; must be reported |
| A-10 | Compatibility = identical backbone fingerprint + target modules + rank | `idea.md` §7.3 demands explicit compatibility but does not define it; this is the minimal defensible definition | changes which adapters aggregate; `federated.compatibility.key` makes it configurable |
| A-11 | Per-adapter seeds are derived as `derive_seed(base, adapter_id)` | Guarantees adapter training is order-independent (the independence claim of brief §1) | if wrong, adapters are not independent — CRITICAL |
| A-12 | Truncated-proxy utility targets are acceptable if their bias is measured (D-010) | Full-convergence supervision is `O(n²)` full fusion runs at 7B | biases the utility network; `proxy_validation_frac` measures it |
| A-13 | `dirty` git trees are allowed under `dummy`/`local_smoke`, gated under `actual` | Iteration speed vs. reproducibility | unreproducible reported runs if the gate is removed |
| A-14 | The evaluated stream must include overlap tasks for REUSE/FUSE to be *correct* (D-002) | With six maximally distinct task types, CREATE is a priori always right | H2/H3 untestable — this is the largest methodological risk in the project |

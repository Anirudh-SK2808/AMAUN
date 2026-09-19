# AMAUN — Independent Reviewer Report (Agent 3)

**Date:** 2026-09-19 · **Scope:** `docs/**`, `src/amaun/**`, `tests/**`, `configs/**`,
`scripts/**`, `results/dummy/**` · **Method:** falsification-first. Every finding is labelled
`VERIFIED` (established by executing something) or `INFERRED` (established by reading).

This is not a summary of the code. It is an attempt to break the system's claims, and a report
of what survived.

---

## Verdict

**ACCEPT WITH REQUIRED FIXES.**

The implementation is internally consistent, and the invariants `idea.md` §16 names are enforced
in code rather than asserted in comments. One **CRITICAL** data-leakage defect was found and
fixed during this audit. The remaining blockers are not implementation defects: **no experiment
has been run**, so no research claim in `docs/master_prompt.md` is currently supported, and the
experiment as configured cannot yet test H2 or H3.

| severity | count | resolved |
|---|---|---|
| CRITICAL | 1 | 1 |
| HIGH | 7 | 3 |
| MEDIUM | 6 | 3 |
| LOW | 3 | 2 |

---

## Falsification questions: results

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Is the base Qwen model actually frozen? | **Yes** | `VERIFIED`. `tests/test_backbone_frozen.py`; 10 `assert_backbone_frozen` call sites; fingerprint identical before/after training and fusion; optimiser param groups contain adapter tensors only. |
| 2 | Were the initial adapters trained independently? | **Yes** | `VERIFIED`. `tests/test_adapter_independence.py` trains the same adapter in two orders and compares content hashes bit-for-bit. A first attempt *failed* here (dropout RNG depended on injection order) and was fixed by re-seeding after injection. |
| 3 | Is SUPPORT byte-disjoint from TEST? | **Yes** | `VERIFIED`. Disjointness is checked by content hash, not by trusting the partition; 276 split pairs audited clean across 6 tasks. SUPPORT is carved from the training pool, never the test pool. |
| 4 | Can the manager reach future-task or test data? | **Now no. It could.** | `VERIFIED` — see **CRITICAL-1**. The manager's continual features were sourced from the TEST-filled performance matrix. Fixed; regression test added. |
| 5 | Is CREATE genuinely necessary? | **Untested** | `VERIFIED`. All three outcomes execute, but with six maximally distinct tasks CREATE is a priori correct, so necessity is not demonstrated. See **HIGH-2**. |
| 6 | Is FUSE better than picking the best adapter? | **Untested** | No run exists. The comparison is wired (`retrieval.mode: single` vs `top_k` share one scoring path), so it is answerable. |
| 7 | Is REUSE reducing adapter growth? | **Mechanically yes, empirically untested** | `VERIFIED`. `tests/test_continual_runner.py` asserts REUSE leaves repository ids and byte count unchanged while `tasks_served` grows. |
| 8 | Does fusion add trainable parameters? | **Yes, and it is disclosed** | `VERIFIED`. `L·m` coefficients: the dummy run reports `alpha_parameters: 28` for 14 layers × m=2, logged in every fusion record. |
| 9 | Do baselines receive equivalent information? | **Information yes; budget no** | `VERIFIED`. Baselines share the identical feature pipeline, metrics, decoding config and training loop. But measured step counts differ by construction (B2: 12, B3/B4: 36 on the dummy stream). See **HIGH-3**. |
| 10 | Does KD improve continual learning? | **Untested** | `VERIFIED` that KD runs, that the teacher is the pre-merge adapter, that teacher capture precedes fusion, and that `λ=0` reproduces the no-KD arm exactly. |
| 11 | Is federated aggregation correct? | **Yes** | `VERIFIED`. Weights sum to 1; aggregation matches the weighted dense sum to 1e-5; incompatible adapters form singleton groups rather than being averaged; backbone tensors in a payload raise. |
| 12 | Are reported gains statistically meaningful? | **N/A — nothing reported** | `VERIFIED` that single-seed comparisons are refused and the shipped config carries one seed. See **HIGH-4**. |
| 13 | Can experiments be reproduced from saved config? | **Partially** | `VERIFIED`. `scripts/reproduce.py` re-resolves the config to an identical hash and diffs the environment. Untestable end to end because no `actual` run exists. |
| 14 | Could gains come from extra training rather than the method? | **Currently unanswerable, but gated** | `VERIFIED`. `BudgetLedger` refuses cross-arm comparisons beyond tolerance, so an unfair comparison cannot be reported silently — but it also means the matrix as configured will refuse most comparisons. See **HIGH-3**. |

Additional checks: `α_l = 0.5` exactly at init (`VERIFIED`, all parameterisations, all `m`);
ΔW-level fusion with a test proving A/B averaging differs (`VERIFIED`); normaliser provenance
enforced across splits and run modes (`VERIFIED`); utility network refuses `fit_split="test"`
(`VERIFIED`); compression is utility-based with the heuristic confined to B6 (`VERIFIED`);
`run_mode` artefact separation enforced at the write site (`VERIFIED`); seeds logged in
`config_used.json` (`VERIFIED`).

---

## Findings

### [CRITICAL] TEST performance fed the adapter-selection decision
- **Type**: METHODOLOGICAL ERROR
- **Category**: F (data leakage)
- **Location**: `src/amaun/continual/runner.py` — `_prior_scores`, consumed by
  `features/continual.py`
- **Status**: **RESOLVED** during this audit
- **Problem**: `_prior_scores` read prior-task scores from `self.matrix`, the performance matrix
  that the evaluator fills from **TEST** (`matrix.record(...)` at the single TEST read site).
  Those scores populate the `continual` family (`F_i`, `KR_i`, `P_i`, `S_i`), which enters `z_i`,
  which the utility network scores, which produces the REUSE/FUSE/CREATE decision.
- **Evidence**: `VERIFIED` by tracing the only writer of `self.matrix` (`grep -n "matrix.record"`
  → one site, fed by `Evaluator.evaluate` on TEST) to the only reader feeding features.
- **Why it matters**: test-set performance was influencing adapter selection, violating
  `docs/master_prompt.md` constraint 13, `idea.md` §16.5, and brief development rule 4. Nothing
  would have looked wrong — the pipeline ran and the numbers were plausible — and the leak would
  have inflated precisely the decision-quality result the paper claims. The typed
  `SplitAccessGuard` did not catch it because the leak travelled through *recorded scores*, not
  through a dataset handle.
- **Fix applied**: a parallel `validation_history`, filled through a guard permitting
  `{TRAIN, VALIDATION, SUPPORT}` and never TEST, is now the source for continual features.
  `tests/test_no_test_leakage_into_decisions.py` poisons the TEST matrix with sentinel values and
  asserts they never appear in the decision features.
- **Residual risk**: guards protect *handles*, not *derived statistics*. Any future feature that
  consumes a recorded metric must be checked by hand. Recommend a standing review rule.

### [HIGH] No experiment has been run; no research claim is supported
- **Type**: EXPERIMENTAL DESIGN ISSUE · **Category**: A, J · **Status**: open (blocked)
- **Problem**: H1–H6 in `docs/master_prompt.md` have zero supporting evidence.
  `results/` contains only `dummy`-tier wiring output, where every ROUGE-L is 0.0000 because the
  backbone is randomly initialised and the data is synthetic.
- **Evidence**: `VERIFIED` — `results/actual/` does not exist; `run_mode=actual` cannot resolve
  on this host (`QuantizationUnavailableError`, no CUDA).
- **Why it matters**: the deliverable is a research system, and the system is complete, but no
  scientific conclusion may be drawn from it yet.
- **Recommended fix**: resolve D-002/D-003/D-006, obtain a CUDA host, run `DQ1` first (cheapest
  real signal), then the main arms.

### [HIGH] The evaluated stream cannot falsify H2 or H3
- **Type**: EXPERIMENTAL DESIGN ISSUE · **Category**: G, J · **Status**: open (documented)
- **Location**: `configs/experiments/main_bootstrap.yaml` → `curriculum.evaluated_stream`
- **Problem**: under `bootstrap_six` the evaluated stream is the same six tasks that seeded the
  repository. With six maximally distinct task types and no overlap tasks, CREATE is the a-priori
  correct decision every time, so a REUSE or FUSE decision can never be *right*, and "does the
  learned utility choose better?" is unanswerable.
- **Evidence**: `VERIFIED` — the runner emits a loud `CONFIGURATION WARNING` and writes
  `configuration_warnings.json`; `DQ1_decision_quality.yaml` has `evaluated_stream: null` so it
  refuses to run rather than produce pseudo-evidence.
- **Why it matters**: this is the single largest methodological risk in the project. Without
  overlap tasks the headline result would only show that AMAUN performs comparably while making
  decisions whose correctness is undefined.
- **Recommended fix**: pin T7–T10 per `docs/experiment_protocol.md` §3 (a second summarization
  corpus, a second extractive QA set, a long-form domain task, a genuinely novel task), then run
  `DQ1` and report a decision confusion matrix.

### [HIGH] Arms have structurally unequal training budgets
- **Type**: EXPERIMENTAL DESIGN ISSUE · **Category**: I · **Status**: open
- **Problem**: FUSE performs post-merge optimisation, so fusing arms consume more optimiser steps
  than creating arms. Measured on the dummy stream: `B2_separate_lora` 12 steps,
  `B3_naive_average` 36, `B4_similarity_selection` 36, `main_bootstrap` 14.
- **Evidence**: `VERIFIED` from each arm's `budget_ledger.json`.
- **Why it matters**: any performance difference is then confounded with compute
  (falsification question 14). The `BudgetLedger` correctly *refuses* such comparisons, which
  means the matrix as configured cannot produce a fair headline table without explicit work.
- **Recommended fix**: equalise budgets deliberately — either cap total optimiser steps per task
  event across arms, or give creating arms the fusion arm's step count — and record the chosen
  equalisation in the protocol. Do not simply set `allow_unequal_budget`.

### [HIGH] Baseline B1 (sequential full fine-tuning) is not implemented
- **Type**: DOCUMENTATION ISSUE / missing capability · **Category**: I · **Status**: open, now honest
- **Problem**: brief §14 item 1 requires sequential full fine-tuning. `B1` has a config that
  declares `model.backbone_trainable: true`, but the trainer has no full-fine-tuning path.
- **Evidence**: `VERIFIED` — the trainer now raises `ConfigError` naming the gap, so B1 cannot
  silently run as a mislabelled LoRA arm. (This guard was added during the audit; previously the
  flag was ignored.)
- **Why it matters**: B1 is the plasticity upper bound and the worst-forgetting reference. Its
  absence weakens the forgetting claim.
- **Recommended fix**: implement a full-parameter trainer behind that flag, with its own budget
  accounting, or drop B1 from the comparison set explicitly in `docs/experiment_matrix.md`.

### [HIGH] Three named baselines cannot be implemented
- **Type**: RESEARCH ASSUMPTION · **Category**: I, M · **Status**: open (D-007)
- **Problem**: LoRM, FedALoRA, FedZKD are named in brief §14 but no references were supplied.
- **Evidence**: `VERIFIED` — each raises `BaselineRequiresCitationError` stating exactly what is
  needed; `unimplemented_report()` emits a machine-readable record into every run's summary.
- **Why it matters**: reconstructing a published method from its name and attaching the name to it
  would fabricate a baseline. Refusing is correct, but the comparison set is incomplete.
- **Recommended fix**: supply the three references, or state in the paper that comparison is
  against method *classes* rather than those specific works.

### [HIGH] Utility targets are proxies, and the proxy bias is weakly measured
- **Type**: RESEARCH ASSUMPTION · **Category**: C · **Status**: open (D-010, A-12)
- **Location**: `src/amaun/manager/bootstrap.py` `_loss_to_score`, `utility.supervision.*`
- **Problem**: the `primary_score` supervision head is trained on `exp(-val_loss/scale)`, not on
  a generated-text metric, and most pairs use a truncated fusion (`proxy_steps`). With
  `max_pairs_per_event: 8` and `proxy_validation_frac: 0.25`, only ~2 pairs per event are run to
  convergence — a thin basis for a bias estimate.
- **Evidence**: `INFERRED` from the code plus the bootstrap report's `proxy_bias` field.
- **Why it matters**: the utility network predicts a quantity that is not the reported metric, so
  `û.primary_score` is not on the same scale as any result. If the proxy's ordering diverges from
  the true metric's, selection degrades for reasons invisible in the logs.
- **Recommended fix**: for the reported runs, raise `proxy_validation_frac` enough to estimate
  bias per head with a stated confidence, and report the rank correlation between proxy score and
  true `primary_score` on the validated subset.

### [HIGH] Single seed in the shipped configuration
- **Type**: EXPERIMENTAL DESIGN ISSUE · **Category**: J · **Status**: open (D-012)
- **Problem**: `configs/experiments/base.yaml` ships `experiment.seeds: [1234]`.
- **Evidence**: `VERIFIED`. The precondition gate blocks a *reportable* run with fewer than three
  seeds, and `compare_arms` refuses below three — so this cannot produce a reported single-seed
  gain. But the default invites single-seed habits.
- **Recommended fix**: set `seeds: [1234, 2345, 3456]` as the shipped default and let smoke runs
  override downward.

### [MEDIUM] Feature-layout fragility was real and is now guarded
- **Type**: BUG · **Category**: B, C · **Status**: **RESOLVED**
- **Problem**: two families emitted a variable number of features depending on data availability
  (`aggregate` returned one key instead of two when values were missing; `semantic` omitted the
  cosine feature when an embedding was absent). The normaliser then rejected vectors mid-run.
- **Evidence**: `VERIFIED` — the failure surfaced at task 1 of the first full run
  (`53 features … got 54`).
- **Why it matters**: had the widths coincided, features would have been *silently misaligned*
  against `μ/σ` — every prediction corrupted with no error.
- **Fix applied**: `aggregate_keys` declares a data-independent key set; `semantic` pre-populates
  its layout; `tests/test_features_all_families.py` asserts one signature across seven
  availability scenarios.

### [MEDIUM] Compression pair utility was asymmetric
- **Type**: METHODOLOGICAL ERROR · **Category**: E · **Status**: **RESOLVED**
- **Problem**: `idea.md` §7.2 takes `argmax` over *unordered* pairs, but the feature vector is
  directional (historical/performance/resource features describe the candidate only), so
  `U(i,j) ≠ U(j,i)` and the chosen pair depended on repository iteration order.
- **Evidence**: `VERIFIED` by reading the context construction (`temp_delta=left.delta`).
- **Fix applied**: `pair_utility` now averages both directions.

### [MEDIUM] Experiment-affecting constant was hardcoded
- **Type**: BUG · **Category**: K · **Status**: **RESOLVED**
- **Problem**: `_loss_to_score` used `exp(-loss/10.0)` with `10.0` inline, violating brief
  development rule 7. Two further keys (`manager.reuse_prescreen_margin`,
  `utility.supervision.loss_to_score_scale`) were read with code defaults and appeared in no
  config file, so they would not have been recorded in `config_used.json`.
- **Evidence**: `VERIFIED` by cross-checking every `cfg.get("…")` against `configs/**`.
- **Fix applied**: both keys added to `configs/utility/default.yaml`; the constant is now read
  from config.

### [MEDIUM] The evaluator reads future-task TEST by default
- **Type**: RESEARCH ASSUMPTION · **Category**: F · **Status**: open, documented
- **Location**: `evaluation.evaluate_unseen_for_fwt: true`
- **Problem**: forward transfer needs `S[j-1][j]`, the score on task `j` before learning it, so
  the evaluator reads a not-yet-arrived task's TEST split.
- **Evidence**: `VERIFIED`. These scores are recorded for reporting only; `_prior_scores` reads
  the validation history, and future tasks are deliberately not appended to `task_order`.
- **Why it matters**: standard practice in continual-learning papers, but it widens TEST exposure.
  The safety now rests on one narrow property (nothing reads those cells back).
- **Recommended fix**: keep it, but state in the paper that FWT required pre-task test evaluation;
  or set the flag false and report FWT as not measured.

### [MEDIUM] Teacher logit cache is unbounded by default
- **Type**: PERFORMANCE ISSUE · **Category**: L · **Status**: open
- **Problem**: `capture_teacher` stores full float32 logits for every retention batch.
  At 7B (vocab ≈ 152k), 64 retention examples × 2048 positions × 152k × 4 bytes is far beyond
  device memory. `kd.top_p_truncation` defaults to `null`.
- **Evidence**: `INFERRED` from the tensor shapes; not reachable on the dummy tier.
- **Recommended fix**: set `top_p_truncation` (or a top-k) before any `actual` run, and record
  the truncation in the KD log — the field exists and is already reported.

### [MEDIUM] `local_smoke` and `actual` paths are untested
- **Type**: DOCUMENTATION ISSUE · **Category**: K · **Status**: open (environmental)
- **Problem**: every test runs on the `dummy` tier. The real-checkpoint loader, the chat template,
  and the quantisation path have never executed.
- **Evidence**: `VERIFIED` — no network access was used; `local_smoke` needs a ~3 GB download and
  `actual` needs CUDA.
- **Recommended fix**: run `scripts/run_smoke.py --run-mode local_smoke` as the first action on a
  machine with network access, before trusting the 7B path.

### [LOW] Acceptance criteria referenced non-existent test files
- **Type**: DOCUMENTATION ISSUE · **Category**: A · **Status**: **RESOLVED**
- **Problem**: `docs/acceptance_criteria.md` cited `tests/test_adapter_io.py`,
  `test_repository.py`, `test_candidate_isolation.py` and others that were consolidated into
  `tests/test_adapters.py`, so the stated verification commands would fail.
- **Fix applied**: every criterion now names a command that actually runs.

### [LOW] `d_F` and `d_2` are perfectly collinear
- **Type**: DOCUMENTATION ISSUE · **Category**: C · **Status**: accepted, documented
- **Problem**: the Frobenius norm *is* the L2 norm of the vectorisation, so `idea.md` §4.2's two
  features are the same quantity. Both are emitted for faithfulness.
- **Evidence**: `VERIFIED` — identical values in the dummy feature dump.
- **Assessment**: harmless for a regression target, but a reviewer could mistake it for a bug, so
  it is called out in `features/parameter.py`.

### [LOW] Softmax coefficients saturate at extreme logits
- **Type**: DOCUMENTATION ISSUE · **Category**: C · **Status**: accepted, documented
- **Problem**: `α` is described as bounded in `(0,1)`; in float32 it reaches exactly 1.0/0.0 for
  |logit| ≳ 40.
- **Evidence**: `VERIFIED` — a strict-interiority assertion failed at logit 9 with m≥2.
- **Assessment**: normalisation (the property fusion depends on) holds exactly; saturation means
  one component dominates a layer, a legitimate outcome. Documented in `fused_module.py`.

---

## What I could not check

1. **Any empirical claim.** No run exists, so every performance, forgetting, growth and
   communication result is `[TO BE FIXED EXPERIMENTALLY]`.
2. **Real-model behaviour.** The dummy backbone is random and the data synthetic; all task metrics
   are 0.0000. Metric *implementations* are unit-tested against hand-computed values, but their
   behaviour on real generations is unobserved.
3. **Novelty.** `docs/methodology.md` §9 explicitly declines to assert priority, and I did not
   perform a literature review. Three named baselines are `[CITATION UNVERIFIED]`. **Any novelty
   claim in a paper must be backed by a review that has not been done.**
4. **QLoRA correctness.** Requires CUDA.

## Required before any result is reported

1. Resolve D-002, D-003, D-005, D-006, D-011, D-012 (six OPEN decisions).
2. Pin overlap tasks T7–T10 so REUSE and FUSE can be *correct* (HIGH-2).
3. Equalise or explicitly account for per-arm training budgets (HIGH-3).
4. Implement B1 or drop it from the comparison set (HIGH-4).
5. Set `kd.top_p_truncation` before the 7B runs (MEDIUM).
6. Run `--run-mode local_smoke` end to end on a networked machine.
7. Raise `experiment.seeds` to ≥ 3 and re-verify the precondition gate.

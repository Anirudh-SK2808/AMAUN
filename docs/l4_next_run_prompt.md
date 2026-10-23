# Prompt: the full AMAUN decision study, one run

You are working in the AMAUN repository. The first L4 decision study
(`results/local_smoke/l4_decision_study/`, analysed in `docs/l4_decision_study_results.md`) ran
cleanly but could not test the method. Build and execute **one** complete run that fixes every
known flaw and tests the full method, including the learned utility network. There is no second
run and no manual checkpoint: every decision the run needs is made automatically, inside the run,
by the rules written here. Do not skip any item. If an item cannot be implemented, stop before the
GPU run and record why in `docs/decision_register.md`.

## Overview

**The run takes about 27–30 hours on the L4** with one seed. `time_budget_hours` is set to 32. The
run keeps run 1's execution settings (evaluation batch 4, gradient checkpointing on). It saves its
progress after each phase and can resume after a VM restart. Before it starts, preflight projects
the total time and stops if the projection exceeds 32 h.

| Phase | What happens | Time |
|---|---|---|
| 0 | Preflight: data integrity, GPU memory, time projection, datasets | 0.3 h |
| 1 | 6 seed adapters (load 3, train qa-B, instr, dialog); zero-shot and standalone scores | 2.3 h |
| 2 | Run-1 reference + 4 fusion settings on VALIDATION; pick the best one automatically | 4 h |
| 3 | All 30 adapter pairs with the chosen fusion setting | 4.5 h |
| 4 | Utility network (or fall back to `probe_rule`) and thresholds, then freeze everything | 0.1 h |
| 5 | 5 new tasks: train, eligibility check, every option on TEST, all policies, with and without an adapter limit | 11 h |
| 6 | A real sequence of 11 tasks arriving one after another (4 stream runs) | 4 h |
| 7 | Memory audit and repeatability check | 0.2 h |
| | **Total** | **~26.5 h (budget 32 h)** |

These times are estimates scaled from run 1's measured timings. The longest-output tasks (ELI5,
instr) are the most uncertain.

**How the run decides things on its own:**
- **Fusion setting:** the setting with the best new-task score among those that lose at most 1 % on
  old tasks.
- **Is fusion usable at all:** a recorded yes/no flag (`fusion_viable`); the run continues either way.
- **Utility network:** trained only if there are at least 20 pairs and 20–80 % are beneficial.
  Otherwise the run uses `probe_rule` and records why.
- **Evaluation tasks:** a task is left out of the policy comparison if its dedicated adapter does not
  beat the plain backbone by at least 5 % on VALIDATION. Its results are still reported.

**Defaults that may be changed before the run:**
- **Two new datasets:** XSum (a second summarisation task) and CoNLL-2003 NER (unlike any seed
  task). Check the dataset IDs on the VM before pinning them.
- **BERTScore for ELI5,** rescaled with its baseline. Its model must be cached on the VM.
- **Thresholds:** 2 % for new-task equivalence, 1 % for forgetting, 5 % for task eligibility.
- **Adapter limit:** 6 in the capped comparison.
- **Seed:** one (1234). It must stay 1234: the data splits of the reused adapters depend on it.

**Deliberately left out to fit ~30 h** (record each in `docs/decision_register.md` as future work):
more seeds; ablations A3 (fixed α) and A8 (global α); 300-step fusion; GSM8K; `probe_rule` and the
other baselines in the sequential stream (they are compared in Phase 5).

**Before the run can start:** about 30 code changes (Part A), each with an automatic dummy-tier
test. That is several days of building before any GPU time. These are methodology changes, so the
order is: `planner` writes the specs, `builder` implements them, `reviewer` checks them.

## Ground rules

1. **Specs first.** Record every methodology change in `docs/decision_register.md`,
   `docs/experiment_protocol.md`, `docs/experiment_matrix.md` and `specs/l4_full_study/`. Implement
   after that, then review against the specs.
2. **Tune on VALIDATION, test on TEST.** Fusion settings, thresholds, τ, tolerances and top-k are
   fixed before TEST is touched. TEST is scored once per configuration.
3. **Reuse what exists.** Load `seed-sum`, `seed-qa`, `seed-sentiment` from
   `adapters/local_smoke/l4_decision_study/*/delta.pt`. Do not retrain them. Their data splits must
   be reproduced exactly (item 23).
4. **Automatic tests only.** All new behaviour has dummy-tier tests that pass before the GPU run.
   The run's preflight must pass. Neither needs a human.
5. **One run, resumable.** The run must checkpoint after every phase and every trained adapter, and
   resume from the last checkpoint after a VM restart (`--resume`).
6. **Budget.** Set `time_budget_hours: 32`. Preflight projects the full run time and fails before
   starting if it exceeds the budget. No phase, pair or option may be skipped silently.
7. **Execution settings as run 1.** Evaluation batch size 4, gradient checkpointing on, no
   speed optimisations. The only exceptions are the three feasibility rules (item 26), which do not
   change any measured value.
8. **Comparability.** Keep all run-1 outputs and field names. Add fields; do not rename.
9. **Decoding.** Greedy decoding for every evaluation, with fixed per-task `max_new_tokens`
   (item 22) and the same prompt template for every option of a task.

## Facts from run 1 (do not re-derive)

- Fused adapters were scored after `fusion.storage: refactorize_svd` truncated them to rank 16
  (`src/amaun/fusion/refactorize.py`): mean per-layer relative error 0.36–0.48, max 0.58.
- KD at λ = 1 dominated: L_KD fell 9–37× in every fusion, while L_new rose in 9 of 12.
- `softmax_k` and `sigmoid_pair` (`src/amaun/fusion/fused_module.py`) both force α to sum to 1, so
  each adapter runs at about half strength.
- FUSE trains the full new adapter before fusing, so it saves no training. One adapter is 73.9 MB
  against a 3.09 GB backbone.
- `average_performance` is an unweighted mean of 4 scores (`src/amaun/continual/decision_study.py:1057`).
  The 0.01 tolerance on it (`:1085`) accepts ~0.04 of new-task loss as "equal".
- Longform (ELI5): CREATE ROUGE-L 0.1270 < zero-shot 0.1281; perplexity 16.41 → 12.97.
- Labels use the loss proxy `_loss_to_score` (`src/amaun/manager/bootstrap.py:132`). It mislabelled
  the one reuse win. Supervision generation checks used only 16 examples.
- Only 3 sample predictions are saved per evaluation (`decision_study.py:350`).
- All 6 supervision pairs were harmful, so τ came from the one-class fallback. The utility network
  (6 examples) fits none of its own targets. The memory/communication heads dominate its loss.
- Embedding similarity picked the wrong reuse candidate on longform. The probe-loss eligibility
  check picked the right one.
- Splits are assigned per example by `sha256(task_id|content_hash|seed) mod 10000`
  (`src/amaun/data/splits.py:97`), then capped from a 9 000-row pool per task (seed 1234).
- FinQA: `max_input_tokens` 768, 64 validation rows; its fusion step peaked at 21.85 GB of 24 GB.
- Three identical memory-audit fusion cycles gave different fused scores.
- BERTScore is already implemented in `src/amaun/evaluation/task_metrics.py` (used for `instr`).

---

## Part A: code changes (all before the run)

Each item names the flaw or finding it fixes: F1–F16 from the run-1 analysis, B1–B3 / G1–G12 from
the prompt review.

**Fusion**
1. **Lossless storage (F1).** `fusion.storage: keep_components` (and `dense_delta`) work end to end:
   fusion, evaluation, repository bytes, memory audit. Record true stored bytes. Always log the SVD
   error that `refactorize_svd` would have had.
2. **KD weight (F2).** `kd.lambda: 0` fully disables the KD term and its retention batches. Log
   L_new and L_KD per step for every fusion.
3. **Unconstrained α (F3).** Add parameterisation `free`: independent per-component coefficients,
   initialised to 1.0 each, not normalised. Make the training-free uniform average (B3) selectable
   from the decision study.
4. **In-run fusion selection.** Phase 2 runs the fusion grid (Part B) on VALIDATION and picks one
   config by this rule, written to `decision_study/fusion_selection.json` with the full table:
   - the highest mean relative new-task score (fused / standalone, measured metric) among configs
     whose mean per-old-task relative drop is ≤ 1 %;
   - if none meets the limit, the one with the lowest forgetting.

   Also record `fusion_viable` = true if the chosen config keeps fused new-task score ≥ 0.98 ×
   standalone with ≤ 1 % forgetting on at least 2 of 6 pairs. The run continues either way.

**Scoring and decisions**

5. **Cost (F4).** Record for every option: training seconds spent, training seconds saved vs
   CREATE, stored bytes. Support a repository cap (`decision_study.max_adapters`). Under the cap,
   CREATE is infeasible once the repository is full.
6. **Acceptability (F5).** Replace the averaged judgement (`decision_study.py:1057/1085/1090`): an
   option is **acceptable** if its new-task score ≥ 0.98 × CREATE's **and** every old task's
   relative drop ≤ 1 %. Decide on point estimates. Mark the comparison **inconclusive** when the
   95 % CI of (option − 0.98 × CREATE) contains 0 (G12). Keep `average_performance` as a secondary
   column.
7. **Oracle and regret (G1).** The oracle is the cheapest acceptable option (fewest adapters, then
   least training time); CREATE if none is acceptable. For every policy and task report:
   - **decision correct**: the chosen option is acceptable and costs no more than the oracle;
   - **quality shortfall**: max(0, 0.98 × CREATE score − chosen score);
   - **cost excess**: extra adapters and extra training seconds vs the oracle;
   - **forgetting**: the maximum relative old-task drop.

   Replace the old `regret_vs_oracle` in all flags with these. Keep the old value as a secondary
   field.
8. **BACKBONE option (F7).** Add option type BACKBONE (no adapter, zero bytes, zero training) to the
   option grid, the oracle and every policy's choice set. Its score is the task's zero-shot score.
9. **Measured-metric labels (F8, G5).** All "beneficial" and "actually OK" labels use the measured
   task metric on VALIDATION, computed on **150 examples** for short-output tasks (qa, qa-B,
   sentiment) and **100** for long-output tasks (sum, instr, dialog). Keep the loss proxy as a
   second field and report agreement.
10. **Per-example scores and CIs (F9).** Save every example's prediction and score for every
    evaluation. Paired bootstrap 95 % CIs (1 000 resamples over test examples) for each option vs
    CREATE and each policy vs `always_separate`.
11. **Determinism (F14).** Add `deterministic: true` (torch deterministic algorithms, fixed cuBLAS
    workspace, seeded dataloaders) and use it. Also repeat one fusion 3× in the memory audit and
    report the spread.

**Utility network (F16)**

12. **Measured targets (G2).** The output heads' targets use measured values: primary score =
    relative new-task metric (fused / standalone), forgetting = maximum relative old-task drop,
    memory and communication = stored bytes. z-score each head's target on the fit split.
13. **Cross-validation (G4).** Leave-one-new-task-out over the 6 seed tasks (6 folds). Report
    out-of-fold error per head and out-of-fold decision accuracy (does the highest U pick the best
    measured candidate?). Choose τ_u from the **out-of-fold** predictions, then refit on all pairs.
14. **Automatic fallback.** Fit the network only if there are ≥ 20 pairs and 20–80 % are beneficial.
    Otherwise the `amaun` policy uses `probe_rule` (item 16) and the run records why.
15. **How `amaun` decides (G3).** In this order:
    1. Reuse gate: if it fires for a candidate → REUSE it; else if the backbone's probe score is
       ≥ 0.98 × the best candidate's → BACKBONE.
    2. Otherwise, if max U ≥ τ_u → FUSE with that candidate.
    3. Otherwise → CREATE.

    `amaun_train_first` uses the same order, with the features computed after training the
    temporary adapter.

**Policies and cost control**

16. **`probe_rule` (baseline and fallback).** On 16 SUPPORT examples of the new task:
    - REUSE the best candidate if it lowers loss vs the backbone **and** its measured probe score is
      ≥ 0.98 × the backbone's;
    - else BACKBONE if the backbone's probe score is ≥ 0.98 × the best candidate's;
    - else CREATE.
17. **Candidate rules for the simple policies (G11).** REUSE-only: best candidate by probe score.
    FUSE-only: best candidate by probe loss gain. `always_merge`: nearest by `semantic.d_task_sq`
    (as in run 1). `fixed_rule`: as in run 1 (B4).
18. **Measured FUSE set (B3).** `decision_study.fuse_top_k: 2` by probe loss gain, **plus** every
    FUSE candidate any policy chooses. Measure all REUSE, BACKBONE and CREATE. Record which FUSE
    candidates were not measured and confirm no policy chose one.

**Tasks and data**

19. **New datasets (G6).** Add to `configs/datasets/curriculum.yaml`, with pinned revisions:
    - `xsum`: EdinburghNLP/xsum. Metric ROUGE-1/2/L (primary ROUGE-L); `max_input_tokens` 512,
      `max_target_tokens` 64.
    - `ner`: CoNLL-2003 (e.g. `eriktks/conll2003`). A row adapter turns token/tag rows into the
      sentence as input and a target of `TYPE: span` lines. Metric: entity-level F1 on exact
      (type, span) matches (primary); `max_input_tokens` 256, `max_target_tokens` 64.

    Verify the IDs and revisions on the VM before pinning.
20. **Duplicate tasks (F10, F11, B1).** Support `qa-B` and `sentiment-B` with task ids of their own,
    the same metrics and token limits as `qa` / `sentiment`, and the **same TEST set** as the
    original. Their TRAIN, VALIDATION and SUPPORT rows come only from upstream rows **outside** run 1's
    pools for `qa` / `sentiment`. (Use the loader's `skip` so the B pools start after the first
    9 000 rows run 1 consumed, and exclude any content hash present in run 1's pools.)
21. **Split sizes.** train 6 000 / validation 750 / support 750 / test 300 for qa-B, sentiment-B,
    instr, dialog and xsum; ner the same, with test from its upstream test split; FinQA train 5 700 /
    validation 250 / support 250 / test 300 (F13). sum, qa, sentiment and longform exactly as run 1.
22. **Generation limits (G7).** Per-task `max_new_tokens` = the task's `max_target_tokens` from the
    curriculum (sum 128, qa 48, sentiment 4, instr 256, dialog 128, longform 512, domain 128,
    xsum 64, ner 64), for every option of that task.
23. **Data integrity for reused adapters (B1).** The data config of sum, qa and sentiment stays
    byte-identical to run 1 (pool 9 000, shuffle buffer, seed 1234, ratios, caps). Preflight
    recomputes their splits and fails unless every split's snapshot key and row hashes match run 1's
    `decision_study/data.json`. The leakage audit runs on content hashes across **all** tasks,
    including qa ↔ qa-B and sentiment ↔ sentiment-B, and fails on any TRAIN row of one appearing in
    another task's VALIDATION, SUPPORT or TEST.
24. **ELI5 metric (F6, G8).** Report ROUGE-L, perplexity and BERTScore-F1 with
    `rescale_with_baseline=True`, using the existing implementation in `task_metrics.py` with a
    pinned model and version, cached on the VM. Primary metric: rescaled BERTScore-F1.
25. **Evaluation-task eligibility (F6, G10).** In Phase 5, after the temporary adapters are trained
    and before any TEST evaluation, check on VALIDATION that each evaluation task's dedicated adapter
    beats zero-shot by ≥ 5 % relative on its primary metric. Tasks that fail are marked
    `non_discriminating`: still measured and reported, but excluded from the policy tables.

**Feasibility rules (they change no measured value)**

26. Without these, the run cannot fit any budget. Each has a dummy-tier test showing it returns
    exactly what the naive computation returns.
    - **R1 (standalone scores once):** each adapter's standalone VALIDATION and TEST score on its
      own task is computed once and reused as the reference for every pair and option. It is an
      input, not an outcome.
    - **R2 (one adapter per task in the stream):** in Phase 6, a CREATE for task *t* uses the
      adapter already trained for *t* in Phase 1 or Phase 5. Same task, same TRAIN rows and same
      seed under deterministic training give the same adapter. Fusions are trained fresh in each
      stream run.
    - **R3 (re-evaluate only what changed):** in Phase 6, after each step only the new task and the
      tasks whose serving adapter changed (a fusion) are evaluated; an unchanged frozen adapter
      cannot change score under greedy decoding. Every task is evaluated once more on the final
      repository.

**Safety**

27. **Memory (F15).** Preflight includes a FinQA-length fusion step with `keep_components` and
    `free` α, and a fusion at the longest sequence of any task. Fail if peak allocated memory
    exceeds 22.5 GB.
28. **Budget and resume.** Preflight projects every phase's time from run 1's measured timings
    (scaled by the planned workload and the per-task generation limits) and from a short timing
    probe on the L4. It writes `decision_study/time_projection.json` and fails if the total exceeds
    `time_budget_hours`. Per-phase checkpoints and `--resume`.

**Dummy-tier tests (must all pass):** `keep_components` round-trips exactly; λ = 0 gives no KD term;
`free` α can sum to > 1; BACKBONE is in every choice set; acceptability, oracle, decision-correct,
shortfall and cost excess follow items 6–7; per-example scores are written; CIs compute; fusion
selection follows the rule; utility targets are measured values; leave-one-task-out folds have no
task overlap and τ comes from out-of-fold predictions; the network refuses one-class or < 20-pair
data and the fallback engages; `amaun` follows the order in item 15; every policy's FUSE choice is
measured; duplicate tasks share no rows with the originals' pools; the run-1 split check passes on
identical config and fails on a changed one; the NER row adapter and entity F1 are correct;
eligibility marks a non-discriminating task; R1–R3 return the same values as the naive computation;
preflight fails on an over-budget projection; resume restarts from a checkpoint.

---

## Part B: the run

Experiment id `l4_full_study`, tier `local_smoke` (report it as such), Qwen2.5-1.5B-Instruct bf16,
one NVIDIA L4, seed 1234, `time_budget_hours: 32`, `deterministic: true`.

### Phase 0: preflight (~0.3 h)
Data integrity against run 1 (item 23); leakage audit; memory checks (item 27); time projection
(item 28); dataset availability; BERTScore model cached.

### Phase 1: seed adapters (~2.3 h)

| Seed | Task / dataset | Action |
|---|---|---|
| seed-sum | sum (CNN/DailyMail) | load from disk |
| seed-qa | qa (SQuAD) | load from disk |
| seed-sentiment | sentiment (IMDb) | load from disk |
| seed-qa-B | qa-B (SQuAD, rows outside run 1's pool) | train |
| seed-instr | instr (Dolly-15k) | train |
| seed-dialog | dialog (DailyDialog) | train |

Training for every new adapter in the run (seeds and Phase 5 temporary adapters): same settings as
run 1 (LoRA r 16, α 32, dropout 0.05, all 7 projection types; lr 2e-4 cosine, 3 % warmup; batch 4 ×
gradient accumulation 4; 375 steps = 1 epoch of 6 000 rows, or all available rows; gradient
checkpointing on). Then zero-shot and standalone VALIDATION and TEST scores for all seed tasks (R1).

### Phase 2: fusion selection on VALIDATION (~4 h)

Same 6 ordered pairs as run 1 (sum / qa / sentiment). F-0 re-runs run 1's setting under this run's
evaluation code, so the comparison is like for like. Run 1's stored numbers are also reported.

| Config | storage | kd.lambda | α | steps |
|---|---|---|---|---|
| F-0 (reference) | refactorize_svd | 1.0 | softmax_k | 100 |
| F-a | keep_components | 0.1 | softmax_k | 100 |
| F-b | keep_components | 0 | softmax_k | 100 |
| F-c | keep_components | 0.1 | free (init 1.0 / 1.0) | 100 |
| F-e | — | — | uniform average, no training (B3) | 0 |

Per config and pair, record: new/old loss, new/old measured metric (item 9 sizes), L_new/L_KD
trend, α summary, bytes, peak memory. Select the config (item 4; F-0 is not eligible) and set
`fusion_viable`.

### Phase 3: supervision with the selected config (~4.5 h)

All 6 × 5 = 30 ordered pairs of the 6 seeds, measured-metric labels (item 9), features, reuse-gate
examples. Report the beneficial fraction overall and by pair type (same-task, related, unrelated).

### Phase 4: utility network and thresholds (~0.1 h)
Apply items 12–14: measured targets, leave-one-task-out CV, τ_u from out-of-fold predictions, or the
automatic fallback. Fit the reuse-gate ridge on VALIDATION data. Freeze everything.

### Phase 5: evaluation tasks (~11 h)

| Evaluation task | Purpose | Expected best answer |
|---|---|---|
| domain (FinQA) | needs its own adapter | CREATE |
| sentiment-B (IMDb, rows outside run 1's pool) | same-task duplicate; backbone is already strong | REUSE or BACKBONE |
| xsum | related to seed-sum | REUSE or FUSE |
| longform (ELI5), rescaled BERTScore | related to seed-sum; judgment call | open |
| ner (CoNLL-2003) | unlike every seed; weak zero-shot expected | CREATE |

In this order:
1. Train the temporary (dedicated) adapter for each task (Phase 1 training settings).
2. Eligibility check on VALIDATION (item 25).
3. Compute every policy's decision (no TEST data used).
4. On TEST, measure CREATE, BACKBONE, REUSE × 6 and the measured FUSE set (item 18) with the frozen
   fusion config, plus old-task scores for every FUSE option.
5. Score the policies: `amaun`, `amaun_train_first`, `probe_rule`, `fixed_rule` (B4),
   `always_separate` (B2 / A11), `always_merge` (A10), REUSE-only (A9), FUSE-only, and the oracle.
   Score them **without** a cap and **with** `max_adapters: 6` (the repository is full, so CREATE is
   infeasible), from the same measurements.

TEST size: 300 for qa, sentiment, domain, ner; 150 for sum, xsum, longform, instr, dialog.

### Phase 6: sequential stream (~4 h)

`main_sequential` with the frozen settings: start with an empty repository. Tasks arrive in a fixed
order (sum, qa, sentiment, instr, dialog, qa-B, sentiment-B, xsum, longform, domain, ner), and each
decision changes the repository. **4 stream runs:** `amaun` and `always_separate`, each with and
without `max_adapters: 6`. If `amaun` fell back to `probe_rule`, say so. Apply R2 and R3 (item 26).
Report the repository-size trajectory, each task's score when it arrives and on the final
repository, forgetting (arrival score vs final score), and decision counts.

### Phase 7: memory audit and repeatability (~0.2 h)
3 cycles with `keep_components` and the selected config. The same fusion runs in each cycle: report
memory growth and the spread of the fused score.

**Projected total ≈ 26.5 h.** Preflight must confirm ≤ 32 h before starting.

---

## Part C: reporting

Write `docs/l4_full_study_results.md`:

1. Run header: commit, config hash, hardware, duration, failures/resumes, tier `local_smoke`.
2. Projected vs actual time per phase.
3. Data integrity: run-1 split match, leakage audit, duplicate-pool provenance.
4. Seeds: training curves and TEST scores, zero-shot vs adapter.
5. Fusion selection: config × pair table with F-0 and run 1, both metrics, L_new/L_KD, α, bytes,
   memory, the chosen config, `fusion_viable`.
6. Supervision: beneficial fraction by pair type, proxy-vs-metric agreement.
7. Utility network: fitted or fallback (and why), out-of-fold error per head, out-of-fold decision
   accuracy, τ_u.
8. Per evaluation task: eligibility, and every option with new-task score ± CI, per-old-task change,
   cost, acceptable / inconclusive; oracle marked.
9. Policy tables (no cap / cap 6): decision, decision correct, quality shortfall, cost excess,
   forgetting, CI vs `always_separate`.
10. Sequential stream: trajectories, arrival vs final scores, decision counts.
11. Repeatability and memory audit.
12. A plain verdict on each question:
    - Does fusion work once fixed?
    - Does the learned utility network beat `probe_rule` and `fixed_rule`?
    - Does AMAUN beat `always_separate` when CREATE is not free?
13. Limitations: `local_smoke` tier, one seed (no seed-to-seed variance), model size, task count,
    and the items left out to fit the budget.

## Acceptance checklist (every box ticked, or the run is not complete)

- [ ] F1 `keep_components` used; would-be SVD error logged
- [ ] F2 λ ∈ {0, 0.1} run
- [ ] F3 `free` α and uniform average run
- [ ] F4 Cost fields recorded; cap-6 policy table reported
- [ ] F5 New-task and forgetting judged separately; inconclusive cases flagged
- [ ] F6 ELI5 on rescaled BERTScore; eligibility check applied to every evaluation task
- [ ] F7 BACKBONE in every choice set
- [ ] F8 Measured-metric labels on 100–150 examples; proxy agreement reported
- [ ] F9 Per-example scores; bootstrap CIs over examples (one seed, recorded as a limitation)
- [ ] F10 Dissimilar task (ner) and duplicate tasks (qa-B, sentiment-B) included
- [ ] F11 30 supervision pairs with same-task and related pairs; beneficial fraction reported
- [ ] F12 Sequential stream run (4 runs)
- [ ] F13 FinQA validation = 250
- [ ] F14 Deterministic mode on; repeatability spread reported
- [ ] F15 Preflight memory check passed with the new storage
- [ ] F16 Utility network: measured targets, leave-one-task-out CV, τ from out-of-fold, automatic fallback
- [ ] B1 Run-1 splits reproduced exactly; duplicates drawn outside run-1 pools; cross-task leakage audit clean
- [ ] B2 Run fits the budget; R1–R3 applied and tested
- [ ] B3 Every policy's FUSE choice measured
- [ ] G1 Decision correct / shortfall / cost excess reported
- [ ] G3 `amaun` decision order as specified
- [ ] G6–G7 New task specs and generation limits as specified; greedy decoding
- [ ] Fusion config chosen on VALIDATION only, by the written rule; F-0 reference reported
- [ ] `probe_rule` and `fixed_rule` both reported
- [ ] Left-out items recorded in the decision register
- [ ] Preflight's time projection ≤ 32 h; projected vs actual reported
- [ ] Every dummy-tier test listed in Part A passes

# AMAUN — Experiment Protocol

No numbers in this document are results. Every `[TO BE FIXED EXPERIMENTALLY]` marker is a
value that only a real run may fill.

---

## 1. Preconditions for any reportable run

A run may write to `results/actual/` only if **all** hold:

1. `pytest -q` is fully green on the `dummy` tier.
2. `scripts/run_smoke.py --run-mode local_smoke` completed end to end, exercising every
   module at least once (`idea.md` §0.2.1). Its prerequisites, in order:
   `scripts/fetch_backbone.py --run-mode local_smoke` (pinned local snapshot, verified to load
   offline) then `scripts/prepare_data.py --run-mode local_smoke` (real data, four-way splits,
   clean leakage audit). See `specs/local-smoke-data-bootstrap/`.
3. The git tree is clean, or `experiment.allow_dirty_git: true` is set explicitly.
4. Every `OPEN` decision in `docs/decision_register.md` that the run depends on is
   `RESOLVED`.
5. `experiment.seeds` lists ≥ 3 seeds (D-012).

The runner checks 1–5 and aborts otherwise. This is the mechanical form of
brief §28 ("never run expensive full training until smoke tests and methodology checks pass").

## 2. Splits

Per task, four disjoint splits derived by seeded content hashing (M2):

| Split | Fraction (config) | Who may read it |
|---|---|---|
| TRAIN | `data.split_ratios.train` | adapter training, utility supervision, KD retention |
| VALIDATION | `data.split_ratios.validation` | early stopping, hyperparameter tuning, τ selection |
| SUPPORT | `data.split_ratios.support` | **the manager, exclusively** |
| TEST | `data.split_ratios.test` | **the evaluator, exclusively, once per arm** |

SUPPORT is carved from the *training pool*, never from the upstream test split, so that
"the manager saw support data" can never mean "the manager saw test data".

Upstream dataset splits map as: upstream-train → {TRAIN, VALIDATION, SUPPORT};
upstream-validation or a held-out slice of upstream-train → TEST (never the upstream
*test* split for datasets whose test labels are hidden, e.g. SQuAD — documented per task in
`configs/datasets/`).

**Single-split corpora.** Where the upstream dataset has only `train`
(`databricks/databricks-dolly-15k`), TEST is carved from a **disjoint row window** of that same
split via `data.volumes.<task>.test_offset`, and `test_offset ≥ rows_consumed_by_the_training_pool`
is enforced against the measured row count (invariant `I2.6`). Without the offset both pools are
the same prefix of the stream, and the resulting TEST ⊆ TRAIN is caught by `assert_disjoint`,
which aborts the run.

**Sampling under a cap.** Every load is capped, so *which* rows the cap selects is an
experimental variable. `data.sampling` has no code default and must be declared per run; it is
**`shuffled_buffer`** for every tier (D-016, resolved 2026-09-22), with
`data.shuffle_buffer_size ≥ max(train_pool, test)` — a narrower window makes the shuffle a
no-op and would misreport the policy. The shuffle seed derives from `data.seed` and is identical
across both loads of a task, which is what keeps the offset-based carving above valid.

## 3. Task curriculum

Base curriculum (`idea.md` §0.2):

| Adapter | Task | Dataset | `actual` volume |
|---|---|---|---|
| LoRA-1 | Summarization | `abisee/cnn_dailymail` @ sha, config `3.0.0` | 30k–50k train |
| LoRA-2 | Question answering | `rajpurkar/squad` @ sha | 30k–50k train |
| LoRA-3 | Instruction following | `databricks/databricks-dolly-15k` @ sha (D-006 B1) | **capped by the corpus: 15 011 rows total** |
| LoRA-4 | Dialogue generation | `li2017dailydialog/daily_dialog` @ `refs/convert/parquet` | ~8k–10k train |
| LoRA-5 | Long-form generation | `Pavithree/eli5` @ sha (D-006 B2) | 20k–30k train |
| LoRA-6 | Domain-specific generation | `dreamerdeo/finqa` @ sha | 10k–20k train |

Row→example rules are fixed by D-015 (resolved 2026-09-22): `dialog` → `dialog_last_turn`
(one example per dialogue), `domain` → `finqa_pre_table_post` with `table_format: markdown`,
`longform` target → `answers.text[argmax(score)]`. All six tasks therefore load.

`actual`-tier volumes remain unresolved (`data.volumes: null` at base level): `idea.md` §0.2
gives ranges, and LoRA-3's range is no longer achievable — dolly-15k has 15 011 rows in total,
so the `actual` volume for `instr` must be set deliberately and its inequality with the other
tasks disclosed. This is now the principal data-side gate on a reportable run.

**Required extension (D-002).** Under `repository_init: bootstrap_six`, the evaluated task
stream consists of *new* tasks `T₇…`, and at least some of them must plausibly overlap an
existing adapter, or REUSE/FUSE can never be the correct decision and H2/H3 are untestable.
Proposed overlap tasks (all `OPEN`, pending D-006-style pinning):

| New task | Overlaps | Purpose |
|---|---|---|
| T7 · second summarization corpus (e.g. XSum) | LoRA-1 | REUSE should be correct |
| T8 · second extractive QA corpus | LoRA-2 | REUSE should be correct |
| T9 · long-form domain QA | LoRA-5 + LoRA-6 | FUSE should be correct |
| T10 · a genuinely novel task type | none | CREATE should be correct |

This gives a *decision ground truth* against which decision accuracy — not only downstream
score — can be reported. Without it, the decision module is evaluated only indirectly.

Exact volume used per run is recorded in `config_used.json` (`idea.md` §0.2 note 3).
Task order is a config field; randomised orders are supported and must be reported with
their seed.

## 4. Procedure per arm

```
for seed in experiment.seeds:
  fix seeds (python, numpy, torch, cuda, dataloader workers)
  load frozen backbone; record fingerprint
  measure zero-shot baselines b_j on every task (FWT reference), cache
  if repository_init == bootstrap_six:
      train LoRA-1..6 independently from θ₀ (fresh injection + fresh optimiser + derived seed)
      register all six unconditionally
  for each arriving task t in the evaluated stream:
      build task representation from SUPPORT
      enumerate candidates; extract 7 feature families; normalise with stored μ/σ
      predict utilities; decide REUSE | FUSE | CREATE
      execute the decision (fusion+KD if FUSE; train on TRAIN if CREATE)
      update repository; compress if Mem(R_c) ≥ R_max
      federated round if t mod federated.round_every == 0
      evaluate on TEST for all tasks seen so far → fill row S[t][·]
  compute AvgPerf, Forgetting, BWT, FWT, efficiency metrics
  verify backbone fingerprint unchanged
```

## 5. Utility-network bootstrap (must precede the evaluated stream)

1. Run the pipeline on a **decision-development task stream** disjoint from the evaluated
   stream (D-005 option a).
2. For each candidate pair, actually perform the integration and record
   `y_i = [s_post, F_post, C_post, Mem_post, Comm_post, L_post]` (`idea.md` §9), under the
   sampling/proxy policy of D-010.
3. Fit `FeatureNormalizer` on **this** split only; persist `μ, σ` + feature order + run mode.
4. Train `g_ω` with `L_U = Σ‖g_ω(z)−y‖²`; select `τ_u`/`τ_reuse` on the dev stream.
5. Freeze `g_ω`, `μ/σ`, and thresholds. They are inputs to the evaluated stream, never
   refitted during it.

## 6. Federated protocol

Axes are independent (D-011): `federated.num_clients`, `federated.partition`,
`federated.round_every`, `federated.participation_rate`, `federated.local_epochs`,
`federated.compatibility.key`, `federated.aggregation.method`.
`one client = one task` is never assumed.

Per round, log: participating clients, compatibility groups, `n_c` weights, measured
payload bytes per client and total, aggregation method, reconstruction error if
re-factorising, and post-aggregation scores.

## 7. Evaluation

Decoding config identical across all arms and logged (`models/generation.py`).
Per task: the task's own metrics plus its `primary_score` (D-009).
Continual: AvgPerf, Forgetting, BWT, FWT from the persisted `S` matrix.
Efficiency: adapter count, tasks-served-per-adapter, trainable params, stored params,
storage bytes, communication bytes, training wall time, inference overhead.

**Reporting rules.** Mean ± std over `experiment.seeds`; paired bootstrap CI over per-task
scores for any claimed difference; effect size alongside any p-value; adapter count never
reported without tasks-served-per-adapter and per-task performance (D-014); any
`local_smoke`/`dummy` figure shown anywhere is labelled `[SMOKE TEST — not a reported result]`.

## 8. Hyperparameter tuning surface

Tuned on VALIDATION only, never on TEST: `r`, LoRA `α`, dropout, `η`, `E`, utility-network
width/depth/regularisation, `τ_u`, `η_α`, `T`, `λ`, `R_max`.
Tuning trials are counted in the `BudgetLedger` and must be comparable across arms
(I13.1) — giving AMAUN 50 trials and a baseline 1 is an unfair comparison even if every
other budget matches.

## 9. Compute plan

| Phase | Tier | Where | Notes |
|---|---|---|---|
| unit/invariant tests | `dummy` | dev box, CPU | seconds |
| pipeline smoke | `local_smoke` | dev box, MPS, bf16, no quantisation | minutes–hours |
| all reported runs | `actual` | **CUDA host required** | QLoRA needs bitsandbytes (D-008) |

Estimated `actual` cost is `[TO BE FIXED EXPERIMENTALLY]`; the runner logs measured wall
time and tokens so the estimate is replaced by measurement after the first arm.

## 10. l4_full_study (one-run decision study, `local_smoke`)

Specification: `docs/l4_next_run_prompt.md`; requirements and design: `specs/l4_full_study/`;
decisions: D-017 – D-024. Config: `configs/experiments/l4_full_study.yaml`; entry point:
`scripts/run_full_study.py [--resume]`. NON-REPORTABLE (`local_smoke`, one seed).

- **Split discipline.** Phases 1–4 and every policy decision read TRAIN / VALIDATION / SUPPORT
  only. Fusion config, τ_u, the reuse gate, the fixed-rule threshold, tolerances and top-k are
  frozen at the end of Phase 4, before any evaluation task's TEST split is read. Each evaluation
  task's eligibility check runs on VALIDATION, and its decisions are logged before its first TEST
  read. Each configuration is scored on TEST once (results are cached, never re-drawn).
- **Run-1 reuse.** seed-sum / seed-qa / seed-sentiment are loaded, not retrained; preflight
  proves their splits are reproduced (D-022) and the content-hash leakage audit covers every task,
  including the duplicates.
- **Judgement.** Acceptability, oracle and decision quality per D-018; labels per D-019; the
  `amaun` order per D-020.
- **Statistics.** One seed: CIs are paired bootstraps over TEST examples (1 000 resamples), which
  measure example sampling noise only, not seed variance (D-024).
- **Resume.** A restart resumes from the last completed unit; resumes are recorded in the
  summary and the results document.

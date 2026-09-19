# AMAUN — Experiment Protocol

No numbers in this document are results. Every `[TO BE FIXED EXPERIMENTALLY]` marker is a
value that only a real run may fill.

---

## 1. Preconditions for any reportable run

A run may write to `results/actual/` only if **all** hold:

1. `pytest -q` is fully green on the `dummy` tier.
2. `scripts/run_smoke.py --run-mode local_smoke` completed end to end, exercising every
   module at least once (`idea.md` §0.2.1).
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

## 3. Task curriculum

Base curriculum (`idea.md` §0.2):

| Adapter | Task | Dataset | `actual` volume |
|---|---|---|---|
| LoRA-1 | Summarization | CNN/DailyMail 3.0.0 | 30k–50k train |
| LoRA-2 | Question answering | SQuAD v1.1 | 30k–50k train |
| LoRA-3 | Instruction following | **unpinned (D-006 B1)** | 30k–50k train |
| LoRA-4 | Dialogue generation | DailyDialog | ~8k–10k train |
| LoRA-5 | Long-form generation | **unpinned (D-006 B2)** | 20k–30k train |
| LoRA-6 | Domain-specific generation | FinQA | 10k–20k train |

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

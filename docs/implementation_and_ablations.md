# AMAUN — Implementation and Ablation Study

How the code actually works, and what each ablation changes. Facts here come from reading
`src/`, `configs/` and the test suite. No result values appear: no experiment has been run on
real data, and every number-shaped item is `[TO BE FIXED EXPERIMENTALLY]`.

---

## 1. What the system does

A frozen language model (`θ₀`, Qwen2.5) serves a stream of NLP tasks. A repository `R_c` holds
LoRA adapters. When a task arrives, a learned **utility network** decides whether to:

- **REUSE** an existing adapter (train nothing new, repository unchanged),
- **FUSE** the best candidates with a new temporary adapter (layer-wise learnable fusion + KD),
- **CREATE** a new adapter.

The aim is to bound adapter growth while limiting forgetting. The backbone never changes and is
never transmitted.

## 2. End-to-end flow

Driver: `ContinualRunner` in `src/amaun/continual/runner.py`.

```
bootstrap_repository     train 6 seed adapters independently from θ₀, register all
bootstrap_utility        on 2 dev tasks: fuse sampled pairs, MEASURE outcomes, fit
                         normaliser + utility network, select thresholds, then FREEZE
for each arriving task:
  1. task representation        SUPPORT split only
  2. train temporary adapter    TRAIN split; held in a candidate store, not the repository
  3. features                   7 families per (A_temp, A_i) pair; normalise with frozen μ/σ
  4. utility + decision         REUSE | FUSE | CREATE
  5. execute decision           see below
  6. compression                if Mem(R_c) >= R_max (utility-based)
  7. evaluate on TEST           fills one row of the performance matrix S[t][*]
  8. validation scores          VALIDATION only, feeds the next task's continual features
  9. assert backbone fingerprint unchanged
metrics: AvgPerf, Forgetting, BWT, FWT from S
```

Executing the decision:

| Outcome | Effect |
|---|---|
| REUSE | existing adapter now also serves the task; `A_temp` discarded |
| FUSE | `ΔW_merge = Σ_k α_k ΔW_k`, `α = softmax(a)` init uniform (0.5 for m=2); loss `L_new + λ·L_KD`; teacher = pre-merge adapter; `α` and LoRA params optimised jointly; result replaces the selected adapter |
| CREATE | `A_temp` registered as a new adapter |

Note: a temporary adapter is trained on **every** arrival, including ones that end in REUSE.
The `prescreen_then_train` option exists in config but the runner always trains first, so REUSE
saves storage, not training time.

## 3. Modules

| Area | Path | Notes |
|---|---|---|
| Config, run modes | `src/amaun/config.py` | `run_mode` required, no default; `dummy`, `local_smoke`, `actual` |
| Frozen backbone | `src/amaun/models/backbone.py` | fingerprint before/after every phase |
| Data and splits | `src/amaun/data/` | TRAIN / VALIDATION / SUPPORT / TEST by content hash; typed access guards |
| Adapters, repository | `src/amaun/adapters/` | `AdapterDelta` (`ΔW = scale·B@A`) is the only currency; registration needs a `Decision` |
| LoRA training | `src/amaun/training/lora_trainer.py` | response-only loss, one loop for every arm |
| Features (7 families) | `src/amaun/features/` | semantic, parameter, gradient, performance, continual, historical, resource |
| Utility network, decision | `src/amaun/manager/` | multi-head MLP, monotone scalarisation, thresholds selected on dev stream |
| Fusion + KD | `src/amaun/fusion/`, `src/amaun/training/kd.py` | `FusedLoRALinear` never materialises dense `ΔW` |
| Compression | `src/amaun/manager/compression.py` | same utility mechanism as selection |
| Federated | `src/amaun/federated/` | compatibility-gated delta-level FedAvg |
| Baselines | `src/amaun/baselines/` | B2–B6 implemented; B1 refuses (no full fine-tuning path); LoRM, FedALoRA, FedZKD need citations |
| Evaluation | `src/amaun/evaluation/` | evaluator is the only TEST reader; shared metrics across arms |

Scripts live in `scripts/`; the run order is in `docs/experiments.md`.

## 4. Invariants the code enforces

- Backbone parameters never get gradients; a fingerprint check runs after each event.
- Manager reads SUPPORT only, trainer TRAIN/VALIDATION, evaluator TEST only.
- Continual features use a validation-only history, never the TEST-filled matrix (an earlier
  leak of this kind was found by audit and fixed; a test poisons the TEST matrix to guard it).
- LoRA factors are never averaged separately; fusion is at the `ΔW` level.
- Normalisation statistics come only from the utility-training split.
- `BudgetLedger` refuses to compare arms whose optimiser steps, tokens or tuning trials differ
  beyond tolerance.
- Reportable runs need at least 3 seeds.

## 5. Ablation study

All ablations are config-only toggles in `configs/experiments/`. Each includes `base.yaml` and
`bootstrap_six`. `tests/test_ablation_toggles.py` asserts on the dummy tier that each toggle
changes an observable, so none can silently be a no-op.

### 5.1 Table

| ID | Toggle (config) | Question | Observable that must change | Hypothesis |
|---|---|---|---|---|
| A1 | only `features.families.semantic` on | Does heterogeneous evidence help? | feature dimension | H1 |
| A2 | `manager.similarity` fixed rule on `semantic.d_task_sq` instead of the network | Does learned utility beat a fixed rule? | decision distribution | H1 |
| A3 | `fusion.coefficient.learning_rate: 0.0` | Does learnable `α` help? | `α` stays at init | H2 |
| A4 | `kd.enabled: false`, `kd.lambda: 0.0` | What does KD contribute? | loss terms present | H4 |
| A5 | `repository.max_bytes: null` | Cost of unmanaged growth | repository trajectory | H5 |
| A6 | `features.families.historical: false` | Does adapter history help? | feature dimension | none |
| A7 | `features.families.resource: false` | Does cost information change selection? | feature dimension | none |
| A8 | `fusion.coefficient.granularity: global` | Is per-layer flexibility needed? | `α` tensor shape | H2 |
| A9 | `manager.allowed_outcomes: [REUSE]` | Is fusion or creation necessary? | decision distribution | H3 |
| A10 | `manager.allowed_outcomes: [FUSE]` | Is creation necessary? | decision distribution | H3 |
| A11 | CREATE only | No-management control | adapter count = task count | H3 |
| A12 | `task_repr.encoder: token_stats` | Does semantic task representation matter? | feature dimension | none |

A1–A8 come from `idea.md` §12.5; A9–A12 were added from the engineering brief.

### 5.2 How to read the results

- Compare each ablation against the full AMAUN arm (`main_bootstrap`) under the same seeds,
  with paired bootstrap CIs and effect sizes (`src/amaun/evaluation/stats.py`).
- A hypothesis is refuted only if its named ablation matches the full model within seed noise
  (see `docs/master_prompt.md` §2).
- Adapter count must always be reported together with tasks-served-per-adapter and per-task
  score (D-014).

### 5.3 Caveats before you run them

1. **No FUSE or REUSE without overlap tasks.** The current evaluated stream is the same six
   tasks that seeded the repository. CREATE is always the correct decision there, so A2, A3, A8,
   A9 and A10 will not separate from the full model. Add the overlap tasks (T7–T10,
   `docs/experiment_protocol.md` §3) and run `DQ1` first.
2. **A5 vs the base config.** `utility/default.yaml` already sets `repository.max_bytes: null`.
   For A5 to differ from the full model, the main arm must set a finite `max_bytes` so
   compression actually triggers.
3. **Equal budgets.** FUSE spends extra optimiser steps, so arms use unequal budgets by
   construction. The ledger will refuse comparisons until you equalise them (do not just set
   `allow_unequal_budget`).
4. **Seeds.** The shipped config uses one seed. Use three or more for anything reported.
5. **Proxy targets.** Utility supervision uses `exp(-val_loss/scale)`, not a generated-text
   metric, so proxy bias should be measured for reported runs (D-010).

### 5.4 Suggested order

```
dummy tests -> local_smoke full pipeline -> DQ1 -> main_bootstrap, main_sequential
-> B2..B6 -> A1..A12 -> F1..F5
```

Cheapest-to-run first within the ablations: A9, A10, A11 (no utility network needed), then A3,
A4, A5, A8 (can share one utility bootstrap), then A1, A6, A7, A12 (each needs its own because
the feature set changes), then A2.

## 6. Status

Implementation is complete and audited (`docs/review_report_2026-09-19.md`). Real-data
evidence so far: two `local_smoke` bootstrap trainings only. No hypothesis H1–H6 is supported
or refuted yet. Open decisions are tracked in `docs/decision_register.md`.

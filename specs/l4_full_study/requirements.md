# l4_full_study — Requirements

Source: `docs/l4_next_run_prompt.md` (the full AMAUN decision study, one run). This file restates
every Part A item as a testable requirement, names the code that satisfies it and the dummy-tier
test that proves it. Decisions the prompt leaves open are recorded in `docs/decision_register.md`
D-017 – D-024 and referenced here; nothing below is an agent choice that is not also recorded there.

Experiment id `l4_full_study`, tier `local_smoke` (NON-REPORTABLE), Qwen2.5-1.5B-Instruct bf16,
one L4, seed 1234, `time_budget_hours: 32`, `deterministic: true`.

## Fusion

| ID | Requirement | Prompt item |
|---|---|---|
| FR-1.1 | `fusion.storage: keep_components` and `dense_delta` produce an exact fused adapter that can be installed for evaluation, loss measurement, probes and fusion at any rank (rank 2r for pairwise fusion). | 1 |
| FR-1.2 | True stored bytes are recorded: factors for `keep_components`, `d_out·d_in` per layer for `dense_delta`, rank-r factors for `refactorize_svd`. | 1 |
| FR-1.3 | Every fusion log records the per-layer relative error `refactorize_svd` *would* have had at rank r, whatever storage is used. | 1 |
| FR-1.4 | Installing any delta reproduces `scaling·B·A` exactly, independent of the PEFT layer's own `alpha/r` (D-017). | finding |
| FR-2.1 | `kd.lambda: 0` captures no teacher, builds no retention batches and adds no KD term; `L_KD` is logged as 0. | 2 |
| FR-2.2 | Every fusion logs `L_new` and `L_KD` per step and a trend summary (first/last 10 % means). | 2 |
| FR-3.1 | Parameterisation `free`: one independent coefficient per component (per layer, or global), initialised to `fusion.coefficient.init_value` (1.0), never normalised. | 3 |
| FR-3.2 | The training-free uniform average (B3) is a fusion config with `steps: 0` and a normalised parameterisation; no optimiser step and no teacher capture run. | 3 |
| FR-4.1 | Phase 2 selects one config from the grid by: highest mean relative new-task score (fused / standalone, measured metric) among configs whose mean per-old-task relative drop ≤ 1 %; else the lowest forgetting. The reference F-0 is never eligible. | 4 |
| FR-4.2 | `fusion_viable` = the chosen config has fused ≥ 0.98 × standalone **and** ≤ 1 % forgetting on ≥ 2 of 6 pairs. Written with the full table to `decision_study/fusion_selection.json`. | 4 |

## Scoring and decisions

| ID | Requirement | Prompt item |
|---|---|---|
| FR-5.1 | Every option records `training_seconds_spent`, `training_seconds_saved_vs_create`, `stored_bytes`, `adapters_after`. | 5 |
| FR-5.2 | `decision_study.max_adapters` caps the repository; at the cap CREATE is infeasible and a CREATE decision is replaced by BACKBONE (D-018). Capped and uncapped tables come from the same measurements. | 5 |
| FR-6.1 | An option is **acceptable** iff new ≥ 0.98 × CREATE and every old task's relative drop ≤ 1 % (point estimates). | 6 |
| FR-6.2 | A comparison is **inconclusive** when the paired 95 % bootstrap CI of (option − 0.98 × CREATE) over TEST examples contains 0. `average_performance` stays as a secondary column. | 6 |
| FR-7.1 | Oracle = cheapest acceptable option (fewest adapters after, then least training seconds, then higher new-task score); CREATE if none (cap: D-018). | 7 |
| FR-7.2 | Per policy and task: `decision_correct`, `quality_shortfall`, `cost_excess` (adapters, seconds), `forgetting` (max relative drop). Flags use these; run-1 `regret_vs_oracle` / `oracle_option` remain as secondary fields. | 7 |
| FR-8.1 | Option BACKBONE (no adapter, 0 bytes, 0 training, score = zero-shot TEST score) is in the option grid, the oracle and every policy's choice set. | 8 |
| FR-9.1 | Every "beneficial" / "actually OK" label uses the measured task metric on VALIDATION with `decision_study.label_examples` per task (150 short-output, 100 long-output). The loss-proxy label is kept as `beneficial_loss_proxy`; agreement is reported. | 9 |
| FR-10.1 | Every evaluation writes every example's prediction, reference and per-metric score (`decision_study/predictions/*.jsonl`). | 10 |
| FR-10.2 | Paired bootstrap 95 % CIs (1 000 resamples over examples, fixed seed) for each option vs CREATE and each policy vs `always_separate`. | 10 |
| FR-11.1 | `deterministic: true` sets `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms(True)`, cuDNN deterministic, seeded loaders, before the backbone loads. | 11 |
| FR-11.2 | The memory audit repeats one identical fusion 3× and reports the spread of the fused score. | 11 |

## Utility network

| ID | Requirement | Prompt item |
|---|---|---|
| FR-12.1 | Head targets are measured: `primary_score` = fused / standalone new-task metric, `forgetting` = max relative old drop, `memory` = `communication` = stored MB, `confidence` = loss-proxy score, `long_term_loss` = post-fusion loss. | 12 |
| FR-12.2 | Each head's target is z-scored on the fit split; the statistics are stored with the network. | 12 |
| FR-13.1 | Leave-one-new-task-out CV over the 6 seed tasks: out-of-fold error per head, out-of-fold decision accuracy; folds share no new task. | 13 |
| FR-13.2 | τ_u is chosen from out-of-fold utilities; the network is then refitted on all pairs. | 13 |
| FR-14.1 | The network is fitted only with ≥ 20 pairs and a beneficial fraction in [0.2, 0.8]; otherwise `amaun` (and `amaun_train_first`) use `probe_rule` and the reason is recorded. | 14 |
| FR-15.1 | `amaun` decides: (1) gate fires → REUSE; else backbone probe ≥ 0.98 × best candidate probe → BACKBONE; (2) max U ≥ τ_u → FUSE best-U; (3) CREATE. `amaun_train_first`: same order, REUSE from the post-training evidence (D-020). | 15 |

## Policies

| ID | Requirement | Prompt item |
|---|---|---|
| FR-16.1 | `probe_rule` on 16 SUPPORT examples as specified (REUSE / BACKBONE / CREATE). | 16 |
| FR-17.1 | REUSE-only: best by probe score; FUSE-only: best by probe loss gain; `always_merge`: nearest by `semantic.d_task_sq`; `fixed_rule`: run-1 B4. | 17 |
| FR-18.1 | Measured FUSE set = top-2 by probe loss gain ∪ every FUSE any policy chose (capped or not). Unmeasured FUSE candidates are listed and the run asserts no policy chose one. | 18 |

## Tasks and data

| ID | Requirement | Prompt item |
|---|---|---|
| FR-19.1 | `xsum` (EdinburghNLP/xsum @ `7d4d486c…`): ROUGE-1/2/L, primary ROUGE-L, 512 / 64 tokens. | 19 |
| FR-19.2 | `ner` (eriktks/conll2003, `refs/convert/parquet` @ `ce85b39f…`): row adapter `conll_entities` (sentence → `TYPE: span` lines, `none` when empty); primary `entity_f1` (per-sentence exact (type, span) F1, mean over sentences; D-021), plus corpus micro F1; 256 / 64 tokens. | 19 |
| FR-20.1 | `qa-B`, `sentiment-B`: own task ids, metrics and limits of `qa` / `sentiment`, identical TEST rows, training pool from the same shuffled stream after `train_offset: 9000`, minus any content hash in the original's pool. | 20 |
| FR-21.1 | Split sizes per item 21; FinQA by exact counts 5 656 / 250 / 250 (`data.task_split_counts`; 5 700 in the prompt, cut to the 6 156 distinct rows, D-021). | 21 |
| FR-22.1 | Per-task `max_new_tokens` = the task's `max_target_tokens` (generation cap raised to 512). Greedy decoding. | 22 |
| FR-23.1 | Preflight verifies sum, qa, sentiment against run 1: identical data config, identical snapshot keys, identical split sizes, TRAIN token count = run-1 `tokens_seen`; row-hash digests when the reference has them (D-022). | 23 |
| FR-23.2 | A content-hash leakage audit across all tasks fails on any TRAIN row of one task in another task's VALIDATION / SUPPORT / TEST; TEST ↔ TEST between declared duplicates is allowed. | 23 |
| FR-24.1 | BERTScore-F1 (roberta-large @ `722cf37b…`, layer 17, no idf, rescaled with the bert_score 0.3.13 baseline). ELI5 primary = rescaled BERTScore-F1. | 24 |
| FR-25.1 | After temp training and before any TEST read, each evaluation task's dedicated adapter must beat zero-shot by ≥ 5 % relative on VALIDATION; otherwise `non_discriminating` (measured, reported, excluded from policy tables). | 25 |

## Feasibility, safety, budget

| ID | Requirement | Prompt item |
|---|---|---|
| FR-26.1 (R1) | Standalone VALIDATION / TEST scores are computed once and reused; equal to recomputation. | 26 |
| FR-26.2 (R2) | A stream CREATE reuses the adapter trained for that task in phase 1 / 5 (same adapter id, so same seed). | 26 |
| FR-26.3 (R3) | Stream steps re-evaluate only the new task and tasks whose serving adapter changed; a final evaluation covers all tasks; results equal the naive evaluation. | 26 |
| FR-27.1 | Preflight memory: FinQA-length fusion with `keep_components` + `free`, and a fusion at the longest sequence of every task; fail above 22.5 GB peak allocated (CUDA). | 27 |
| FR-28.1 | Preflight writes `decision_study/time_projection.json` (per phase, from run-1 timings scaled by workload and generation limits, refined by an L4 timing probe) and fails if the total > `time_budget_hours`. | 28 |
| FR-28.2 | Per-phase and per-unit checkpoints; `--resume` continues from the last one, refusing a changed config. | 28 |

## Out of scope (recorded as future work, D-024)

More seeds; ablations A3 / A8 in the run; 300-step fusion; GSM8K; `probe_rule` and other baselines
in the sequential stream.

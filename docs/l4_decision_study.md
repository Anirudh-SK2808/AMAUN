# L4 decision study — 12-hour single-seed run

Written for: whoever runs this experiment on the NVIDIA L4 and whoever reviews its outputs.
Config: `configs/experiments/l4_decision_study.yaml`. Driver: `src/amaun/continual/decision_study.py`.
Entry point: `scripts/run_decision_study.py`.

**Status.** Implemented and verified end to end on the `dummy` tier (CPU, synthetic data):
`tests/test_decision_study.py` and `tests/test_memory_and_numerics.py`. Real-data split
preparation was run and checked (§3). **No training or evaluation has run on the L4.** Every
statement below about L4 memory or wall time is an estimate until the run's own
`decision_study/preflight.json` and `summary.json` replace it.

---

## 1. Question

When a new task arrives that is *related* to an existing adapter, does the learned utility network
choose a better integration (REUSE / FUSE / CREATE) than simple policies? "Better" is scored on
outcomes measured on TEST, not on the integrated model's score alone.

## 2. Assignments

| Adapter | Task id | Dataset | Role | Training rows |
|---|---|---|---|---|
| `seed-sum` | `sum` | CNN/DailyMail 3.0.0 | existing repository adapter | 6 000 |
| `seed-qa` | `qa` | SQuAD v1.1 | incoming task designed to test fusion | 6 000 |
| `seed-sentiment` | `sentiment` | IMDb | repository adapter (role not specified) | 6 000 |

Three adapters, three datasets. No two adapters share training rows: the run computes the
content-hash intersection of their TRAIN splits and records it in
`decision_study/initial_adapters.json` (`shares_training_rows_with`, expected `{}`).

| Evaluation task | Task id | Dataset | Kind | Expected relation |
|---|---|---|---|---|
| ELI5 long-form explanation | `longform` | `Pavithree/eli5` | similar | `seed-sum` |
| FinQA numerical QA | `domain` | `dreamerdeo/finqa` | similar | `seed-qa` |

`related_to` is recorded for reporting only. No policy reads it.

**No different-task evaluation task was specified.** Each evaluation task is still measured
against every candidate, so ELI5 × `seed-qa`/`seed-sentiment` and FinQA × `seed-sum`/`seed-sentiment`
are unrelated pairings with measured outcomes. To add a genuinely different arrival, append
`{task: dialog, kind: different, related_to: null}` to `decision_study.evaluation_tasks`, and add
`dialog` to `datasets.order` and `data.volumes`. DailyDialog is already pinned.

The original brief mentioned "3 adapters across 2 initial tasks"; the assignment you supplied is
3 adapters on 3 tasks. The run follows the latter.

## 3. Final resolved configuration

| Item | Value | Where |
|---|---|---|
| Backbone | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798…` bf16, sdpa, no quantisation | registry `local_smoke`; `model.quantization: none` |
| Run mode | `local_smoke` → `results/local_smoke/l4_decision_study/` (non-reportable) | CLI `--run-mode local_smoke` |
| Seed | 1234 (one seed) | `experiment.seeds` |
| TRAIN / VALIDATION / SUPPORT / TEST per task | 6 000 / 750 / 750 / 300 (80/10/10 + TEST), exact, strict; FinQA 6 000 / 64 / 64 / 300 | `data.split_ratios`, `data.split_caps`, `data.task_split_caps` |
| Tasks | sum, qa, sentiment (adapters); longform, domain (evaluation) | `decision_study.*` |
| LoRA | r 16, α 32, dropout 0.05, q/k/v/o/gate/up/down | `configs/model/qwen.yaml` |
| Training | 1 epoch = 375 optimiser steps (batch 4 × accumulation 4), lr 2e-4 cosine, no early stop | `training.*` |
| Input / target tokens | sum 512 / 128; domain 768 / 128 (input restored to the curriculum value); qa 512/48; sentiment 512/4; longform 256/512 | `datasets.tasks.*` |
| Prompt truncation | `middle` for all five tasks (§6.1) | `datasets.tasks.*.prompt_truncation` |
| Gradient checkpointing | on for training (FinQA's longest batch does not fit without it, §6.3); fusion is not checkpointed | `training.gradient_checkpointing` |
| Intermediate evaluation | 16 VALIDATION examples (loss every 125 steps; generation checks in supervision) | `training.intermediate_eval_examples`, `decision_study.supervision_generation_examples` |
| Final evaluation | 150 TEST examples per (adapter, task) | `evaluation.max_test_examples` |
| Fusion | layer-wise softmax α (init 0.5), 100 steps, KD λ 1, T 2, 16 retention examples from TRAIN | `fusion.*`, `kd.*` |
| Decision | `prescreen_then_train`: REUSE gate before training, then top-1 FUSE/CREATE; τ selected on supervision (`dev_stream`), one-class policy `conservative` | `manager.*`, `utility.reuse_gate` |
| Time budget | 12 h wall clock; later work is skipped and recorded | `decision_study.time_budget_hours` |

The full resolved tree, package versions, git state and GPU name are written to
`config_used.json` and `provenance.json` at start-up.


### 3a. Measured split sizes

Measured on 2026-10-22 (re-measured after the 80/10/10 change) with `scripts/run_decision_study.py --run-mode local_smoke --prepare-only`
(real pinned datasets, no model). It was re-run offline from the snapshot cache with the same result.

| Task | Training pool | Ratios | TRAIN / VAL / SUPPORT before cap | Final TRAIN | VALIDATION | SUPPORT | TEST |
|---|---|---|---|---|---|---|---|
| sum | 9 000 | 80/10/10 | 7 208 / 917 / 875 | 6 000 | 750 | 750 | 300 |
| qa | 9 000 | 80/10/10 | 7 206 / 891 / 903 | 6 000 | 750 | 750 | 300 |
| sentiment | 9 000 | 80/10/10 | 7 193 / 951 / 856 | 6 000 | 750 | 750 | 300 |
| longform | 9 000 | 80/10/10 | 7 168 / 935 / 897 | 6 000 | 750 | 750 | 300 |
| domain | 6 203 (48 of 6 251 rows have empty answers and are dropped by the row adapter) | 97/1.5/1.5 | 6 041 / 81 / 81 | 6 000 | 64 | 64 | 300 |

No task falls short of any cap.

IMDb class balance: TRAIN 2 953 negative / 3 047 positive; TEST 151 / 149.

**Truncation, measured with the real Qwen2.5 tokenizer on the first 150 TEST prompts per task:**

| Task | Prompts over the input limit | Answer cue + assistant header kept, `head` | kept, `middle` |
|---|---|---|---|
| domain (FinQA, 512) | 146 / 150 | 4 / 150 | 150 / 150 |
| sum (CNN/DM, 512) | 121 / 150 | 29 / 150 | 150 / 150 |
| sentiment (IMDb, 512) | 19 / 150 | 131 / 150 | 150 / 150 |
| qa (SQuAD, 512) | 1 / 150 | 149 / 150 | 150 / 150 |
| longform (ELI5, 256) | 0 / 150 | 150 / 150 | 150 / 150 |

Under the original head truncation, almost every FinQA prompt at 512 tokens would have lost its
question.

**FinQA input restored to 768 tokens.** Measured FinQA prompt lengths (real Qwen2.5 tokenizer):
the median is ~1 014 (TEST) / ~1 042 (TRAIN) tokens, the 90th percentile ~1 500, and the maximum
3 300. At 768, 122 / 150 TEST prompts (81%) and 5 025 / 6 000 TRAIN prompts (84%) are still
truncated in the middle, against 146 / 150 at 512. The question is always kept; part of the
context (often table rows) is not. Covering 90% of prompts would need ~1 500 input tokens.
The backbone allows 32 768 positions, so that limit is memory and time, not the model.

FinQA is the one task that is not 80/10/10. It has 6 203 usable upstream train rows, fewer than
the 7 500 that 6 000 / 750 / 750 needs, and at 80/10/10 its TRAIN split would be about 4 960. It
therefore uses `data.task_split_ratios.domain` 0.97/0.015/0.015 and `data.task_split_caps.domain`
64 / 64, which keeps 6 000 TRAIN rows. FinQA is an evaluation task, so its small VALIDATION split
is never used for utility supervision. Its SUPPORT split (64 rows) covers the 32 task-representation
rows and the 16 probe rows.

## 4. Procedure

```
0 preflight      worst-case train step, fusion step and generation per task at batch 4
1 initial        train seed-sum, seed-qa, seed-sentiment from θ0 on TRAIN; register
                 standalone TEST score of each on its own task; zero-shot TEST of all 5 tasks
2 supervision    for each ordered pair (new=A_i, old=A_j), i≠j  -> 6 integrations
                   features: the runner's probe protocol on new task's SUPPORT
                   fuse (A_j, A_i): 100 steps, KD teacher = A_j on A_j's TRAIN
                   measure on VALIDATION: loss of A_i, A_j, fused on both tasks;
                                          16-example generation score of the same
                   y = measure_outcome_targets(...)   (existing definition, loss→score proxy)
                   gate label: VALIDATION gap score(A_j on task i) - score(A_i on task i),
                     gate features: A_j on task i's SUPPORT (the decision-time function)
                 fit normaliser + g_ω + τ + REUSE gate on these 6 only; select fixed-rule
                 threshold on the same 6
                 freeze
3 evaluation     per evaluation task (independent of each other):
                   REUSE gate on SUPPORT (no training): fires -> amaun decides REUSE here
                   train temp adapter on its TRAIN (kept even when the gate fired, to measure
                     the counterfactual options; the saved training time is reported)
                   features vs every repository adapter (SUPPORT)
                   every policy decides; decisions logged            <- no TEST read yet
                   measure on TEST: CREATE, REUSE_j, FUSE_j for all 3 candidates
                   score each policy's choice on the measured options
4 memory audit   3 × (load → evaluate → fuse → evaluate → release → gc) with snapshots
```

### Policies compared

| Policy | Rule | Threshold source |
|---|---|---|
| `amaun` | REUSE gate before training; if closed, learned utility network decides FUSE / CREATE | gate and τ_u fitted on phase-2 measurements |
| `amaun_train_first` | same utility network, always trains first (`train_then_decide`), REUSE / FUSE / CREATE | τ_u, τ_reuse from phase 2 |
| `always_merge` | FUSE with the candidate nearest by `semantic.d_task_sq` | none |
| `always_separate` | CREATE (B2) | none |
| `fixed_rule` | single-metric similarity on `semantic.d_task_sq` (B4) | selected on the same phase-2 measurements |

All four use the same fusion procedure, the same temporary adapter, and the same measured
options, so the only difference between them is the decision.

### Metrics reported per option and per policy

New-task score after integration and for a separate adapter, negative transfer vs a separate
adapter, per-old-task score and retention ratio, forgetting, average performance across all
four tasks, adapters retained, repository bytes, backbone bytes, integration seconds, and
evaluation seconds. The best measured option is reported as an oracle for regret only. Each
choice is also flagged `unnecessary_create` or `harmful_integration` when it meets those
definitions within `decision_study.outcome_tolerance`.

### REUSE before training (stage-1 gate)

The gate (`src/amaun/manager/reuse_gate.py`) predicts the reuse gap, defined as an existing
adapter's score on the new task minus a dedicated adapter's score. It uses nine features that
exist without a temporary adapter: task distance, the candidate's loss / confidence / entropy /
ECE / token accuracy on the new task's SUPPORT, the frozen backbone's loss there, and the
candidate's gain over the backbone. It is a ridge regression, so it is not a single-metric rule
(idea.md §16.2). It fires when the best eligible candidate's predicted gap is at least
`-utility.reuse_gate.tolerance` and that candidate beats the frozen backbone on SUPPORT.
Gate features are produced by the same function (`ContinualRunner.reuse_gate_features`) during
supervision and at decision time.

The study measures the gate against TEST counterfactuals for every candidate
(`evaluation_<task>.json` → `reuse_gate.predicted_vs_actual`: predicted vs measured gap, and
whether each said "ok"). It also reports the training time and optimizer steps a deployment would
have saved. The gate learns from the same 6 phase-2 pairs as the utility network, all
cross-task, so it is expected to stay closed on these tasks. That result is correct, not a
failure: none of the initial pairs was a case where reuse was acceptable.

## 5. Leakage controls

- Phases 1–2 read TRAIN / VALIDATION / SUPPORT only. Utility targets, the normaliser, g_ω, τ and
  the fixed-rule threshold are all fitted on phase-2 VALIDATION measurements.
- KD retention data comes from TRAIN, so VALIDATION is never both the data KD optimises and
  the data that measures retention.
- In phase 3 every policy's decision is logged before any TEST read of that task, and the gate's
  decision is logged before any TRAIN read of that task: it reads SUPPORT only
  (`tests/test_decision_study.py::test_gate_decides_before_training_and_its_reuse_is_the_amaun_decision`,
  which fails if a TRAIN read is injected before the gate). TEST outcomes
  are never fed back. The two evaluation tasks are independent: neither sees the other's
  integration, and the repository is not mutated.
- The zero-shot and standalone TEST scores from phase 1 are cached for reporting. No fit,
  threshold or decision reads them.
- `tests/test_decision_study.py::test_no_test_split_is_read_before_fitting_or_before_a_tasks_decisions`
  records every split access during a run and asserts this ordering. Injecting a single TEST
  read into phase 2 makes it fail (checked).

## 6. Code changes made for this run

### 6.1 Correctness

| Change | Why |
|---|---|
| `prompt_truncation: middle` option (`data/preprocess.encode_prompt`) | Head truncation at 512 tokens dropped the end of the prompt. That is FinQA's `Question:` and the chat template's assistant header. `head` stays the default for every existing arm. The evaluator now encodes prompts with the same function as training. |
| Bootstrap forgetting measured on all prior tasks (`manager/bootstrap.py`) | The lookup searched `dev_tasks` only, so forgetting on a seed task was silently 0.0 and an unmeasured integration could be labelled beneficial. It is now measured. If it cannot be measured, the example is labelled not beneficial and counted in the notes. |
| KD retention from the teacher's tasks (`continual/runner._retention_batches`) | Retention came from the evaluated stream only, so fusing with a bootstrapped seed adapter ran KD on an empty set. `kd.retention_examples` is now honoured; before, it was 2 per task regardless. |
| Seed adapters carry task embeddings (`runner.bootstrap_repository`) | `semantic.d_task_sq` was a missing-value sentinel for every seed candidate. |
| `one_class_policy` for τ (`manager/threshold.py`) | When every observation is harmful, the median policy places τ inside the harmful range. `conservative` places it above every harmful observation. The default remains `median`. |
| Per-task split ratios and exact strict split caps (`experiment.prepare_data`, `data/splits.TaskSplits.cap`) | Needed for exactly 6 000 TRAIN / 300 TEST rows. A shortfall raises instead of training on fewer rows. |
| IMDb task, `sentiment_classification` type, `label_to_text` row adapter | New dataset. It is registered but not in the default `datasets.order`, so no existing arm changes. |
| Snapshot manifest comparison in JSON form (`data/snapshots.read`) | Nested integer option keys (IMDb's label map) are stringified by JSON, so the cached snapshot never matched and every run re-downloaded; offline runs failed. Found by the offline re-run in §3a. |
| `fit_utility_model` extracted from the bootstrap | The study and the existing bootstrap fit g_ω through the same code. |
| `AlwaysMergeStrategy` | Decision baseline. It uses AMAUN's learnable fusion, unlike B3's uniform coefficients. |

### 6.2 Memory-leak and resource audit

The audit was done by reading the full lifecycle (training → intermediate evaluation → fusion →
final evaluation), followed by a repeated-cycle measurement on the dummy tier.

**Fixed**

| Finding | Effect | Fix | Value preserved? |
|---|---|---|---|
| `masked_cross_entropy` upcast the full `[B, S, V]` logits to fp32 | ~1.6 GB transient at batch 4 × 640 × 151 936 | gather response positions first | yes, test-pinned |
| KD teacher cache held fp32 `[B, S, V]` per retention batch on CPU | ~778 MB per 2-example batch | cache response positions only | yes, test-pinned |
| `kd_loss` made three full-size fp32 copies before masking | GPU transient | mask first | yes, test-pinned |
| Bootstrap `_validation_loss` used one un-batched forward pass | logits grew with example count (OOM'd on MPS) | batched, token-weighted | yes, test-pinned |
| CUDA peak memory never reset | every adapter after the first reported the process peak | reset per adapter | n/a |
| Trainer kept `logits`, optimiser, scheduler and parameter list alive to function end | delayed release | explicit `del`; before/after snapshots in `TrainingRecord.memory` | n/a |
| FUSE built `list(loader)[:steps]` | tokenised all ~1 500 TRAIN batches to use 100 | `itertools.islice` (runner and study) | yes, same batches |
| `evaluate_loss` / `_validation_loss` used `no_grad` | autograd version-counter overhead | `torch.inference_mode()` | yes |
| `FusedLoRALinear` scaled the `d_out`-wide term by α, so backward saved an fp32 `[B, S, d_out]` tensor per component per module | fusion+KD step out of memory at batch 4 on every task (L4) | scale the rank-r intermediate instead | yes, test-pinned (`tests/test_fusion_memory.py`) |
| Fusion kept the task graph alive through the KD retention forward | both graphs resident in one step | one backward per loss term (`backward_fusion_losses`) | yes (same summed gradient), test-pinned |
| Preflight chose its worst case by characters, not tokens | checked FinQA at 774 tokens; TRAIN reaches 884, which ran out of memory in training | `worst_case_examples` by token count after truncation | n/a, test-pinned (`tests/test_decision_study_preflight.py`) |
| Preflight never applied `training.gradient_checkpointing` | could not check that setting | train_step check enables it as `train_adapter` does | n/a, test-pinned |

**Checked, no problem found**

- One backbone instance. `BackboneCache` is single-slot and LoRA is injected once per model.
  Adapter switching copies factors into the same modules.
- Temporary adapters live in the candidate store, are discarded after each evaluation task, and
  are held as CPU deltas.
- Fused modules are always removed in a `finally` block. `FusionState` and the optimiser are local.
- The evaluator stores Python floats and strings only, never logits.
- No forward or backward hooks are registered anywhere in `src/` or `scripts/`.
- There are no DataLoader workers or pinned memory: `GuardedLoader` is in-process. Datasets are held
  once as `Example` lists.
- Checkpoints are written once per registered adapter. The study does not save fused deltas,
  only JSON logs.

**Measured on the dummy tier (CPU).** 3 cycles: live-tensor count 157 → 157 → 157, CPU RSS
unchanged, verdict `stable`. This shows that Python-side references are not accumulating. It
says nothing about the CUDA allocator: GPU `allocated` / `reserved` are only measured on the L4,
by phase 4 of the real run, which writes `decision_study/memory_audit.json` and
`decision_study/resources.jsonl`. `stability_report` treats cycle 0 as warm-up and judges
growth from cycle 1. Growth in `reserved` without growth in `allocated` is reported as
allocator caching, not a leak.

`torch.cuda.empty_cache()` is called only inside `cleanup(release_cache=True)` after an OOM in
preflight. Cleanup otherwise relies on dropping references and `gc.collect()`.

### 6.3 Batch size 4 on the L4 (measured)

Measured on the L4 on 2026-10-22 (`results/local_smoke/l4_time_estimate/`). The preflight runs, per
task, the longest TRAIN batch **by token count** through one training step, one fusion+KD step,
and one generation from the longest prompt. On out-of-memory it writes the error and a per-check
adjustment to `decision_study/preflight.json`, exits with status 3, and changes nothing itself.

Training step at batch 4 on the token-longest rows, peak allocated GiB (usable: 22.0):

| Task (tokens) | No checkpointing | Checkpointing (configured) |
|---|---|---|
| sum (641) | 16.2 | 5.4 |
| qa (522) | 14.1 | 4.9 |
| sentiment (514) | 13.6 | 4.9 |
| longform (621) | 18.2 | 7.4 |
| domain / FinQA (884) | out of memory (also with `expandable_segments`) | 6.3 |

Fusion+KD step (not checkpointed) after the §6.2 fix: sum 15.2, qa 13.3, sentiment 12.8,
longform 17.3, **FinQA 20.4 GiB allocated, 21.8 reserved**. FinQA fusion fits with little margin.
The preflight's retention batch is the same 884-token rows; in the run, retention comes from the
old adapter's task (at most 641 tokens), so the run's peak is at or below this. Launching with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation and changes no value.

## 7. Does it fit in 12 hours?

**Projected 7.8 h; 12.7 h if every generation runs to its cap.** Measured on the L4 by
`scripts/estimate_decision_study_time.py` (2026-10-22, commit `bb73604`, checkpointing on), which
times the study's own code paths and multiplies by its work counts. It reads no TEST data.

| Work | Count |
|---|---|
| LoRA trainings | 5 × 375 steps (1 500 micro-batches each) |
| Fusion + KD runs | 6 supervision + 6 evaluation, 100 steps each (+ preflight and audit) |
| TEST evaluations (150 examples, greedy, one example at a time) | 3 standalone + 5 zero-shot + 20 option evaluations = 28 |
| VALIDATION loss passes (750 examples) / generation checks (16 examples) | 21 / 15 |

Measured throughput: 0.73–2.0 s per training micro-batch (FinQA slowest), 0.9–1.9 s per fusion
step, and **~75 ms per generated token** at batch 1 (2–4× the earlier assumption), plus ~0.08 s
per prompt.

| Phase | Expected (h) | All at cap (h) |
|---|---|---|
| Preflight | 0.03 | 0.03 |
| Phase 1: train 3 adapters | 1.38 | 1.38 |
| Phase 1: standalone + zero-shot TEST | 1.37 | 1.95 |
| Phase 2: supervision (probes, fusion, loss, checks) | 0.73 | 0.92 |
| Phase 3: ELI5 (train, fusion, 10 TEST evals) | 2.81 | 3.88 |
| Phase 3: FinQA (train, fusion, 10 TEST evals) | 1.41 | 4.43 |
| Phase 4: memory audit | 0.04 | 0.07 |
| **Total** | **7.77** | **12.66** |

"Expected" assumes trained adapters stop at the reference length (VALIDATION targets: sum 63,
qa 6, sentiment 2, ELI5 95, FinQA 5 tokens) and zero-shot stops where the backbone stopped (sum
119, ELI5 123, FinQA 98). The largest uncertainty is FinQA's three REUSE options: an adapter
from another task may answer at length. At the zero-shot length they add ~0.9 h (≈ 8.6 h total).
On the dummy tier the estimator came in 16% under the measured run (untimed per-call overhead),
so read the projection as roughly 8–9 h.

Generation is over half of the expected total. If the 12 h budget is reached, remaining items are
skipped and listed in `summary.json`; phase 3 FinQA comes last. The largest lever is
`evaluation.max_test_examples`: 100 instead of 150 cuts about a third of evaluation time.

## 8. Limitations

1. **Six supervision examples.** Three adapters give 6 ordered integrations, all cross-task. The
   utility network cannot be claimed to generalise from that. The run says so in
   `utility_supervision.json` (`limitations`). If all six have the same beneficial label, τ comes
   from the one-class policy, not from learning.
2. **Two evaluation tasks, one seed.** Policy differences are descriptive. No significance test
   is possible, and the code does not run one.
3. **Proxy targets.** Supervision targets use the existing `exp(-loss/10)` proxy, not a
   generated-text metric. Generation scores on 16 VALIDATION examples are recorded beside them
   as a check, but they are not used as targets.
4. **Missing feature families.** With no prior stream history, the continual family
   (`F_i, KR_i, P_i, S_i, signed_transfer`), `historical.PerfHist_i`,
   `historical.StepsSinceUse_i` and `performance.primary_score_cand` are sentinel values. They are
   missing identically in supervision and evaluation, so this is consistent, but they carry no
   signal.
5. **Mixed metrics in averages.** "Average performance" averages ROUGE-L, token F1 and exact
   match, as the existing CL metrics do. Per-task scores are reported alongside.
6. **ELI5 target length.** Training targets go up to 512 tokens, but generation is capped at
   `generation.max_new_tokens: 128` for every task, so ELI5 ROUGE-L is computed on truncated
   outputs. This is pre-existing behaviour shared by all arms, and it is unchanged here.
7. **IMDb ordering.** IMDb rows are sorted by label, so `data.shuffle_buffer_size: 25000` covers the
   whole split to avoid a class-skewed sample.
8. **Remaining defect in the existing bootstrap, not used by this study.**
   `bootstrap_utility_network` builds features from a reduced context (losses only), while
   decisions use the full probe protocol, so its train-time and decision-time features differ.
   The study avoids this by using the runner's probe protocol for both. Fixing the bootstrap
   itself is a separate change.
9. **IMDb licence.** The dataset card lists the licence as `other`. Re-verify it before any release.

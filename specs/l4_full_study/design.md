# l4_full_study — Design

How the requirements in `requirements.md` map onto the code. The run-1 `DecisionStudy`
(`src/amaun/continual/decision_study.py`) stays as it was so run 1 remains reproducible; the
full study is a subclass that reuses its measured primitives and replaces its orchestration and
scoring.

## 1. Module layout

| Module | Responsibility |
|---|---|
| `continual/full_study.py` | `FullStudy(DecisionStudy)`: phases 0–7, checkpoints, reports. |
| `continual/study_criteria.py` | Pure, model-free rules: relative drop, acceptability, oracle, policy scoring (capped / uncapped), fusion selection, `fusion_viable`, eligibility, `probe_rule`, simple-policy candidate rules, paired bootstrap CI. Unit-tested on numbers. |
| `continual/study_state.py` | `StudyCheckpoint`: JSON state under `decision_study/checkpoint/`, unit-level completion records, config-hash guard. |
| `continual/time_projection.py` | Workload model per phase × unit costs (run-1 timings, scaled; timing probe overrides). |
| `continual/sequential_stream.py` | Phase 6: an in-memory repository that each decision mutates; R2 / R3. |
| `continual/data_integrity.py` | FR-23.1: run-1 split verification; split row digests. |
| `manager/utility_cv.py` | Measured targets, gating (FR-14.1), leave-one-task-out CV, τ from out-of-fold, refit. |
| `evaluation/bertscore.py` | BERTScore scorer (pinned model, layer, baseline); dummy tier builds a tiny random encoder. |

Edits to existing modules:

- `adapters/lora.py` — `load_delta_into` rescales `B` by `delta.scaling / layer.scaling[adapter]`
  (D-017) and resizes the LoRA factors to the delta's rank; `reset_lora_parameters` restores the
  spec rank first, so training always runs at rank r.
- `adapters/delta.py` — `AdapterDelta.storage` (`factors` | `dense`) drives `storage_bytes()`;
  saved and loaded.
- `fusion/fused_module.py` — `AlphaParameterisation.FREE`.
- `fusion/layerwise.py` — `free` init / global vector; init checks only for normalised forms.
- `fusion/optimize.py` — no teacher when `λ = 0` or `steps = 0`; loss trend; would-be SVD error.
- `fusion/refactorize.py` — `would_be_svd_error`, `stored_bytes` in the report; dense storage tag.
- `evaluation/evaluator.py` — per-example scores, optional metric scorers (BERTScore), per-task
  example caps.
- `evaluation/task_metrics.py` — `entity_f1`, `entity_f1_micro`, BERTScore through a scorer.
- `data/registry.py`, `data/providers.py`, `data/row_adapters.py` — NER task type, synthetic NER
  rows, `conll_entities`.
- `data/splits.py` — `Example.content_hash` (prompt + target, no task id); exact-count splits.
- `data/leakage.py` — `content_audit` across tasks with declared duplicate groups.
- `experiment.py` — duplicate tasks (`shuffle_as`, `train_offset`, `exclude_pools_of`), exact
  counts, content audit, split digests.
- `utils/seed.py` — `configure_determinism`.
- `manager/utility_network.py` — stored target statistics; `predict_outcomes` stays in z-space.

## 2. Options and cost model

Options for an evaluation task with repository `R` (|R| = 6 seeds):

| Option | adapters after | training seconds | stored bytes after | new-task score | old-task scores |
|---|---|---|---|---|---|
| BACKBONE | \|R\| | 0 | bytes(R) | zero-shot TEST | standalone |
| REUSE:j | \|R\| | 0 | bytes(R) | A_j on TEST | standalone |
| FUSE:j | \|R\| | t_train + t_fuse | bytes(R) − bytes(A_j) + bytes(fused) | fused on TEST | fused on A_j's tasks, standalone elsewhere |
| CREATE | \|R\| + 1 | t_train | bytes(R) + bytes(A_temp) | A_temp on TEST | standalone |

`training_seconds_saved_vs_create = t_train(CREATE) − spent`. Under `max_adapters = 6`, CREATE
is infeasible; any policy's CREATE is executed as BACKBONE (D-018), and the oracle is chosen
among feasible options.

## 3. Phase flow

```
P0 preflight      data integrity (run 1) → content leakage audit → memory checks (22.5 GB)
                  → timing probe → time projection (fail > budget) → BERTScore model loadable
P1 seeds          load seed-sum/qa/sentiment (content hash checked) ; train qa-B, instr, dialog
                  (checkpoint each) ; zero-shot + standalone VALIDATION and TEST (R1)
P2 fusion grid    F-0, F-a, F-b, F-c, F-e × 6 pairs (sum/qa/sentiment) on VALIDATION
                  (checkpoint each config×pair) → fusion_selection.json
P3 supervision    30 ordered pairs with the selected config (checkpoint each pair)
P4 utility        gating → LOTO CV → τ from OOF → refit ; reuse-gate ridge ; fixed-rule τ ; freeze
P5 evaluation     train temps (checkpoint) → eligibility (VALIDATION) → decisions (SUPPORT only)
                  → TEST: CREATE, BACKBONE, REUSE×6, measured FUSE set → policy tables (no cap, cap 6)
P6 stream         4 runs (amaun, always_separate) × (no cap, cap 6) over 11 tasks, R2/R3
P7 audit          3 identical fusion cycles: memory stability + fused-score spread
```

Every unit (adapter, config×pair, pair, task, stream run) writes its result into the checkpoint
before the next starts. On `--resume` the repository and temporary adapters are reloaded from
disk, caches (standalone / zero-shot scores) are restored, and finished units are skipped.

## 4. Measured labels (FR-9.1, FR-12.1)

For a supervision pair (new `A_i` on task i fused into old `A_j` on task j), with standalone
VALIDATION scores `s_i(A_i)`, `s_j(A_j)` (R1) and fused scores `s_i(F)`, `s_j(F)`, each on
`label_examples[task]` generations:

- relative new score `r = s_i(F) / s_i(A_i)` (1.0 when both are 0, 0.0 when only the standalone is);
- relative old drop `d = max(0, (s_j(A_j) − s_j(F)) / s_j(A_j))`;
- `beneficial = r ≥ 0.98 and d ≤ 0.01`;
- `beneficial_loss_proxy` = run 1's definition on validation losses.

Reuse-gate label: `gap = s_i(A_j) / s_i(A_i) − 1`, "actually OK" iff `gap ≥ −0.02`.

## 5. Utility network (FR-12–FR-15)

`utility_cv.fit_with_cv(cfg, examples)`:

1. gate: n ≥ 20 and beneficial fraction ∈ [0.2, 0.8], else return a fallback record;
2. for each new task t: fit normaliser + network on pairs with new task ≠ t, predict pairs with
   new task = t (out-of-fold);
3. per-head OOF MSE (z-space) and OOF decision accuracy (argmax U vs best measured candidate:
   the acceptable candidate with the highest relative new score, else the highest relative new
   score);
4. τ_u = `select_thresholds` on OOF utilities and measured flags;
5. refit on all pairs; the refitted network and τ_u are frozen.

## 6. Probes (policies)

On the first 16 SUPPORT examples of the new task: probe loss (teacher-forced CE) and probe score
(greedy generation, task primary metric) for the backbone and each candidate. `probe_rule`,
REUSE-only, FUSE-only and the `amaun` BACKBONE step read only these. Probes are SUPPORT-only and
are computed before any TEST read of the task.

## 7. Sequential stream (Phase 6)

State: an ordered list of `AdapterEntry` (the stream repository) and `serving[task] → adapter id |
None`. Each arrival: probes and gate on current entries; for `amaun` the temporary adapter (R2) and
U features; decision; execute (REUSE: serving[t] = j; BACKBONE: None; FUSE: fresh fusion with the
frozen config, replacing j, serving both j's tasks and t; CREATE: add A_t, or BACKBONE at the cap).
Evaluate on TEST the new task and every task whose serving adapter changed (R3); at the end
evaluate every task. Evaluations are cached by (adapter content hash, task), which makes R3 an
equality by construction; the dummy test checks it against the naive loop.

## 8. Checkpoint format

`decision_study/checkpoint/state.json`:

```json
{"config_hash": "...", "experiment_id": "...", "phases": {"P1": "done", ...},
 "units": {"P2:F-a:seed-qa->seed-sum": {...result...}, ...},
 "caches": {"test": {...}, "val_generation": {...}, "val_loss": {...}},
 "resumes": [{"at": "...", "completed_phases": [...]}],
 "elapsed_seconds_before_resume": 1234.5}
```

Adapters live where they always do (`adapters/<run_mode>/<experiment_id>/` and `_candidates/`).
The utility network, normaliser and gate are saved under `utility_network/` and reloaded.

## 9. Time projection

`time_projection.project(workload, unit_costs, budget_hours)` → per-phase seconds, total, `fits`.
Unit costs per task: train seconds per optimiser step, eval seconds per example (prefill+loss) and
per generated token, fusion seconds per step, probe seconds. Sources, in priority: the preflight
timing probe on this host; run-1 `metrics.jsonl` / `initial_adapters.json` / fusion logs scaled by
`max_new_tokens` ratios; for a task with neither, the mean of comparable tasks with the same output
length class. The workload counts mirror the phase loops in §3 (generations × expected length).

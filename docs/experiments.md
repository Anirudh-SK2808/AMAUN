# AMAUN — Running experiments

Written for: whoever executes the runs. Assumes `pytest -q` and the `dummy` smoke test pass.

---

## 0. Status

**No experiment has been run.** The reasons are concrete, not incidental:

1. `actual` (7B + QLoRA) requires CUDA; the development machine is Apple Silicon (D-008).
2. Two datasets are unpinned (D-006), so real-data runs would need `data.provider: synthetic`,
   which produces no meaningful metric.
3. Three decisions that determine what the headline table *means* are still OPEN (D-002, D-003,
   D-005).

Everything below is the procedure, verified end to end on the `dummy` tier.

**L4 decision study.** The 12-hour single-seed run (Qwen2.5-1.5B-Instruct; 3 adapters; ELI5 and
FinQA similar-task evaluation) has its own driver and document: `scripts/run_decision_study.py`,
`docs/l4_decision_study.md`.

## 1. Order of operations

```bash
# PHASE 4-5: scaffolding and dataset validation
.venv/bin/python -m pytest -q
.venv/bin/python scripts/prepare_data.py \
    --config configs/experiments/main_bootstrap.yaml --run-mode dummy

# PHASE 6-7: model loading and a single LoRA
.venv/bin/python scripts/run_smoke.py \
    --config configs/experiments/main_bootstrap.yaml --run-mode dummy
.venv/bin/python scripts/train_adapter.py \
    --config configs/experiments/main_bootstrap.yaml --run-mode dummy --task sum

# PHASE 8-9: six adapters, independently, into a repository
.venv/bin/python scripts/train_all_adapters.py \
    --config configs/experiments/main_bootstrap.yaml --run-mode dummy

# PHASE 10-12: task representation, retrieval, utility network, thresholds
.venv/bin/python scripts/train_manager.py \
    --config configs/experiments/main_bootstrap.yaml --run-mode dummy --dev-tasks sum qa

# PHASE 13-15: fusion, KD, and the continual stream
.venv/bin/python scripts/run_continual_experiment.py \
    --config configs/experiments/main_bootstrap.yaml --run-mode dummy \
    --strategy amaun --dev-tasks sum qa

# PHASE 16: federated
.venv/bin/python scripts/run_federated_experiment.py \
    --config configs/experiments/F1_gated_aggregation.yaml --run-mode dummy --rounds 2

# PHASE 17-19: baselines and ablations
for arm in B2_separate_lora B3_naive_average B4_similarity_selection; do
  .venv/bin/python scripts/run_continual_experiment.py \
      --config configs/experiments/$arm.yaml --run-mode dummy --strategy $arm
done
for arm in A1_reduced_features A3_fixed_alpha A4_no_kd A8_global_alpha; do
  .venv/bin/python scripts/run_continual_experiment.py \
      --config configs/experiments/$arm.yaml --run-mode dummy \
      --strategy amaun --dev-tasks sum qa
done

# PHASE 18: standalone evaluation
.venv/bin/python scripts/evaluate.py \
    --config configs/experiments/main_bootstrap.yaml --run-mode dummy --include-zero-shot

# aggregation (defaults to run_mode=actual; pass --run-mode to inspect a smoke run)
.venv/bin/python scripts/aggregate_results.py --run-mode dummy
.venv/bin/python scripts/aggregate_results.py --run-mode dummy \
    --compare main_bootstrap B2_separate_lora
```

Swap `--run-mode dummy` for `local_smoke` to validate against the real 1.5B backbone, and for
`actual` on a CUDA host to produce reportable results.

## 2. Before any `actual` run

`check_reportable_preconditions` aborts unless all hold:

- `experiment.seeds` lists ≥ 3 seeds (D-012)
- the git tree is clean, or `experiment.allow_dirty_git: true` is explicit
- `curriculum.repository_init` is set (D-002 resolved)
- `pytest -q` green and the smoke test passed

## 3. What each arm answers

| arm | config | hypothesis |
|---|---|---|
| `main_bootstrap` | decision quality against a populated repository | H1, H2, H3 |
| `main_sequential` | growth control from an empty repository | H3 |
| `B2_separate_lora` | the growth reference | H3 |
| `B4_similarity_selection` | single-metric rule | H1 |
| `B5_fixed_fusion` / `A3_fixed_alpha` | coefficient learnability | H2 |
| `A8_global_alpha` | is layer-wise necessary | H2 |
| `A4_no_kd` | KD's contribution | H4 |
| `B6_heuristic_compression` / `A5_no_compression` | compression mechanism | H5 |
| `F1` vs `F2` | is the compatibility gate doing work | H6 |
| `DQ1_decision_quality` | decision confusion matrix | the research question, directly |

`DQ1` is the arm that measures the decision itself rather than inferring it from downstream
score. It is **blocked**: it needs overlap tasks (T7–T10) that plausibly match an existing
adapter. Without them, `CREATE` is always the correct answer and H2/H3 cannot be falsified
(assumption A-14).

## 4. Interpreting output

Each run writes `results/<run_mode>/<experiment_id>/`:

| file | contents |
|---|---|
| `config_used.json` | the full resolved config + provenance; the input to `reproduce.py` |
| `summary.json` | decisions, CL metrics, final repository, budget, efficiency trajectory |
| `decisions.jsonl` | one record per decision: outcome, every `U_i`, every `û_i`, thresholds |
| `metrics.jsonl` | per-task and per-evaluation records |
| `repository_events.jsonl` | the lifecycle journal (register / reuse / fuse / archive / compress) |
| `fusion_logs/*.json` | α trajectories per layer, KD losses, refactorisation error |
| `federated_rounds/*.json` | compatibility groups, weights, measured payload bytes |
| `performance_matrix.json` | `S[i][j]`, so metrics recompute without re-running |
| `budget_ledger.json` | steps, tokens, examples, tuning trials — the parity input |

Anything from `dummy` or `local_smoke` is labelled
`[SMOKE TEST — not a reported result]` and must never enter a table.

## 5. Estimated `actual` cost

`[TO BE FIXED EXPERIMENTALLY]`. The runner logs measured wall time, tokens and steps, so the
estimate is replaced by measurement after the first arm. The dominant term is the utility
bootstrap (D-010): `O(T·n)` fusion+KD runs before the evaluated stream begins. With 26 arms × ≥3
seeds, budget is the reason D-012 is still OPEN.

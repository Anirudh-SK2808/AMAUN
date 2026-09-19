# AMAUN

**Adaptive Multi-criteria Adapter Utility Network** — a federated continual-learning framework
for small/medium language models that manages LoRA adapters dynamically.

> **Research question.** Can a language model dynamically determine whether an incoming NLP
> task should *reuse* an existing LoRA adapter, *fuse* multiple existing adapters, or *create* a
> new adapter, while controlling adapter growth and mitigating continual-learning degradation?

**Status: implementation complete and verified; no experiment has been run.** Every number this
repository can produce comes from a real run, and none has been made. Seven research decisions
are still open (see below) and three of them block the headline experiment.

---

## Quick start

```bash
uv venv --python 3.11 .venv                       # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m pytest -q                     # 274 tests, ~22s, fully offline
.venv/bin/python scripts/run_smoke.py \
    --config configs/experiments/main_bootstrap.yaml --run-mode dummy
```

Both must pass before anything expensive runs — that gate is `idea.md` §14 and is enforced in
code by `check_reportable_preconditions`.

## Run modes

`--run-mode` is **required and has no default** (`idea.md` §0.1). It selects the backbone:

| mode | backbone | purpose | may produce a reported result |
|---|---|---|---|
| `dummy` | tiny randomly-initialised Qwen2, built in-process | tests, CI, wiring checks (seconds, offline, CPU) | **no** |
| `local_smoke` | `Qwen/Qwen2.5-1.5B-Instruct` | full-pipeline validation on a dev box (MPS/bf16) | **no** |
| `actual` | `Qwen/Qwen2.5-7B-Instruct` | real experiments | **yes, only these** |

Artefacts are namespaced `results/<run_mode>/<experiment_id>/` and a cross-mode write raises.

## What is implemented

| Component | Where | Notes |
|---|---|---|
| Frozen backbone + fingerprint verification | `src/amaun/models/backbone.py` | `θ` provably unchanged across training and fusion |
| LoRA / QLoRA training | `src/amaun/training/lora_trainer.py` | one loop for every arm; response-only loss |
| Six adapters trained **independently** from `θ₀` | `scripts/train_all_adapters.py` | order-independence verified bit-for-bit |
| Four disjoint splits with typed access guards | `src/amaun/data/` | manager sees SUPPORT only; evaluator sees TEST only |
| Adapter repository + lifecycle journal | `src/amaun/adapters/repository.py` | registration needs a `Decision` capability token |
| Task representation | `src/amaun/manager/task_representation.py` | configurable encoder, SUPPORT only |
| All 7 feature families | `src/amaun/features/` | 54 features; each family independently toggleable |
| Utility network + REUSE/FUSE/CREATE | `src/amaun/manager/` | monotone scalarisation; thresholds selected, never invented |
| Layer-wise learnable fusion + KD | `src/amaun/fusion/`, `training/kd.py` | exact ΔW-level fusion; `α` init 0.5; teacher = pre-merge adapter |
| Utility-based compression | `src/amaun/manager/compression.py` | same mechanism as selection (`idea.md` §16.7) |
| Continual runner + CL metrics | `src/amaun/continual/` | AvgPerf, Forgetting, BWT, FWT |
| Federated layer | `src/amaun/federated/` | compatibility-gated delta-level FedAvg, measured payloads |
| Baselines B1–B6 + citation gate | `src/amaun/baselines/` | LoRM/FedALoRA/FedZKD refuse without references |
| 12 ablations A1–A12 | `configs/experiments/` | each toggle proven non-trivial by a test |
| Statistics | `src/amaun/evaluation/stats.py` | ≥3 seeds enforced; paired bootstrap; effect sizes |

## Design decisions that carry scientific weight

1. **`AdapterDelta` is the only currency.** Every comparison, feature, fusion and aggregation
   operates on effective updates `ΔW = scaling·B@A`. There is no API that hands out `A` and `B`
   for arithmetic, so the error `idea.md` §16.3 forbids (`mean(B)@mean(A) ≠ mean(B@A)`) is hard
   to commit rather than merely discouraged. `tests/test_adapters.py` proves the two differ.
2. **`FusedLoRALinear` never materialises `ΔW`.** It computes
   `Σ_k α_k·s_k·((x A_kᵀ) B_kᵀ)`, exactly equal to applying the dense fused delta, in `O(r)`
   memory instead of `O(d_in·d_out)`. At 7B this is the difference between runnable and not.
3. **Guards are types, not conventions.** The manager is *given* a guard permitting `{SUPPORT}`.
   There is no code path where it holds a TEST handle, so leakage requires an edit a reviewer
   can see.
4. **Budget parity is mechanical.** `BudgetLedger` refuses to emit a comparison between arms
   whose steps, tokens, examples or tuning trials diverge beyond tolerance — the enforcement of
   "could the gain come from extra training rather than the method?"
5. **`α = softmax(a)` generalises `idea.md`'s pairwise `sigmoid`.** At `m=2` the two agree to
   float32 precision (verified: 5.96e-08), including the mandated `α⁽⁰⁾ = 0.5`.

## Open decisions (these block reported results)

See `docs/decision_register.md`. Three matter most:

- **D-002** — `bootstrap_six` vs `empty_sequential` repository initialisation. Your brief and
  `idea.md` specify different protocols that answer different questions and produce different
  headline tables. Both are implemented; neither is the default.
- **D-003** — the three-way REUSE/FUSE/CREATE decision is an addition beyond `idea.md`'s
  two-way MERGE/CREATE. Both decision orders are implemented.
- **D-006** — two datasets are unpinned: "FLAN-style" is not a dataset, and the original ELI5
  HF dataset was withdrawn over Reddit licensing. The loader raises rather than guessing.

Also open: D-005 (what validation selects `τ_u`), D-007 (LoRM/FedALoRA/FedZKD references),
D-011 (federated partition), D-012 (seed count), D-014 (adapter-count accounting).

**Assumption A-14 is the largest methodological risk**: with six maximally distinct task types,
`CREATE` is a priori always correct, so REUSE and FUSE can never be *right* and H2/H3 are
untestable. The evaluated stream needs overlap tasks. `run_continual_experiment.py` warns
loudly when the stream merely re-arrives the seed tasks.

## Reproducing an experiment

```bash
.venv/bin/python scripts/reproduce.py results/actual/<experiment_id>
```

Reports every provenance difference (package versions, git commit, uncommitted-diff hash,
platform) and re-runs with `--execute`.

## Repository layout

```
docs/          the specification: read master_prompt.md then system_specification.md
configs/       every experiment-affecting value; 26 arms in configs/experiments/
src/amaun/     the implementation (see docs/architecture.md for the dependency rules)
scripts/       CLI entry points, all thin wrappers over src/amaun/experiment.py
tests/         274 tests; every invariant in the spec has one
.claude/agents/ the three development agents (architect, implementer, reviewer)
```

## Hard constraints

1. Never fabricate results, datasets, metrics, or citations.
2. Never use a single similarity metric as the decision rule.
3. Never average LoRA `A`/`B` factors independently.
4. The backbone is frozen forever and never transmitted.
5. Never fit normalisation statistics on, or train the utility network on, the test sequence.
6. Never replace utility-based compression with an age/usage heuristic.
7. `α_l = 0.5` at initialisation; the KD teacher is the pre-merge old adapter.
8. The temporary adapter stays out of the repository until the decision completes.

Full list in `docs/master_prompt.md` §3.

## Licensing note

Dataset licences differ and are recorded per task in `configs/datasets/curriculum.yaml`.
**DailyDialog is CC BY-NC-SA 4.0 (non-commercial)**; SQuAD is CC BY-SA 4.0 (share-alike applies
to derivatives). Verify each at download time before any release.

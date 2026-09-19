# AMAUN — Experiment Matrix

Every run has an id, a config path, and a question it answers. `results/<run_mode>/<id>/`.
No result values appear here.

## Main arms

| id | Description | Config | Question |
|---|---|---|---|
| `main_bootstrap` | Full AMAUN, `repository_init=bootstrap_six` | `configs/experiments/main_bootstrap.yaml` | decision quality against a populated repository (D-002 O1) |
| `main_sequential` | Full AMAUN, `repository_init=empty_sequential` | `configs/experiments/main_sequential.yaml` | growth control from an empty repository (D-002 O2, `idea.md` §3) |

## Baselines (all reproducible from the two source documents)

| id | Method | Config | Notes |
|---|---|---|---|
| `B1` | Sequential full fine-tuning | `configs/experiments/B1_sequential_full_ft.yaml` | upper bound on plasticity, worst forgetting; only arm where the backbone is *not* frozen — run under an explicit `backbone_trainable: true` flag that is otherwise rejected |
| `B2` | Separate LoRA per task | `configs/experiments/B2_separate_lora.yaml` | no forgetting, unbounded growth; the growth reference for H3 |
| `B3` | Naive uniform LoRA averaging | `configs/experiments/B3_naive_average.yaml` | delta-level uniform average (never A/B averaging) |
| `B4` | Similarity-based selection | `configs/experiments/B4_similarity_selection.yaml` | single-metric rule; the direct H1 comparison |
| `B5` | Fixed-coefficient fusion | `configs/experiments/B5_fixed_fusion.yaml` | fusion with non-learnable α |
| `B6` | Heuristic compression | `configs/experiments/B6_heuristic_compression.yaml` | age/usage deletion; the H5 comparison. Exists **only** as a baseline (`idea.md` §16.7) |

### Registered but NOT implemented (D-007)

`LoRM`, `FedALoRA`, `FedZKD` — raise `BaselineRequiresCitationError`. Implementing a method
from its name alone would fabricate a baseline. Supply references to enable them.

## Ablations

`idea.md` §12.5 ids A1–A8 are preserved verbatim; A9–A11 come from brief §16 and are added.

| id | Toggle | Question | Observable that must change |
|---|---|---|---|
| `A1` | reduced feature set (semantic only) | does heterogeneous evidence help? | `z` dimension, decision distribution |
| `A2` | fixed selection rule instead of `g_ω` | does learned utility help? | decision distribution |
| `A3` | fixed α instead of learnable | does learnability help? | α trajectory is constant |
| `A4` | `λ = 0` (no KD) | KD's contribution | loss terms present |
| `A5` | no repository compression | cost of unmanaged growth | repository trajectory |
| `A6` | no historical features | does adapter history help? | `z` dimension |
| `A7` | no resource features | does cost info change selection? | `z` dimension |
| `A8` | one global α instead of layer-wise | is layer-level flexibility needed? | α tensor shape |
| `A9` | REUSE-only (no fuse, no create) | is fusion/creation necessary? | decision distribution |
| `A10` | FUSE-only | is creation necessary? | decision distribution |
| `A11` | CREATE-always | the no-management control | adapter count = task count |
| `A12` | no task representation (`token_stats` only) | does semantic representation matter? | `z` dimension |

`tests/test_ablation_toggles.py` asserts each toggle changes its observable on the `dummy`
tier, so no ablation can silently be a no-op.

## Federated arms

| id | Description | Question |
|---|---|---|
| `F1` | compatibility-gated aggregation | H6 |
| `F2` | naive aggregation of all adapters | is the compatibility gate doing work? |
| `F3` | non-IID partition (`disjoint_task_subsets`) | realistic FCL setting |
| `F4` | Dirichlet partition | skew sensitivity |
| `F5` | no aggregation (fully local) | does federation help at all? |

## Decision-quality arms (needed by D-002 O1)

| id | Description | Question |
|---|---|---|
| `DQ1` | overlap-task stream with decision ground truth (T7–T10, protocol §3) | decision accuracy / confusion matrix over {REUSE, FUSE, CREATE} |

This is the only arm that measures the decision *directly* rather than through downstream
score, and it is what makes the core research question answerable rather than merely
correlated with performance.

## Run order

```
dummy tests  →  local_smoke full pipeline  →  DQ1 (cheapest real signal)
→  main_bootstrap, main_sequential  →  B1..B6  →  A1..A12  →  F1..F5
```

Every arm runs for every seed in `experiment.seeds`. Total reportable runs =
`(2 main + 6 baselines + 12 ablations + 5 federated + 1 DQ) × |seeds|` =
`26 × |seeds|` — the dominant budget line, and the reason D-012 is `OPEN`.

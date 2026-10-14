# AMAUN — Code Architecture

## Dependency direction (strict; no cycles)

```
utils  ←  config
  ↑         ↑
data  →  models  →  adapters  →  training
                        ↑            ↑
                     features  →  manager  →  fusion
                                     ↑          ↑
                                 continual  ←────┘
                                     ↑
                                federated
                                     ↑
                    evaluation   baselines   (both read everything below)
                                     ↑
                                  scripts
```

Rules: a module may import only from layers below it. `manager` never imports `evaluation`
(it would gain a path to TEST). `baselines` reuse `training`, `fusion`, and `evaluation`
verbatim so a baseline can never accidentally get a different metric implementation.

## File map

```
src/amaun/
├── config.py                  M0  run modes, BACKBONE_REGISTRY, resolution, hashing
├── errors.py                  typed exceptions (every invariant violation has one)
├── models/
│   ├── backbone.py            M1  load/freeze/fingerprint, single-slot cache, dummy tier
│   └── generation.py              shared decoding config (identical across arms)
├── adapters/
│   ├── delta.py               M3  AdapterDelta: lazy ΔW = scale·B@A
│   ├── lora.py                M3  PEFT injection, extraction, apply/remove
│   ├── metadata.py            M3  AdapterMetadata schema + JSON persistence
│   └── repository.py          M3/M9  repository, candidate store, event journal
├── data/
│   ├── registry.py            M2  TaskSpec table, licenses, revisions, pin checks
│   ├── providers.py           M2  huggingface | synthetic providers
│   ├── row_adapters.py        M2  named row→example rules, no default (D-015)
│   ├── snapshots.py           M2  content-keyed bounded offline dataset cache
│   ├── splits.py              M2  deterministic hash-based 4-way split
│   ├── preprocess.py          M2  prompt templates, tokenisation, response masking
│   ├── loaders.py             M2  SplitAccessGuard-wrapped dataloaders
│   └── leakage.py             M2  content-hash disjointness checks
├── training/
│   ├── lora_trainer.py        M4  the one training loop
│   ├── kd.py                  M8  teacher capture + KD loss
│   └── budget.py              M13 BudgetLedger
├── features/
│   ├── base.py                M6  FeatureFamily ABC, named ordered vectors
│   ├── semantic.py parameter.py gradient.py performance.py
│   ├── continual.py historical.py resource.py
│   ├── extractor.py           M6  orchestration + reference-batch protocol
│   └── normalizer.py          M6  train-only μ/σ, run-mode-namespaced
├── manager/
│   ├── task_representation.py M5
│   ├── retrieval.py           M7  single | top_k
│   ├── utility_network.py     M7  MLP, multi-head, monotone ψ/w
│   ├── targets.py             M7  supervision generation (D-010 policy)
│   ├── threshold.py           M7  τ selection (D-005)
│   ├── decision.py            M7  REUSE | FUSE | CREATE + Decision record
│   └── compression.py         M9  utility-based repository compression
├── fusion/
│   ├── fused_module.py        M8  FusedLoRALinear (no dense ΔW materialisation)
│   ├── layerwise.py           M8  α parameterisation, build/teardown, joint optimisation
│   └── refactorize.py         M8  dense | svd | components storage (D-013)
├── continual/
│   ├── runner.py              M10 per-client task-stream driver
│   └── metrics.py             M10 AvgPerf, Forgetting, BWT, FWT
├── federated/
│   ├── client.py server.py    M11
│   ├── compatibility.py       M11 explicit gate (never assumed)
│   ├── aggregation.py         M11 delta-level FedAvg (never A/B averaging)
│   └── comm.py                M11 measured payload accounting
├── evaluation/
│   ├── task_metrics.py cl_metrics.py efficiency.py
│   ├── evaluator.py           M12 the ONLY TEST reader
│   └── stats.py               M12 seeds, paired bootstrap, effect size
├── baselines/
│   ├── registry.py            B1..B6 + citation-gated stubs
│   └── *.py
└── utils/
    ├── seed.py paths.py tracking.py provenance.py profiling.py logging.py
```

## Key design decisions and why

**1. `AdapterDelta` is the currency, not PEFT model state.**
Every comparison, feature, fusion, and aggregation operates on effective updates
`ΔW = scale·B@A`. This structurally prevents the A/B-averaging error (`idea.md` §16.3):
there is no API that exposes `A` and `B` for arithmetic.

**2. `FusedLoRALinear` never materialises `ΔW`.**
`y = xW₀ᵀ + Σ_k α_k·scale_k·(xA_kᵀ)B_kᵀ`. Exact, and `O(r)` memory instead of
`O(d_in·d_out)`. At 7B, materialising fused deltas for all target modules would not fit;
this is the difference between the method being runnable and not.

**3. Guards are types, not conventions.**
`SplitAccessGuard` is constructed with the permitted split set and wraps the dataset handle.
The manager is *given* a guard permitting `{SUPPORT}`. There is no code path where it holds a
TEST handle, so leakage requires editing the guard — a visible, reviewable act.

**4. `Decision` is a capability token.**
`repository.register(adapter, authorised_by=decision)` means a temporary adapter physically
cannot be registered without a completed decision (`idea.md` §16.10).

**5. Single-slot backbone cache.**
One 7B instance maximum. Teacher and student in KD share the same frozen backbone with
different adapter deltas applied — two adapters, one backbone — which is why KD is feasible
in 24 GB at 1.5B and on a single 40–80 GB GPU at 7B.

**6. `BudgetLedger` gates comparisons.**
Baseline fairness is enforced mechanically rather than trusted (falsification question 9/14).

**7a. Dataset access is config-declared end to end.**
A task's identity (`hf_id` + a 40-hex `hf_revision`), its row→example rule (`row_adapter`), its
volume cap (`data.volumes.<task>`), and its sampling policy (`data.sampling`) all come from
`configs/`, with no code default for any of them. The provider is task-agnostic: there is no
`if task_id == …` branch anywhere in `data/`, so adding or re-pinning a task is a config edit.
The on-disk snapshot cache is keyed by every one of those values, so changing one can never
silently reuse data produced under another.

**7. Run-mode-namespaced everything.**
`paths.py` is the only module that constructs result paths, and it takes `RunMode` as a
required argument, so `idea.md` §0.1.1's separation cannot be bypassed by a stray
`open(...)`.

## Execution modes

| Mode | Backbone | Data | Purpose |
|---|---|---|---|
| `dummy` | tiny local random Qwen2 | `synthetic` | `pytest`, CI, wiring/invariant tests, seconds |
| `local_smoke` | Qwen2.5-1.5B-Instruct, from a pinned local snapshot | real, capped, snapshotted | full pipeline validation, minutes, MPS |
| `actual` | Qwen2.5-7B-Instruct | full | reported experiments, CUDA required for QLoRA |

`dummy` is the only tier that stays offline by construction, and it is therefore the tier that
guarantees *all six* curriculum tasks are exercised (`idea.md` §0.2.1). A `local_smoke` run
covers whatever subset of tasks is fully specified — tasks blocked by an `OPEN` decision are
skipped, reason-coded, and reported, never guessed at.

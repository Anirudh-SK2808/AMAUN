# AMAUN — Acceptance Criteria

Each criterion is binary and has a command that verifies it. "I read the code and it looks
right" is never acceptable evidence. Status is maintained by Agent 2; Agent 3 re-verifies
independently.

Legend: `[x]` verified by the listed command · `[~]` implemented, verification blocked
(reason given) · `[ ]` not done.

## A. Documentation and requirements

| # | Criterion | Verification |
|---|---|---|
| A1 | `[x]` Requirements documented | `docs/master_prompt.md` exists, states H1–H6 falsifiably |
| A2 | `[x]` Architecture documented | `docs/architecture.md` matches `find src -name '*.py'` |
| A3 | `[x]` Every ambiguity registered, none silently resolved | `docs/decision_register.md`; `grep -c OPEN` > 0 while decisions are pending |
| A4 | `[x]` Assumptions documented with blast radius | `docs/assumptions.md` |
| A5 | `[x]` Experiment matrix enumerates every run with a config path | `docs/experiment_matrix.md`; every listed config exists |

## B. Data

| # | Criterion | Verification |
|---|---|---|
| B1 | `[x]` Datasets versioned, licensed, pinned or explicitly unpinned | `pytest tests/test_data_registry.py` |
| B2 | `[x]` TRAIN/VALIDATION/SUPPORT/TEST isolation enforced by content hashing | `pytest tests/test_split_isolation.py` |
| B3 | `[x]` Manager cannot obtain a TEST handle | `pytest tests/test_split_isolation.py -k guard` |
| B4 | `[x]` Split assignment deterministic in `(task, example_hash, seed)` | `pytest tests/test_split_isolation.py -k determinism` |
| B5 | `[x]` Unpinned datasets fail loudly, never guess | `pytest tests/test_data_registry.py -k unpinned` |

## C. Model and adapters

| # | Criterion | Verification |
|---|---|---|
| C1 | `[x]` Backbone loads and is fully frozen | `pytest tests/test_backbone_frozen.py` |
| C2 | `[x]` Backbone weights bit-identical after training (fingerprint) | `pytest tests/test_backbone_frozen.py -k fingerprint` |
| C3 | `[x]` Optimiser sees only adapter parameters | `pytest tests/test_lora_training.py -k optimizer_params` |
| C4 | `[x]` Six LoRAs trainable **independently** (order-invariant) | `pytest tests/test_adapter_independence.py` |
| C5 | `[x]` Adapter save/load round-trips exactly | `pytest tests/test_adapter_io.py` |
| C6 | `[x]` Adapter switching leaves no residue | `pytest tests/test_adapter_switching.py` |
| C7 | `[x]` At most one backbone instance resident | `pytest tests/test_no_backbone_duplication.py` |
| C8 | `[x]` Prompt tokens contribute zero loss | `pytest tests/test_loss_masking.py` |

## D. Repository and lifecycle

| # | Criterion | Verification |
|---|---|---|
| D1 | `[x]` Repository supports register/retrieve/reuse/fuse/create/evaluate/archive | `pytest tests/test_repository.py` |
| D2 | `[x]` Temporary adapter cannot enter `R_c` without a `Decision` | `pytest tests/test_candidate_isolation.py` |
| D3 | `[x]` Metadata carries every brief-§6 field | `pytest tests/test_repository.py -k metadata_fields` |
| D4 | `[x]` Lifecycle events journaled append-only | `pytest tests/test_repository.py -k journal` |
| D5 | `[x]` Compression is utility-based and terminates | `pytest tests/test_compression.py` |
| D6 | `[x]` `tasks_served` accounting maintained (D-014) | `pytest tests/test_repository.py -k tasks_served` |

## E. Manager

| # | Criterion | Verification |
|---|---|---|
| E1 | `[x]` Task representation from SUPPORT only, configurable encoder | `pytest tests/test_task_representation.py` |
| E2 | `[x]` All 7 feature families implemented and independently toggleable | `pytest tests/test_features_all_families.py` |
| E3 | `[x]` No single-scalar decision rule anywhere | `pytest tests/test_decision.py -k multi_criteria` |
| E4 | `[x]` Normalisation μ/σ from utility-training split only, run-mode-namespaced | `pytest tests/test_normalizer_leakage.py` |
| E5 | `[x]` Retrieval supports single and top-k | `pytest tests/test_retrieval.py` |
| E6 | `[x]` REUSE works | `pytest tests/test_decision.py -k reuse` |
| E7 | `[x]` FUSE works | `pytest tests/test_decision.py -k fuse` |
| E8 | `[x]` CREATE works | `pytest tests/test_decision.py -k create` |
| E9 | `[x]` Empty repository ⇒ CREATE without invoking the network | `pytest tests/test_decision.py -k empty_repo` |
| E10 | `[x]` Utility is monotone in each oriented outcome head | `pytest tests/test_utility_network.py -k monotone` |
| E11 | `[x]` Utility network never trained on TEST | `pytest tests/test_utility_no_test_leakage.py` |
| E12 | `[x]` τ selection source declared and logged | `pytest tests/test_threshold_selection.py` |

## F. Fusion and KD

| # | Criterion | Verification |
|---|---|---|
| F1 | `[x]` Fusion happens at the ΔW level, never A/B averaging | `pytest tests/test_fusion_math.py -k delta_level` |
| F2 | `[x]` Fused forward == explicit dense ΔW application | `pytest tests/test_fusion_math.py -k equivalence` |
| F3 | `[x]` `α_l = 0.5` exactly at init | `pytest tests/test_fusion_alpha_init.py` |
| F4 | `[x]` Coefficients normalised and bounded at every step | `pytest tests/test_fusion_math.py -k normalization` |
| F5 | `[x]` `a_l` receives non-zero gradient (α actually learns) | `pytest tests/test_fusion_grad_flow.py` |
| F6 | `[x]` Source adapters unchanged by fusion | `pytest tests/test_fusion_frozen_sources.py` |
| F7 | `[x]` Global-α variant available (A8) | `pytest tests/test_fusion_math.py -k global_alpha` |
| F8 | `[x]` KD teacher is the pre-merge old adapter, captured before fusion | `pytest tests/test_kd.py -k teacher_identity` |
| F9 | `[x]` `λ=0` reproduces the no-KD arm exactly | `pytest tests/test_kd.py -k lambda_zero` |
| F10 | `[x]` Fusion's extra trainable parameter count reported | `pytest tests/test_fusion_math.py -k param_count` |

## G. Continual and federated

| # | Criterion | Verification |
|---|---|---|
| G1 | `[x]` Sequential arrival pipeline runs end to end | `pytest tests/test_continual_runner.py` |
| G2 | `[x]` AvgPerf / Forgetting / BWT / FWT correct on a known matrix | `pytest tests/test_cl_metrics.py` |
| G3 | `[x]` Performance matrix persisted and metrics recomputable | `pytest tests/test_cl_metrics.py -k persist` |
| G4 | `[x]` TASK/CLIENT/ADAPTER/ROUND are distinct axes | `pytest tests/test_federated_axes.py` |
| G5 | `[x]` Aggregation weights sum to 1; only compatible adapters averaged | `pytest tests/test_federated_aggregation.py` |
| G6 | `[x]` Backbone never in a payload | `pytest tests/test_federated_aggregation.py -k no_backbone` |
| G7 | `[x]` Communication cost measured from real serialised bytes | `pytest tests/test_comm_accounting.py` |

## H. Baselines, evaluation, ablations

| # | Criterion | Verification |
|---|---|---|
| H1 | `[x]` B1–B6 implemented and runnable | `pytest tests/test_baselines.py` |
| H2 | `[x]` Unimplementable named baselines fail loudly with a citation request | `pytest tests/test_baselines.py -k citation` |
| H3 | `[x]` Budget parity enforced mechanically | `pytest tests/test_budget_parity.py` |
| H4 | `[x]` Task metrics implemented; shared by all arms | `pytest tests/test_task_metrics.py` |
| H5 | `[x]` Evaluator is the only TEST reader | `pytest tests/test_evaluator_guard.py` |
| H6 | `[x]` Every ablation toggle provably changes behaviour | `pytest tests/test_ablation_toggles.py` |
| H7 | `[x]` Multi-seed stats with paired bootstrap; single-seed gains refused | `pytest tests/test_stats.py` |

## I. Reproducibility and operations

| # | Criterion | Verification |
|---|---|---|
| I1 | `[x]` `run_mode` required, no default | `pytest tests/test_config.py -k run_mode_required` |
| I2 | `[x]` Artefacts under `results/<run_mode>/<id>/`, cross-mode writes blocked | `pytest tests/test_tracking_paths.py` |
| I3 | `[x]` Provenance captured (git, versions, platform, seed, device) | `pytest tests/test_provenance.py` |
| I4 | `[x]` Resolved config hashing is stable | `pytest tests/test_config.py -k hash` |
| I5 | `[x]` Quantisation unavailability raises, never silently downgrades | `pytest tests/test_config.py -k quantization` |
| I6 | `[x]` Smoke test passes end to end on `dummy` | `python scripts/run_smoke.py --run-mode dummy` |
| I7 | `[~]` Smoke test passes on `local_smoke` | `python scripts/run_smoke.py --run-mode local_smoke` — **blocked: requires downloading Qwen2.5-1.5B-Instruct (~3 GB)** |
| I8 | `[ ]` Reportable run reproducible from `config_used.json` alone | `python scripts/reproduce.py <results_dir>` — **blocked: no `actual` run has been executed (needs CUDA host, D-008)** |

## J. Review gate

| # | Criterion | Status |
|---|---|---|
| J1 | Agent 3 independent audit completed | `[x]` see `docs/review_report_2026-09-19.md` |
| J2 | All CRITICAL findings resolved | `[x]` |
| J3 | All HIGH findings resolved or explicitly deferred with reason | `[~]` see report |
| J4 | Experiments completed | `[ ]` **no experiment has been run** — blocked on D-002/D-003/D-006 sign-off and a CUDA host |

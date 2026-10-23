# l4_full_study — Tasks

Ordered. `P` = `.venv/bin/python`. Each task names its requirements and its completion test.
Run `P -m pytest -q` after each group; the whole suite must stay green.

## Group 1 — Adapter install and fusion (FR-1, FR-2, FR-3)

- [ ] T-01 `load_delta_into` honours `delta.scaling` and installs any rank; `reset_lora_parameters`
  restores rank r (FR-1.1, FR-1.4). *Test:* `tests/test_full_study_fusion.py::test_install_*`.
- [ ] T-02 `AdapterDelta.storage` + true stored bytes; `refactorize` reports would-be SVD error
  (FR-1.2, FR-1.3). *Test:* `test_keep_components_round_trips_exactly`, `test_dense_storage_bytes`.
- [ ] T-03 `free` parameterisation; init checks per parameterisation (FR-3.1). *Test:*
  `test_free_alpha_can_sum_above_one`.
- [ ] T-04 `λ = 0` and `steps = 0` paths; loss trend (FR-2.1, FR-2.2, FR-3.2). *Test:*
  `test_lambda_zero_has_no_kd_term`, `test_uniform_average_takes_no_step`.

## Group 2 — Criteria (FR-4 … FR-8, FR-10.2, FR-16 … FR-18, FR-25)

- [ ] T-05 `study_criteria.py` with unit tests on hand-built numbers
  (`tests/test_study_criteria.py`).

## Group 3 — Metrics and evaluation (FR-10.1, FR-19.2, FR-22, FR-24)

- [ ] T-06 `entity_f1`, `entity_f1_micro`; `conll_entities` row adapter; NER task type and
  synthetic rows. *Test:* `tests/test_ner.py`.
- [ ] T-07 BERTScore scorer and registry wiring. *Test:* `tests/test_bertscore.py`.
- [ ] T-08 Evaluator per-example scores and per-task caps. *Test:* `test_per_example_scores_are_written`.

## Group 4 — Data (FR-19.1, FR-20, FR-21, FR-23)

- [ ] T-09 `content_hash`, exact-count splits, duplicate tasks, content audit, split digests.
  *Test:* `tests/test_full_study_data.py`.
- [ ] T-10 Run-1 split verification. *Test:* `test_run1_split_check_*`.

## Group 5 — Utility network (FR-12 … FR-15)

- [ ] T-11 `utility_cv.py`; target statistics in the network. *Test:* `tests/test_utility_cv.py`.

## Group 6 — Orchestration (FR-5, FR-9, FR-11, FR-26 … FR-28)

- [ ] T-12 `configure_determinism`. *Test:* `test_deterministic_mode_is_configured`.
- [ ] T-13 `study_state.py`, `time_projection.py`. *Test:* `tests/test_study_state.py`.
- [ ] T-14 `full_study.py`, `sequential_stream.py`, `scripts/run_full_study.py`,
  `configs/experiments/l4_full_study.yaml`. *Test:* `tests/test_full_study.py` (dummy end to end,
  resume, over-budget preflight, R1–R3, every policy's FUSE measured).

## Group 7 — Before the GPU run (on the VM, not automatable here)

- [ ] T-15 Verify the XSum / CoNLL-2003 / roberta-large revisions resolve on the VM
  (`scripts/run_full_study.py --prepare-only`), and cache roberta-large
  (`scripts/fetch_bertscore_model.py`).
- [ ] T-16 Copy run 1's `results/local_smoke/l4_decision_study/` and the three seed adapters to
  the VM paths named in the config (`decision_study.reference_run`).
- [ ] T-17 Launch; on any restart, relaunch with `--resume`.

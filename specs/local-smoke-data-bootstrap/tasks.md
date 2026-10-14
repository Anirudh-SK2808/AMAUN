# local-smoke-data-bootstrap — Tasks

Ordered. Each task names its traceability and its completion condition. `P` abbreviates
`.venv/bin/python`; `CFG` abbreviates `configs/experiments/main_bootstrap.yaml`.
**Do not mark anything complete during planning.** Run `P -m pytest -q` before T-01 to confirm
the 280-pass baseline.

## Phase A — Backbone snapshot (FR-1, G1)

- [x] **T-01 · Add `revision` to the backbone contract**
  `src/amaun/config.py`: add `BackboneSpec.revision: str | None = None`; leave `DUMMY` and
  `ACTUAL` as `None` (the latter with a comment citing D-001 — it cannot be resolved from a
  non-CUDA host); honour an optional `model.revision` override in `resolve()` alongside the
  existing `model_id` override. `configs/model/qwen.yaml`: add `model.revision: null`,
  `model.cache_dir: data/models`, `model.local_files_only: false`.
  Extend `tests/test_config.py` with a `revision` case asserting `config_used.json →
  backbone.revision` is present and that a `model.revision` override is honoured.
  *Traces:* FR-1.1, FR-1.2. *Done when:* `P -m pytest tests/test_config.py -k revision` passes
  and the rest of `test_config.py` is unchanged and green.

- [x] **T-02 · Thread cache/revision/offline into backbone loading**
  `src/amaun/models/backbone.py::_load_pretrained`: pass `cache_dir`, `revision`, and
  `local_files_only` (read from the resolved config, no literals) to both
  `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`. `dummy` path
  untouched.
  *Traces:* FR-1.2. *Done when:* `P -m pytest tests/test_backbone_frozen.py -q` passes
  unchanged.

- [x] **T-03 · `scripts/fetch_backbone.py`**
  New thin script on `add_common_arguments` + `resolve_from_args`. Refuses `run_mode=dummy`
  with `ConfigError` before any network call; `snapshot_download(repo_id, revision,
  cache_dir=model.cache_dir)`; resolves the actual commit sha; raises if it differs from a set
  `BackboneSpec.revision`; writes `backbone_cache.json` via `ExperimentTracker.write_json`
  with `repo_id`, `revision`, `files`, `bytes_total`, `verified_offline_load`; then re-loads
  with `local_files_only=True` (releasing `BackboneCache` first and after) to set that flag.
  Add `data/models/**` to `.gitignore` and a `data/models/.gitkeep`.
  *Traces:* FR-1.3, FR-1.4, EC-1, EC-2. *Done when:* AC-04 holds
  (`--run-mode dummy` → non-zero, no network).

- [x] **T-04 · Execute the download and pin the sha**
  Run `P scripts/fetch_backbone.py --config CFG --run-mode local_smoke`. Record the resolved
  sha into `BACKBONE_REGISTRY[RunMode.LOCAL_SMOKE].revision` (cross-check against
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, observed 2026-09-22 — if it differs, use the
  real one and note it), then re-run so `backbone_cache.json` records the match. Verify
  offline load with the network disabled:
  `HF_HUB_OFFLINE=1 P -c "import sys; sys.path.insert(0,'src');
  from amaun.config import load_config, resolve; from amaun.models.backbone import load_backbone;
  c=resolve(load_config('configs/experiments/main_bootstrap.yaml', config_root='configs'),
  run_mode='local_smoke', experiment_id='offline-check');
  b=load_backbone(c); print(b.frozen_parameter_count, sum(p.requires_grad for p in b.model.parameters()))"`
  *Traces:* AC-01, AC-02, AC-03. *Done when:* the snapshot file exists (> 1 GB), the manifest
  shows a 40-hex `revision` and `verified_offline_load: true`, and the offline one-liner prints
  a positive frozen count and `0`.

## Phase B — Row adaptation (FR-6, G7)

- [x] **T-05 · `src/amaun/data/row_adapters.py` + `UnresolvedRowAdapterError`**
  Add the error to `src/amaun/errors.py` (subclass of `DataError`, docstring citing D-015).
  New module with the `RowAdapter` protocol, `ROW_ADAPTERS` registry, `resolve(spec)`, and
  `passthrough` implementing today's provider logic verbatim plus the explicit
  `list_target_pick ∈ {first, highest_score}` option. Present-but-empty fields are valid;
  absent required fields raise `DataError`; empty reduced target raises `DataError`.
  *Traces:* FR-6.1, FR-6.3, FR-6.4, FR-6.6, FR-6.7, EC-9, EC-10.
  *Done when:* the module imports with no dependency above M2.

- [x] **T-06 · Implement the four D-015 adapters**
  `dialog_last_turn` (selected by `dialog`), `finqa_pre_table_post` with
  `table_format ∈ {markdown, pipe_rows}` (selected by `domain`, `markdown`), plus the rejected
  but retained `dialog_all_prefixes` and `finqa_text_only`. Each declares
  `required_row_fields`. `dialog_last_turn` emits nothing for a row with < 2 utterances;
  `finqa_pre_table_post` tolerates an empty `table`. Record the fraction of examples whose
  rendered prompt exceeds `max_input_tokens` into the load report, so FinQA truncation is
  measured.
  *Traces:* FR-6.5, FR-6.5a, EC-12, EC-13, design §5. *Done when:* all four are in
  `ROW_ADAPTERS` and selection happens only through config.

- [x] **T-07 · `tests/test_row_adapters.py`**
  Offline. Asserts: `passthrough` output byte-identical to the pre-change provider on a fixed
  row set (for `sum`/`qa` shapes); both `list_target_pick` values on an ELI5-shaped
  `{a_id,score,text}` row, with `highest_score` provably selecting a maximum that is **not** at
  index 0; a dolly-shaped row with `context == ""` renders without error; an ELI5-shaped row
  with empty `answers.text` raises `DataError`; `dialog_last_turn` splits a fixed utterance list
  into `history`/`response` and emits nothing for a single-utterance row;
  `finqa_pre_table_post` renders a markdown table and tolerates an empty one; each of the four
  adapters (including the two unselected ones) is registered and declares
  `required_row_fields`; `row_adapter: null` and an unknown name each raise
  `UnresolvedRowAdapterError` whose message contains `D-015`.
  *Traces:* AC-14.

## Phase C — Snapshots (FR-5, G4)

- [x] **T-08 · `src/amaun/data/snapshots.py`**
  `SnapshotSpec` (with `key()` over all nine content-determining inputs), `SnapshotManifest`,
  `path_for`, `read` (returns `None` on absence or manifest mismatch), `write` (tmp + atomic
  rename, JSONL rows + sibling `.manifest.json`). Rows are raw and restricted to the
  adapter's required fields.
  *Traces:* FR-5.1, FR-5.2, FR-5.3.

- [x] **T-09 · `tests/test_data_snapshots.py`**
  Offline. Key changes when any one of `hf_id`, `hf_revision`, `hf_config`, `split`, `size`,
  `skip`, `sampling`, `row_adapter`, `row_adapter_options` changes; round-trip preserves rows
  exactly; a matching snapshot is served while the HF loader is monkeypatched to raise
  (proving no network call); a mismatched manifest is a miss.
  *Traces:* AC-15, FR-5.4.

## Phase D — Provider (FR-4, FR-5, FR-6)

- [x] **T-10 · Rework `huggingface_pool`**
  `src/amaun/data/providers.py`: add `skip`, `sampling`, `shuffle_buffer_size`,
  `offline_snapshots`, `cache_dir`, and the resolved row adapter as parameters (all supplied by
  the caller from config — no defaults that affect an experiment). Order: build
  `SnapshotSpec` → try `snapshots.read` → else stream, apply
  `.shuffle(seed=derive_seed(data.seed, task_id), buffer_size=data.shuffle_buffer_size)` when
  `sampling == shuffled_buffer`, then `.skip(skip)`, iterate rows, adapt, render, stop at the
  example cap → write snapshot. **Shuffle strictly before skip, and derive the shuffle seed
  deterministically** so both loads of a task see the same order (FR-3.8; without this the
  §4 carving guarantee is void). Return `(examples, LoadReport)` carrying `rows_consumed`,
  `n_examples`, `exhausted`, `source`, `snapshot_key`, `over_length_fraction`. Keep the
  `require_pinned()` guard and the zero-examples `DataError`.
  *Traces:* FR-3.8, FR-4.1, FR-4.3, FR-5.4, FR-5.5, FR-5.6, FR-6, FR-6.5a, EC-6, EC-7.
  *Done when:* `P -m pytest -q` still passes (synthetic path unaffected) and T-09 passes.

- [x] **T-11 · `tests/test_single_split_carving.py` — the leak regression**
  Offline, with a fake iterable loader producing uniquely identifiable rows. Asserts:
  (a) `skip=test_offset` pools share no `example_id`; (b) **with `test_offset = 0`,
  `build_splits` raises `LeakageError`** — i.e. the pre-fix behaviour is caught, which is what
  licenses trusting the fix; (c) `test_offset < rows_consumed` raises `DataError`; (d) under
  `sampling: shuffled_buffer`, two independently constructed streams for the same task yield
  **identical row order**, without which (a) proves nothing.
  Run (b) against the pre-T-12 code first and confirm it fails there for the right reason.
  *Traces:* FR-4.6, FR-4.7, FR-3.8, AC-12, design §4.

## Phase E — `prepare_data` wiring (FR-3, FR-4, FR-7)

- [x] **T-12 · Require volumes, enforce the offset invariant, report blocked tasks**
  `src/amaun/experiment.py::prepare_data`:
  - read `data.volumes[task]` with mandatory `train_pool`, `test`, `test_split`; raise
    `ConfigError` naming task + key when absent; **delete the `1000` / `200` /
    `"validation"` literals**;
  - require `test_offset` when `volumes.test_split` equals the training-pool split; pass it as
    `skip`; after both loads, raise `DataError` if `test_offset < pool_report.rows_consumed`;
  - read `data.sampling` and `data.shuffle_buffer_size` (both mandatory) and validate
    `shuffle_buffer_size ≥ max(train_pool, test)` per task; read `data.offline_snapshots`,
    `data.cache_dir`;
  - resolve the row adapter per task; catch only `UnpinnedDatasetError` →
    `blocked[task] = "unpinned_dataset"` and `UnresolvedRowAdapterError` →
    `blocked[task] = "unresolved_row_adapter"`; let everything else propagate;
  - add `PreparedData.blocked`, keep `unpinned` as the `unpinned_dataset` subset, and carry the
    per-load `LoadReport`s;
  - extend the `PreconditionError` on zero available tasks to cite D-006 **and** D-015.
  *Traces:* FR-3.4, FR-3.7, FR-4.2, FR-4.4, FR-4.5, FR-7.1, FR-7.2, FR-7.5, EC-4, EC-5, EC-14.

- [ ] **T-13 · Extend `tests/test_experiment_preconditions.py`**
  Add `volumes_required` cases (each of `train_pool`/`test`/`test_split` missing,
  `test_offset` missing when `test_split == "train"`, `data.sampling` missing,
  `data.shuffle_buffer_size` missing, and a buffer below `max(train_pool, test)`). Keep the
  existing dummy-tier tests green.
  *Traces:* AC-13, EC-14.

- [ ] **T-14 · `tests/test_prepare_data_blocked.py`**
  Offline. With `provider: huggingface` on the `dummy` tier and the HF loader monkeypatched to
  raise, using **fixture-modified** task specs (one with `row_adapter` blanked, one with
  `hf_id: null`) — *not* the shipped curriculum, which now configures all six tasks: both are
  reported with the right reason code **without the loader being called**, and a loader
  `RuntimeError` for an otherwise-available task propagates out of `prepare_data`.
  *Traces:* AC-16, FR-7.2.

- [ ] **T-15 · Enrich `dataset_report.json`**
  `scripts/prepare_data.py`: add `available_tasks`, `blocked_tasks` (code, human reason,
  decision id), `volumes`, `sampling`, and `snapshots`
  (`task, upstream_split, key, source, rows_consumed, n_examples, exhausted`). Print the
  blocked table under the smoke label. Exit code still driven solely by
  `leakage_audit.clean`.
  *Traces:* FR-7.3, FR-7.4, AC-07…AC-11.

## Phase F — Config: pins, schemas, volumes (FR-2, FR-3, G6)

- [ ] **T-16 · Pin every dataset revision and fix the four schema mismatches**
  `configs/datasets/curriculum.yaml`:
  - `sum` → `hf_revision: <sha>` (replace `"3.0.0"`), keep `hf_config: "3.0.0"`;
  - `qa` → `hf_revision: <sha>`;
  - `instr` → `hf_id: databricks/databricks-dolly-15k`, `pinned: true`, drop
    `unpinned_reason`, `hf_revision: <sha>`, `input_fields: [instruction, context]`,
    `target_field: response`, license `CC BY-SA 3.0` with a share-alike note, a
    `prompt_template` that tolerates an empty `context`, `upstream_test_source` describing the
    offset-carved window;
  - `dialog` → `hf_revision: refs/convert/parquet` plus `hf_revision_resolved: <sha>`, with a
    comment stating that `main` carries only a loading script that `datasets` 5.x cannot run;
  - `longform` → `hf_id: Pavithree/eli5`, `pinned: true`, drop `unpinned_reason`,
    `hf_revision: <sha>`, `input_fields: [title]`, `target_field: answers`,
    `prompt_template` using `{title}`, license recorded as unspecified-on-mirror with the
    Reddit-derived caveat;
  - `domain` → `hf_revision: <sha>`;
  - every task gains `row_adapter` and `row_adapter_options`, set to the D-015 resolutions:
    `passthrough` for `sum`/`qa`/`instr`; `passthrough` with
    `row_adapter_options.list_target_pick: highest_score` for `longform`;
    `dialog_last_turn` for `dialog`; `finqa_pre_table_post` with
    `row_adapter_options.table_format: markdown` for `domain`. Each carries a short comment
    citing the D-015 sub-part it implements.
  Resolve each sha with `huggingface_hub.HfApi().dataset_info(id).sha` (and `list_repo_refs`
  for the conversion branch); cross-check against the values in `design.md` §1.3 and use the
  real ones. `src/amaun/data/registry.py`: carry the new fields on `TaskSpec`.
  *Traces:* FR-2.1…FR-2.8, FR-6.2, FR-6.3, FR-6.5.

- [ ] **T-17 · Define `data.volumes`, `data.sampling`, `data.offline_snapshots`**
  `configs/datasets/curriculum.yaml`: `volumes: null` and `shuffle_buffer_size: null` at base
  level (`actual` volumes remain unresolved — `idea.md` §0.2 gives ranges), `sampling:
  shuffled_buffer` per D-016, `offline_snapshots: true`.
  `configs/experiments/base.yaml` → `run_mode_overrides.local_smoke.data`:
  `provider: huggingface`, an explicit `shuffle_buffer_size`, and per-task `volumes` for **all
  six** tasks with `train_pool`, `test`, `test_split` — plus `test_offset` for `instr`
  (`test_split: train`). Constraints the numbers must satisfy: FR-3.5 (`train_pool ≥
  smoke_max_examples_per_split / 0.15`, `test ≥ cap`); FR-3.7
  (`shuffle_buffer_size ≥ max(train_pool, test)`); and for `instr`,
  `train_pool + test_offset + test ≤ 15011`. `test_split` is `validation` for
  `sum`/`qa`/`dialog`/`longform`, `test` for `domain`, `train` for `instr`. `dummy` overrides
  stay `synthetic`.
  *Traces:* FR-3.1…FR-3.8, FR-4.5, EC-7, EC-8, EC-14.

- [ ] **T-18 · Update `tests/test_data_registry.py` for D-006 RESOLVED**
  Invert `test_unpinned_tasks_fail_loudly` to assert the unpinned set is now **empty** and
  every task passes `require_pinned()` (keep a unit-level check that an *artificially*
  unpinned spec still raises, so B5's guarantee is not lost). Add
  `test_revision_is_pinned` asserting every `hf_revision` matches
  `^[0-9a-f]{40}$|^refs/convert/parquet$`. Keep the license assertions, including `dialog`'s
  non-commercial flag, and add `instr`'s share-alike and `longform`'s unspecified-license
  notes.
  *Traces:* AC-05, AC-06, FR-2.5, FR-2.8, C-8.

## Phase G — Execute and verify on real data

- [ ] **T-19 · First real-data run — all six tasks**
  `P scripts/prepare_data.py --config CFG --run-mode local_smoke`. Expect exit 0,
  `available_tasks == [sum, qa, instr, dialog, longform, domain]` (curriculum order),
  `blocked_tasks == {}`, `unpinned_tasks == []`, `sampling == "shuffled_buffer"`, all four
  split counts equal to the cap for every task, `leakage_audit.clean == true`, and `data/raw`
  populated with a JSONL + manifest per `(task, upstream_split)`. Report `domain`'s recorded
  `over_length_fraction` — FinQA contexts are long against `max_input_tokens: 768` and the
  truncation rate must be a measured number, not an assumption.
  *Traces:* AC-07, AC-08, AC-09, AC-10, AC-12 (artefact half), FR-6.5a.

- [ ] **T-20 · Offline re-run determinism**
  `HF_HUB_OFFLINE=1 P scripts/prepare_data.py --config CFG --run-mode local_smoke
  --experiment-id main_bootstrap_offline`, then diff the two reports' `split_counts` and
  `split_provenance` (they must be identical) and assert every `snapshots[*].source ==
  "local"`.
  *Traces:* AC-11, FR-5.4.

- [ ] **T-21 · `local_smoke` end-to-end smoke — closes `I7`**
  `P scripts/run_smoke.py --run-mode local_smoke`. All checks `PASS`,
  `smoke_report.json.all_passed == true`. Then update `docs/acceptance_criteria.md` `I7` from
  `[~]` to `[x]` with this command as its evidence, and fill the new `B6`–`B12` and `I9` rows'
  status from the commands actually run.
  *Traces:* AC-17, G9.

- [ ] **T-22 · Full regression**
  `P -m pytest -q` (≥ 280 passed, 0 failed) and `P scripts/run_smoke.py --run-mode dummy`
  (`21/21 checks passed`). Confirm `grep -n "1000\|200\|validation\"" src/amaun/experiment.py`
  shows no surviving volume default.
  *Traces:* AC-18, AC-13, C-3, NG3.

- [ ] **T-23 · Record the measured outcome in steering**
  Update `docs/acceptance_criteria.md`'s verification-status header with the newly measured
  pytest count and the `local_smoke` smoke result, and add the two executed commands to
  `docs/experiment_protocol.md` §1 as the concrete form of precondition 2. Only measured
  values — no `[TO BE FIXED EXPERIMENTALLY]` marker may be replaced by an estimate.
  *Traces:* §12 reporting duty; `docs/master_prompt.md` constraint 1.

## Phase H — Decision-resolution follow-through

- [ ] **T-24 · Verify the resolved values are config-only and reversible**
  (Previously blocked on D-015/D-016; both `RESOLVED` 2026-09-22, so this is now buildable.)
  Confirm that switching `dialog` to `dialog_all_prefixes`, `domain` to `finqa_text_only`,
  `longform` to `list_target_pick: first`, or `data.sampling` to `head_of_stream` requires
  **only** a `configs/` edit — no code change — and that each switch changes an observable
  (example count, rendered context, selected target, or snapshot key). Revert afterwards to the
  resolved values. This is the ablation-toggle contract applied to the data layer: a config
  knob that provably changes nothing is a defect.
  *Traces:* C-7, FR-6.2, FR-6.5, design §3.1, §5.

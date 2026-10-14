# local-smoke-data-bootstrap — Requirements

## Overview

Bootstrap the `local_smoke` tier onto **real data and a real (small) backbone**, so that the
real-data path — model download → local cache → offline load → pinned HF dataset → row
adaptation → four-way splits → leakage audit → `dataset_report.json` — is proven end to end
**before** any `actual`-tier (7B, CUDA, QLoRA) run is attempted.

This is `docs/experiment_protocol.md` §1 precondition 2 and acceptance criterion `I7`. It is
**non-reportable by construction** (`RunMode.LOCAL_SMOKE.is_reportable == False`); no number
produced by this feature may appear in any table, and every artefact carries the
`[SMOKE TEST (local_smoke) -- not a reported result]` label (`idea.md` §0.1.1).

Scope covers **all six** curriculum tasks, all of them buildable. Three decisions were resolved
on 2026-09-22 and are wired through as config values here (see `docs/decision_register.md` for
each entry, including the attribution note on how the decisions reached the register):

| Decision | Resolution |
|---|---|
| D-006 B1 / B2 | `instr` → `databricks/databricks-dolly-15k`; `longform` → `Pavithree/eli5` |
| D-015 (a) / (b) / (c) | `dialog` → `dialog_last_turn`; `domain` → `finqa_pre_table_post` with `table_format: markdown`; `longform` target → `list_target_pick: highest_score` |
| D-016 | `data.sampling: shuffled_buffer` for every tier |

No `OPEN` decision blocks any of the six tasks. The **mechanisms** stay as designed: every one
of these values is named in `configs/`, none has a code default, and an unset value still fails
loudly — resolution supplied the values, it did not remove the guards.

## Problem statement

`local_smoke` cannot currently touch real data or a real backbone:

1. `configs/experiments/base.yaml` → `run_mode_overrides.local_smoke.data.provider: synthetic`,
   so the only provider ever exercised outside `pytest` is the offline fixture.
2. `Qwen/Qwen2.5-1.5B-Instruct` has never been downloaded. `BackboneSpec` carries no
   `revision`, and `_load_pretrained` passes no `cache_dir` / `local_files_only`, so there is
   no pinned, offline-reusable snapshot — although D-001 explicitly requires the checkpoint
   tag to be pinned once chosen.
3. `data.volumes` is read at `src/amaun/experiment.py:123` but **is defined in no config
   file**. The code substitutes the literals `1000`, `200`, `"validation"` — three
   experiment-affecting defaults in code, which `docs/system_specification.md` M0 forbids.
4. The HF provider always streams (`streaming=True`), and streaming **never populates**
   `data.cache_dir`. There is no offline-reusable dataset cache, so every run is as
   network-dependent as the first.
5. **`instr` (dolly-15k) has exactly one upstream split.** `prepare_data` draws the training
   pool from `split="train"` and the TEST pool from `volumes.test_split`. With only `train`
   available and no offset mechanism, both calls return the *identical* first *N* rows →
   TEST ⊂ TRAIN. `huggingface_pool` has no `skip`/offset parameter, and HF streaming does not
   accept slice syntax (`split="train[12000:]"` raises `ValueError: Bad split`). This is a
   leakage hazard against `docs/master_prompt.md` constraint 13 and invariant `I2.1`.
6. **Four of the six configured rows do not match the real upstream schema** (verified
   against the HF API on 2026-09-22; see `design.md` §1):
   - `sum` — `hf_revision: "3.0.0"` is a *config name, not a git ref*; the repo has only
     `main` and `script` branches, so revision resolution fails.
   - `instr` — configured `input_fields: [instruction, input]`, `target_field: output`;
     dolly-15k actually provides `instruction`, `context`, `response`, `category`.
   - `longform` — configured `input_fields: [question]`, `target_field: answer`;
     `Pavithree/eli5` actually provides `title`, `selftext`, `document`, `subreddit`, `url`,
     `answers{a_id,score,text}`, `q_id`, `*_urls`.
   - `dialog` — `li2017dailydialog/daily_dialog` is a **loading-script-only** repo
     (`daily_dialog.py`, no data files) and `datasets==5.0.1` (installed) removed script
     execution. Its schema is `dialog`/`act`/`emotion`, not `history`/`response`.
   - `domain` — `dreamerdeo/finqa` exposes `pre_text`/`post_text`/`table`/`question`/`answer`;
     there is **no `context` field**, which `input_fields: [context, question]` requires.
7. Only `sum`, `qa`, `dialog`, `domain` carry any `hf_revision`, and only `sum`'s is set (and
   it is wrong). "Pinned" in this repo means *a specific revision*, so five rows are
   effectively unpinned-in-practice.

## Goals

- G1 Download and locally cache `Qwen/Qwen2.5-1.5B-Instruct` at a **pinned commit**, and
  prove a subsequent `local_smoke` load succeeds with the network disabled.
- G2 Switch `run_mode_overrides.local_smoke.data.provider` to `huggingface`.
- G3 Define `data.volumes` and the sampling policy in `configs/`, removing the `1000` /
  `200` / `"validation"` literals from `src/amaun/experiment.py`.
- G4 Give the HF provider a **bounded, offline-reusable on-disk snapshot** under
  `data.cache_dir`, so run 2 needs no network and yields identical splits.
- G5 **Carve a disjoint TEST pool from a single-split corpus** (`instr`) by an explicit,
  config-declared row offset, and prove by regression test that the unfixed path leaks.
- G6 Pin every dataset to a **40-hex commit SHA** and correct the four schema mismatches.
- G7 Make row→example adaptation an explicit, config-named, **no-default** mechanism, and wire
  the D-015 resolutions through it so all six tasks load.
- G8 Make `prepare_data` correct with **any subset** of the curriculum available: blocked
  tasks are reason-coded and reported; available tasks produce full four-way splits and pass
  the leakage audit. With every decision resolved the expected subset is all six, but the
  skip-and-report path must remain exercised and tested — it is the guard that stops a future
  unpinned or unadapted task from being silently guessed at.
- G9 Close acceptance criterion `I7` (`scripts/run_smoke.py --run-mode local_smoke`).

## Non-goals

- **NG1 Re-opening or re-deciding D-006, D-015, or D-016.** The dataset identities, the
  row-adaptation rules, and the sampling policy are taken as given by the recorded decisions.
  No alternative is proposed, no *third* dataset is introduced, and the rejected options are
  kept implemented-but-unselected rather than deleted.
- **NG2 Removing the no-default guards.** Resolution supplies values in `configs/`; it does not
  license a code default for `row_adapter`, `list_target_pick`, `data.sampling`,
  `data.shuffle_buffer_size`, or `data.volumes`.
- NG3 Changing `dummy`-tier behaviour. `dummy` stays `synthetic`, offline, CPU, 6 tasks.
- NG4 Any `actual`-tier download, run, or volume choice. `data.volumes` for `actual` stays
  unresolved (`idea.md` §0.2 gives ranges, not values).
- NG5 Training, fusion, manager, or evaluation logic. The feature ends at `PreparedData`
  plus a cached backbone.
- NG6 Reporting anything. No metric, loss, or score from this feature is admissible evidence
  for H1–H6.
- NG7 Fixing the pre-existing `requirements.txt` drift (`datasets==4.5.1` pinned, `5.0.1`
  installed). Named in `design.md` §8 as an out-of-scope defect.

## Relation to the hypotheses

This feature serves **no hypothesis directly** and must not be described as if it did. It is
the mechanical precondition for all of them: `docs/experiment_protocol.md` §1 forbids a
reportable run until the `local_smoke` pipeline has completed end to end, and
`docs/master_prompt.md` §8 gates "done" on `docs/acceptance_criteria.md`, of which `I7` is
currently `[~] blocked`.

The feature's own claim is binary and falsifiable by execution: *the real-data path either
completes with clean, provably disjoint four-way splits for every available task, or it does
not.* AC-01…AC-18 are that experiment. G5 in particular is falsifiable in the strong sense:
T-08's regression test fails against the pre-fix code.

## Functional requirements

### FR-1 Backbone snapshot (G1)

- FR-1.1 `BackboneSpec` gains `revision: str | None = None`. `BACKBONE_REGISTRY[LOCAL_SMOKE]`
  pins the exact commit downloaded. `DUMMY` stays `None` (built in-process); `ACTUAL` stays
  `None` with a comment citing D-001 (it cannot be resolved from this host).
- FR-1.2 New config keys `model.cache_dir`, `model.local_files_only`, and an optional
  `model.revision` override (in `configs/model/qwen.yaml`) are threaded into
  `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`.
- FR-1.3 New `scripts/fetch_backbone.py --config … --run-mode …` downloads the run-mode's
  checkpoint into `model.cache_dir` via `huggingface_hub.snapshot_download`, **refuses
  `--run-mode dummy`** (no `model_id`), and writes `backbone_cache.json` under
  `results/<run_mode>/<experiment_id>/` recording `repo_id`, the resolved **40-hex commit
  sha**, the file list, total bytes, and the result of an offline re-load verification.
- FR-1.4 If `BackboneSpec.revision` is set and the resolved sha differs, the script raises —
  a silently different checkpoint changes every adapter.

### FR-2 Dataset pinning and schema correction (G6)

- FR-2.1 Every task in `configs/datasets/curriculum.yaml` carries an `hf_revision` that is
  either a 40-hex commit sha or the literal `refs/convert/parquet` pinned **together with**
  that ref's resolved sha in a sibling `hf_revision_resolved` field.
- FR-2.2 `sum`: `hf_revision` becomes the commit sha (not `"3.0.0"`); `hf_config` stays
  `"3.0.0"`.
- FR-2.3 `instr`: `hf_id: databricks/databricks-dolly-15k`, `pinned: true`,
  `input_fields: [instruction, context]`, `target_field: response`, license
  `CC BY-SA 3.0`, and a `prompt_template` that renders correctly when `context` is the empty
  string (dolly's `context` is empty for a large share of rows — an empty value is **not** a
  missing field and must not raise).
- FR-2.4 `longform`: `hf_id: Pavithree/eli5`, `pinned: true`, `input_fields: [title]`,
  `target_field: answers`, `prompt_template` referring to `{title}`. Upstream column names
  are used verbatim in both `input_fields` and the template — exactly as `sum`/`qa` already
  do — so no renaming indirection is introduced.
- FR-2.5 `longform`'s license is recorded as unspecified-on-the-mirror with the Reddit-derived
  redistribution caveat retained from D-006 B2. `license_notes` must say so explicitly; this
  must not silently become "MIT" or "unknown".
- FR-2.6 `dialog`: `hf_revision: refs/convert/parquet` (the `main` branch contains no data
  files, only a loading script `datasets` 5.x cannot execute), with the conversion commit
  recorded. The dataset *identity* pinned by D-006 is unchanged.
- FR-2.7 `upstream_test_source` is updated per task to state which upstream split becomes
  TEST, including `instr`'s offset-carved window.
- FR-2.8 A test asserts every task's `hf_revision` matches `^[0-9a-f]{40}$|^refs/convert/parquet$`,
  so "pinned but floating on `main`" cannot recur.

### FR-3 Provider switch, volumes, and sampling (G2, G3)

- FR-3.1 `run_mode_overrides.local_smoke.data.provider: huggingface`.
- FR-3.2 `run_mode_overrides.dummy.data.provider` stays `synthetic`; base-level
  `data.provider` stays `synthetic` so `pytest` remains offline.
- FR-3.3 `data.volumes` is `null` at base level (unresolved for `actual`) and defined
  per-task in the `local_smoke` overlay with required keys `train_pool`, `test`,
  `test_split`, plus `test_offset` where FR-4 requires it.
- FR-3.4 Under `provider: huggingface`, a missing
  `data.volumes.<task>.{train_pool,test,test_split}` raises `ConfigError` naming the task and
  key. **No code default.** The `1000` / `200` / `"validation"` literals are removed.
- FR-3.5 `local_smoke` volumes must be large enough that all four splits reach
  `data.smoke_max_examples_per_split` after ratio partitioning and subsampling
  (`train_pool ≥ cap / min(split_ratios)`, `test ≥ cap`).
- FR-3.6 New key `data.sampling` (`head_of_stream | shuffled_buffer`) with **no default** in
  code. Set to **`shuffled_buffer`** per D-016, together with `data.shuffle_buffer_size`
  (also no default). Both are `ConfigError` when absent under `provider: huggingface`.
- FR-3.7 `data.shuffle_buffer_size` must satisfy `≥ max(train_pool, test)` for every task, or
  the shuffle window is smaller than the sample and the shuffle is a no-op; violation raises
  `ConfigError`. The value is recorded in `config_used.json` and in every snapshot manifest,
  because it determines how near-random the sample is (D-016 condition 1).
- FR-3.8 The shuffle seed is derived deterministically from `data.seed` and is **identical
  across both loads of the same task**, so that FR-4's `skip`-based TEST carving remains a
  disjointness guarantee (D-016 condition 2). A per-call random seed would silently break it.

### FR-4 Disjoint TEST carving from a single-split corpus (G5)

- FR-4.1 `huggingface_pool` gains `skip: int = 0`, applied via the streaming iterable's
  native `.skip(n)` before any row is consumed.
- FR-4.2 New per-task config key `data.volumes.<task>.test_offset` (rows). It is
  **required** whenever `volumes.test_split` equals the split the training pool is drawn from
  (i.e. `"train"`); absent in that case → `ConfigError` citing this requirement and
  `I2.1`.
- FR-4.3 The provider reports `rows_consumed` for every load (rows read from the stream,
  which differs from examples emitted when a row adapter expands rows).
- FR-4.4 `prepare_data` raises `DataError` if `test_offset < rows_consumed_by_the_training_pool`
  — the windows would overlap. This is checked against the **measured** row count, not the
  configured example cap.
- FR-4.5 `instr` is configured with `test_split: train` and a `test_offset` satisfying
  FR-4.4.
- FR-4.6 A regression test asserts (a) offset-carved pools are disjoint by `example_id`, and
  (b) with `test_offset = 0` — the pre-fix behaviour — `build_splits` raises `LeakageError`.
  (b) must be verified to genuinely fire before the fix is trusted: it is what proves the
  existing `splits.py` / `leakage.py` machinery actually catches this failure mode rather
  than being assumed to.
- FR-4.7 The same test asserts that two independently constructed streams for one task, under
  `sampling: shuffled_buffer`, yield **identical row order** — without which `skip` no longer
  carves disjoint windows (FR-3.8).

### FR-5 Offline dataset snapshots (G4)

- FR-5.1 A snapshot of the *capped* sample is written under `data.cache_dir` as
  `<task_id>/<upstream_split>-<key>.jsonl` with a sibling `.manifest.json`.
- FR-5.2 The snapshot key hashes **every** content-determining input: `hf_id`, resolved
  `hf_revision`, `hf_config`, upstream split, requested size, `skip`, `data.sampling`
  (+ buffer size), the row-adapter name and options, and the adapter's required upstream
  field list. Any change yields a new key; a stale snapshot is never silently reused.
- FR-5.3 Snapshots store **raw upstream rows restricted to the required fields**, not
  rendered prompts, so a prompt-template edit needs no re-download (it does change the
  resulting `Example`s, and that is visible through `split_provenance`).
- FR-5.4 When a matching snapshot exists it is read from disk and **no network call is made**.
  Each load records `source: "download" | "local"`.
- FR-5.5 Snapshotting is governed by `data.offline_snapshots: true|false`; when `false`,
  behaviour is today's pure streaming.
- FR-5.6 Downloads stay bounded: stream and stop at the configured size. Non-streaming
  `load_dataset` over a full corpus is never called (brief §4).

### FR-6 Row adaptation (G7)

- FR-6.1 New module `src/amaun/data/row_adapters.py` (layer M2) with a named registry. Each
  adapter declares `required_row_fields` and maps one upstream row to
  `list[(fields: dict[str, str], target: str)]` — a list, because some candidate rules yield
  several examples per row.
- FR-6.2 Per-task config keys `row_adapter` (string, **no default**) and
  `row_adapter_options` (mapping).
- FR-6.3 `row_adapter: passthrough` reproduces today's behaviour exactly (select
  `input_fields` from the row, read `target_field`, unwrap `{"text": [...]}` and plain-list
  targets). `sum`, `qa`, `instr`, `longform` are declared `passthrough`; this documents
  existing behaviour rather than introducing a choice.
- FR-6.4 `passthrough` takes an explicit option `list_target_pick` (`first | highest_score`)
  governing how a list-valued target is reduced. Today's code silently takes `first`; making
  it an explicit config value removes a hidden choice. `longform` is configured
  **`highest_score`** per D-015 (c) — ELI5's `answers` carries a parallel `score` list, so the
  target is `text[argmax(score)]`, not `text[0]`.
- FR-6.5 `dialog` is declared `row_adapter: dialog_last_turn` and `domain`
  `row_adapter: finqa_pre_table_post` with `row_adapter_options.table_format: markdown`, per
  D-015 (a)/(b). The rejected candidates (`dialog_all_prefixes`, `finqa_text_only`) stay
  **implemented but unselected**, so the comparison remains reachable by config without a code
  change.
- FR-6.5a `finqa_pre_table_post` composes `context` = `pre_text` + markdown-serialised `table`
  + `post_text`. Because FinQA contexts are long against `max_input_tokens: 768`, the provider
  records the fraction of examples whose rendered prompt exceeds that budget in the snapshot
  manifest, so truncation is measured rather than assumed (D-015 consequence note).
- FR-6.6 `row_adapter: null`, or a name absent from the registry, raises
  `UnresolvedRowAdapterError` (new, subclass of `DataError`) naming the task and citing D-015.
- FR-6.7 A row missing a required upstream field raises `DataError` and **aborts** — never
  skipped, because a partially-adapted corpus silently changes the task. A field that is
  *present but empty* (dolly `context`) is valid input, not an error.
- FR-6.8 `TaskSpec.input_fields` keeps meaning *prompt-template slots*; upstream row fields
  are the row adapter's concern. `render_prompt` is unchanged.

### FR-7 Partial-curriculum robustness (G8)

- FR-7.1 `PreparedData` gains `blocked: dict[task_id, reason_code]` with codes
  `unpinned_dataset` (D-006) and `unresolved_row_adapter` (D-015). `unpinned` is retained as
  the `unpinned_dataset` subset so existing callers keep working.
- FR-7.2 Only those two conditions are skippable. Any other failure — network error, 404,
  revision mismatch, schema mismatch, empty result, offset violation — **propagates**. Silent
  degradation on a transient error would be indistinguishable from a real block.
- FR-7.3 `dataset_report.json` gains `available_tasks`, `blocked_tasks` (code + human reason
  + decision id), `volumes`, `sampling`, and a `snapshots` list of
  `(task, upstream_split, key, source, rows_consumed, n_examples, exhausted)`.
- FR-7.4 `scripts/prepare_data.py` prints the blocked table and the smoke label, and exits
  non-zero only if the leakage audit is unclean. A blocked task is **not** a failure exit.
- FR-7.5 Zero available tasks continues to raise `PreconditionError`, now naming both D-006
  and D-015.

## Expected behaviour

After BUILD completes, with D-006, D-015, and D-016 all `RESOLVED`, **all six tasks produce
four-way splits** on real data:

| task | dataset | row adapter | TEST source |
|---|---|---|---|
| `sum` | `abisee/cnn_dailymail` @ sha, config `3.0.0` | `passthrough` | upstream `validation` |
| `qa` | `rajpurkar/squad` @ sha, config `plain_text` | `passthrough` | upstream `validation` |
| `instr` | `databricks/databricks-dolly-15k` @ sha | `passthrough` | offset window of upstream `train` (single split) |
| `dialog` | `li2017dailydialog/daily_dialog` @ `refs/convert/parquet` | `dialog_last_turn` | upstream `validation` |
| `longform` | `Pavithree/eli5` @ sha | `passthrough`, `list_target_pick: highest_score` | upstream `validation` |
| `domain` | `dreamerdeo/finqa` @ sha | `finqa_pre_table_post`, `table_format: markdown` | upstream `test` |

So `blocked_tasks` is expected to be **empty** and `available_tasks` to be all six. The
skip-and-report path (FR-7) is nevertheless required and tested: it is the guard that stops a
future re-pin, schema change, or added task from being silently guessed at, and
`tests/test_prepare_data_blocked.py` exercises it with synthetic task specs rather than by
leaving a real task unconfigured.

## Edge cases

- EC-1 `--run-mode dummy` + `fetch_backbone.py` → clear `ConfigError`, no network.
- EC-2 `--run-mode actual` + `fetch_backbone.py` on this host → `resolve()` still raises
  `QuantizationUnavailableError` (D-008). The script must surface that, never downgrade.
- EC-3 Snapshot present but manifest mismatched → new key, re-download; the old file is left
  untouched (append-only cache).
- EC-4 `HF_HUB_OFFLINE=1` with no snapshot → the underlying HF error propagates with a hint
  to run `scripts/prepare_data.py` online first. It must **not** be masked as "blocked".
- EC-5 `volumes.test_split` names a non-existent upstream split → `DataError` naming the task
  and the available splits.
- EC-6 A row adapter emitting several examples per row: the provider counts **examples**
  against the size cap and **rows** for `rows_consumed`/offset accounting, and stops
  deterministically mid-row-expansion.
- EC-7 Corpus exhausted before the cap → use what was collected; the manifest records
  `exhausted: true` and the true count. Zero examples remains a `DataError`. This is a live
  risk for `instr`: dolly has 15,011 rows total, so `train_pool + test_offset + test` must fit.
- EC-8 `data.smoke_max_examples_per_split` exceeding a split's population → the split keeps
  its natural size; AC-07 is stated over `min(cap, natural)`.
- EC-9 `instr` rows whose `context` is `""` → valid; rendered template contains an empty
  context section. A test covers this explicitly.
- EC-10 `longform` rows whose `answers.text` is empty → `DataError` (zero-length target would
  train on nothing); the row is not silently dropped.
- EC-11 `dialog`'s `refs/convert/parquet` commit changes upstream → the pinned sha no longer
  resolves, and the load fails loudly rather than silently fetching different data.
- EC-12 A `dialog` row with fewer than 2 utterances → no `(history, response)` pair exists;
  `dialog_last_turn` emits nothing for that row and the provider continues to the next one.
  The manifest counts such rows, so an unexpectedly high count is visible rather than silent.
- EC-13 A `domain` row whose `table` is empty → `finqa_pre_table_post` renders the text parts
  only; this is valid, not an error, and is counted in the manifest.
- EC-14 `data.shuffle_buffer_size < max(train_pool, test)` → `ConfigError` (FR-3.7): a shuffle
  window narrower than the sample is indistinguishable from `head_of_stream` and would
  misreport the sampling policy.

## Constraints

- C-1 `run_mode` stays required with no default (`idea.md` §0.1).
- C-2 No artefact leaves `results/local_smoke/<experiment_id>/`; `assert_write_allowed` is
  not bypassed (`idea.md` §0.1.1).
- C-3 No experiment-affecting numeric or string literal may be added to `src/**` or
  `scripts/**`. Every new value lives in `configs/` (M0).
- C-4 Dependency direction (`docs/architecture.md`): new modules are M2 (`data/`) and may
  import only `config`, `errors`, `utils`, and `data`. Nothing in `manager` gains a path to
  `evaluation`.
- C-5 Split isolation is untouched: SUPPORT still comes from the training pool, TEST still
  from a separate upstream source or a disjoint offset window (`I2.2`), assignment still a
  pure function of `(task_id, example_hash, seed)` (`I2.3`).
- C-6 `pytest` must remain fully offline. Every new test runs on the `dummy` tier with
  monkeypatched loaders; no test may hit the network.
- C-7 The values resolved in D-006 / D-015 / D-016 are the only ones configured. Rejected
  options stay implemented and unselected; no value acquires a code default.
- C-8 Licensing stays visible: `dialog` remains CC BY-NC-SA 4.0 non-commercial, `instr` is
  CC BY-SA 3.0 (share-alike propagates to derivatives), `longform`'s mirror carries no
  declared license over Reddit-derived content. Snapshot manifests record the task's
  `license` string so a release audit can be mechanical.
- C-9 Download budget: ~3.1 GB backbone plus a few MB of capped dataset samples. No
  full-corpus download.

## Dependencies

- Installed and verified 2026-09-22: `datasets 5.0.1`, `huggingface-hub 1.32.0`,
  `transformers 5.17.0`, `peft 0.21.0`, `torch 2.14.0`; `.venv/bin/hf` and
  `.venv/bin/huggingface-cli` present. **No new third-party dependency.**
- Network access for the first run only; ~3.1 GB free disk for the backbone.
- Baseline measured 2026-09-22: `.venv/bin/python -m pytest -q` → **280 passed**.
- D-006, D-015, D-016 all `RESOLVED` 2026-09-22 — nothing in this feature is decision-blocked.
  D-002 / D-005 / D-011 / D-012 remain `OPEN` but gate only `actual` runs, not this one.

## Acceptance criteria

Each criterion names the command that verifies it. `P` abbreviates `.venv/bin/python`;
`CFG` abbreviates `configs/experiments/main_bootstrap.yaml`. Paths are repo-relative.

- **AC-01** `P scripts/fetch_backbone.py --config CFG --run-mode local_smoke` exits 0 and
  `data/models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<sha>/model.safetensors` exists
  with size > 1 GB.
- **AC-02** `results/local_smoke/main_bootstrap/backbone_cache.json` has
  `repo_id == "Qwen/Qwen2.5-1.5B-Instruct"`, a `revision` matching `^[0-9a-f]{40}$` equal to
  `BACKBONE_REGISTRY[RunMode.LOCAL_SMOKE].revision`, and `verified_offline_load == true`.
- **AC-03** With the network disabled (`HF_HUB_OFFLINE=1`), the one-liner in `tasks.md` T-04
  loads the `local_smoke` backbone and reports `frozen_parameter_count > 0` and
  `trainable == 0`.
- **AC-04** `P scripts/fetch_backbone.py --config CFG --run-mode dummy` exits non-zero with
  `ConfigError`, making no network call.
- **AC-05** `P -m pytest tests/test_data_registry.py -k revision_is_pinned` passes: every
  task's `hf_revision` matches `^[0-9a-f]{40}$|^refs/convert/parquet$`.
- **AC-06** `P -m pytest tests/test_data_registry.py -k unpinned` **now asserts the empty
  set** of unpinned tasks (D-006 resolved), and every task passes `require_pinned()`.
- **AC-07** `P scripts/prepare_data.py --config CFG --run-mode local_smoke` exits 0, and
  `results/local_smoke/main_bootstrap/dataset_report.json` has `provider == "huggingface"`,
  `leakage_audit.clean == true`, and for every task in `available_tasks` all four
  `split_counts` equal `min(data.smoke_max_examples_per_split, natural population)`.
- **AC-08** In the same report, `unpinned_tasks == []` and `blocked_tasks == {}`.
- **AC-09** In the same report, `available_tasks` is exactly
  `["sum", "qa", "instr", "dialog", "longform", "domain"]` (curriculum order), every task has a
  `split_provenance.hf_revision` matching `^[0-9a-f]{40}$|^refs/convert/parquet$`, and
  `sampling == "shuffled_buffer"`.
- **AC-10** After AC-07, `data/raw` is non-empty: for every `(task, upstream_split)` load
  there is a `data/raw/<task>/<split>-<key>.jsonl` with `n_rows > 0` and a `.manifest.json`
  whose `hf_id`, `hf_revision`, `hf_config`, `size`, `skip`, `sampling`, `row_adapter`, and
  `license` fields are all populated.
- **AC-11** Re-running AC-07's command with `HF_HUB_OFFLINE=1` exits 0, produces **identical**
  `split_counts` and `split_provenance`, and every entry of `snapshots` has
  `source == "local"`.
- **AC-12** `instr`'s TEST pool is disjoint from its training pool in the real artefact:
  in `dataset_report.json`, `leakage_audit.clean == true` **and**
  `P -m pytest tests/test_single_split_carving.py` passes, which asserts (a) offset-carved
  pools share no `example_id`, (b) with `test_offset = 0` `build_splits` raises
  `LeakageError` — the pre-fix behaviour — and (c) under `sampling: shuffled_buffer` two
  independently constructed streams for one task yield identical row order (FR-3.8/FR-4.7).
- **AC-13** `P -m pytest tests/test_experiment_preconditions.py -k volumes_required` passes:
  removing `data.volumes` (or any of `train_pool` / `test` / `test_split`, or `test_offset`
  when `test_split == "train"`, or `data.sampling`, or `data.shuffle_buffer_size`, or setting
  the buffer below `max(train_pool, test)`) raises `ConfigError` naming the offending key.
- **AC-14** `P -m pytest tests/test_row_adapters.py` passes: `passthrough` is byte-identical
  to the pre-change provider output on a fixed synthetic row set; `list_target_pick` is
  honoured for both values, and `highest_score` provably selects `text[argmax(score)]` on a row
  whose maximum is not at index 0; a dolly-style row with `context == ""` renders without error;
  an ELI5-style row with an empty `answers.text` raises `DataError`; `dialog_last_turn` splits a
  fixed utterance list into `history`/`response` and emits nothing for a single-utterance row;
  `finqa_pre_table_post` renders a markdown table and tolerates an empty one; every rejected
  D-015 candidate is still registered with declared `required_row_fields`; `row_adapter: null`
  and an unknown name each raise `UnresolvedRowAdapterError` citing D-015.
- **AC-15** `P -m pytest tests/test_data_snapshots.py` passes: the key changes when any one
  of `{hf_id, hf_revision, hf_config, split, size, skip, sampling, row_adapter,
  row_adapter_options}` changes; a matching snapshot is served with the loader monkeypatched
  to raise (proving no network call); round-trip preserves row content exactly.
- **AC-16** `P -m pytest tests/test_prepare_data_blocked.py` passes: on the `dummy` tier with
  `provider: huggingface` and the HF loader monkeypatched to raise, a config in which one task's
  `row_adapter` has been blanked and another's `hf_id` set to `null` reports exactly
  `{unresolved_row_adapter, unpinned_dataset}` **without invoking the loader**, while a loader
  `RuntimeError` for an otherwise-available task **propagates** (FR-7.2). The blanking happens
  in the test fixture, not in `configs/` — the shipped curriculum has all six tasks configured.
- **AC-17** `P scripts/run_smoke.py --run-mode local_smoke` reports all checks `PASS` and
  `results/local_smoke/main_bootstrap/smoke_report.json` has `all_passed == true`. This is
  `I7` in `docs/acceptance_criteria.md`.
- **AC-18** Regression: `P -m pytest -q` reports ≥ 280 passed, 0 failed; and
  `P scripts/run_smoke.py --run-mode dummy` reports `21/21 checks passed` — proving `dummy`
  stayed synthetic and offline.

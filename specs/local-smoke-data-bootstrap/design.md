# local-smoke-data-bootstrap — Design

## 1. Ground truth gathered before designing

All of the following was verified on 2026-09-22 against the live HF API (read-only calls, no
downloads) and by reading the installed environment. It is recorded because the existing
`configs/datasets/curriculum.yaml` was written against an *assumed* schema and is wrong in
four places.

### 1.1 Environment

| Package | Installed | `requirements.txt` |
|---|---|---|
| `datasets` | **5.0.1** | `4.5.1` (drift — see §8) |
| `huggingface-hub` | 1.32.0 | not pinned |
| `transformers` | 5.17.0 | 5.17.0 |
| `torch` | 2.14.0 | 2.14.0 |

`datasets` ≥ 4 **removed dataset-loading-script execution**. Any HF dataset repo that ships
only a `*.py` loader is unloadable by `load_dataset` in this environment.

`.venv/bin/hf` and `.venv/bin/huggingface-cli` exist; `hf_transfer` is not installed
(downloads will be plain HTTPS, which is fine for 3.1 GB).

Baseline: `.venv/bin/python -m pytest -q` → **280 passed** (0 failed, 0 skipped).

### 1.2 Backbone

`Qwen/Qwen2.5-1.5B-Instruct`, `main` sha `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
(observed 2026-09-22). Files: `config.json`, `generation_config.json`, `model.safetensors`
(single shard), `merges.txt`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
`LICENSE`, `README.md`, `.gitattributes`. **BUILD must re-resolve the sha at download time
and record what it actually got** — the value above is a cross-check, not an authority.

### 1.3 Datasets — real schemas and revisions

| task | `hf_id` | `main` sha (observed) | `hf_config` | upstream splits | real columns |
|---|---|---|---|---|---|
| `sum` | `abisee/cnn_dailymail` | `96df5e686bee6baa90b8bee7c28b81fa3fa6223d` | `3.0.0` | train / validation / test | `article`, `highlights`, `id` |
| `qa` | `rajpurkar/squad` | `7b6d24c440a36b6815f21b70d25016731768db1f` | `plain_text` | train / validation | `id`, `title`, `context`, `question`, `answers` |
| `instr` | `databricks/databricks-dolly-15k` | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | `default` | **train only, 15 011 rows** | `instruction`, `context`, `response`, `category` |
| `dialog` | `li2017dailydialog/daily_dialog` | `ffde34acefcd956529c39c4fc78d993b5b7f7520` (script only) | `default` | train / validation / test | `dialog` (list[str]), `act`, `emotion` |
| `longform` | `Pavithree/eli5` | `15e4d1933a321088880215257aa25b621a091335` | `default` | train 216 147 / validation 3 020 / test 10 000 | `q_id`, `title`, `selftext`, `document`, `subreddit`, `url`, `answers{a_id,score,text}`, `*_urls` |
| `domain` | `dreamerdeo/finqa` | `ebf0fda6e42b597b3b301baeba8da9f97e9aef03` | `default` | train / validation / test | `id`, `pre_text`, `post_text`, `question`, `answer`, `gold_evidence`, `table` |

Five consequential facts:

1. **`abisee/cnn_dailymail` has no git ref named `3.0.0`** — only branches `main` and
   `script`. The current `hf_revision: "3.0.0"` confuses the *config name* with a revision and
   cannot resolve.
2. **`li2017dailydialog/daily_dialog` ships only `daily_dialog.py`.** With `datasets` 5.0.1
   `load_dataset` fails. Its automatic parquet conversion exists at
   `refs/convert/parquet`, sha `469b7118f52336674f1d78693797e9fcc7e47d2a`, with
   `default/{train,validation,test}/0000.parquet`. Loading from that ref is the *same
   dataset*, mechanically converted — it preserves the identity D-006 pinned. Substituting a
   different repo (e.g. `roskoN/dailydialog`, also script-only) would *not*, and is therefore
   not done.
3. **`databricks/databricks-dolly-15k` has exactly one split.** This is the leakage hazard
   (§4).
4. **`Pavithree/eli5` carries no license tag.** Content is Reddit-derived; D-006 B2 already
   flags redistribution restrictions. Recorded, not silently normalised.
5. **`dreamerdeo/finqa` has no `context` column**; `dialog` has no `history`/`response`. Both
   need a row→example rule that does not exist in the codebase (§5).

### 1.4 Relevant existing code

- `src/amaun/experiment.py::prepare_data` — the single place splits are built. Under
  `provider == "huggingface"` it calls `task.require_pinned()`, appends to `unpinned` on
  `UnpinnedDatasetError` and `continue`s, then calls `huggingface_pool` twice (train pool +
  test pool) with `int(volumes.get("train_pool", 1000))`,
  `int(volumes.get("test", 200))`, `volumes.get("test_split", "validation")`. Those three
  literals are the M0 violation.
- `src/amaun/data/providers.py::huggingface_pool` — `spec.require_pinned()`, then
  `load_dataset(..., streaming=True)`, then a row loop that selects `spec.input_fields`,
  reads `spec.target_field`, unwraps `{"text": [...]}` and plain lists by taking element 0,
  and `render_prompt`s. Raises `DataError` on a missing field or a zero-length result. No
  `skip`, no cache write.
- `src/amaun/data/splits.py` — `build_splits` partitions the training pool 70/15/15 into
  TRAIN/VALIDATION/SUPPORT by `sha256(task_id|example_id|seed) mod 10000`, attaches the test
  pool verbatim, then calls `assert_disjoint()` which compares all six split pairs by
  content hash and raises `LeakageError`. **This is what will catch the dolly single-split
  leak** — verified by reading the code path, and asserted by T-08.
- `src/amaun/data/leakage.py::audit` — whole-corpus cross-task/cross-split audit; any
  TEST↔{TRAIN,VALIDATION,SUPPORT} overlap is `critical` and raises.
- `src/amaun/models/backbone.py::_load_pretrained` — `from_pretrained(spec.model_id, dtype,
  attn_implementation)`; no `cache_dir`, `revision`, or `local_files_only`.
- `src/amaun/config.py` — `BACKBONE_REGISTRY` is the only checkpoint table;
  `ResolvedConfig.get()` is mandatory-by-default (omitting `default` raises `ConfigError`),
  which is the mechanism FR-3.4 / FR-4.2 use.
- `src/amaun/utils/paths.py` — the only path constructor; `DATA_ROOT = Path("data")` already
  exists but is unused by the data layer.
- `scripts/prepare_data.py` — thin wrapper: tracker → `prepare_data` → `dataset_report.json`
  → prints → exit code from `leakage_audit.clean`.

## 2. What is reused vs. modified vs. added

**Reused unchanged.** `splits.py` (ratios, hashing, `assert_disjoint`, `subsample`,
`SplitAccessGuard`), `leakage.py`, `preprocess.py`, `registry.py::load_curriculum` ordering,
`tracking.py`, `paths.py`, `render_prompt`, the whole `synthetic` provider, and every module
above M2.

**Modified (small, additive).**

| File | Change |
|---|---|
| `src/amaun/config.py` | `BackboneSpec.revision: str \| None = None`; pin `LOCAL_SMOKE`; honour `model.revision` override |
| `src/amaun/models/backbone.py` | thread `cache_dir`, `revision`, `local_files_only` into both `from_pretrained` calls |
| `src/amaun/data/registry.py` | `TaskSpec` gains `hf_revision_resolved`, `row_adapter`, `row_adapter_options`, `row_adapter_reason`; `require_pinned` unchanged |
| `src/amaun/data/providers.py` | `huggingface_pool` gains `skip`, `sampling`, `row_adapter`, snapshot integration; returns `(examples, LoadReport)` |
| `src/amaun/experiment.py` | volumes required from config; `skip`/offset wiring; offset-vs-`rows_consumed` check; `PreparedData.blocked` |
| `src/amaun/errors.py` | `UnresolvedRowAdapterError(DataError)` |
| `scripts/prepare_data.py` | richer report + blocked table |
| `configs/datasets/curriculum.yaml` | revisions, schemas, `row_adapter`, `data.volumes`, `data.sampling`, `data.offline_snapshots` |
| `configs/experiments/base.yaml` | `local_smoke.data.provider: huggingface` + local_smoke volumes/sampling |
| `configs/model/qwen.yaml` | `model.cache_dir`, `model.local_files_only`, `model.revision` |
| `.gitignore` | `data/models/**` |
| `tests/test_data_registry.py` | the unpinned-set assertion inverts (D-006 resolved) |

**Added.**

| File | Layer | Purpose |
|---|---|---|
| `src/amaun/data/row_adapters.py` | M2 | named row→example registry, no default |
| `src/amaun/data/snapshots.py` | M2 | content-keyed bounded on-disk cache |
| `scripts/fetch_backbone.py` | scripts | download + pin + offline verification |
| `tests/test_row_adapters.py` | — | FR-6 |
| `tests/test_data_snapshots.py` | — | FR-5 |
| `tests/test_single_split_carving.py` | — | FR-4, the leak regression |
| `tests/test_prepare_data_blocked.py` | — | FR-7 |

Dependency direction (`docs/architecture.md`): both new modules sit in `data/` and import
only `config`, `errors`, `utils`, `data`. Nothing acquires a new path from `manager` to
`evaluation`.

## 3. Proposed architecture

```
scripts/fetch_backbone.py
   └─ config.resolve() ─ BACKBONE_REGISTRY[run_mode] ─(model_id, revision)
        └─ huggingface_hub.snapshot_download(cache_dir=model.cache_dir)
             └─ backbone.load_backbone(local_files_only=True)   # offline verification
                  └─ results/<mode>/<id>/backbone_cache.json

scripts/prepare_data.py
   └─ experiment.prepare_data(cfg)
        for task in load_curriculum(...):
            require_pinned()                    → blocked[unpinned_dataset]      (D-006)
            row_adapters.resolve(task)          → blocked[unresolved_row_adapter](D-015)
            volumes = cfg["data"]["volumes"][task]      # required, no default
            pool, rep_pool  = providers.huggingface_pool(spec, volumes.train_pool,
                                  split="train", skip=0, ...)
            test, rep_test  = providers.huggingface_pool(spec, volumes.test,
                                  split=volumes.test_split,
                                  skip=volumes.test_offset or 0, ...)
            assert test_offset >= rep_pool.rows_consumed  if same split   (FR-4.4)
            build_splits(...) → assert_disjoint()                        (I2.1)
            subsample(smoke_max_examples_per_split)
        audit(all splits) → LeakageReport
        → dataset_report.json  (+ blocked table, snapshots, volumes, sampling)

providers.huggingface_pool
   ├─ snapshots.key(spec, split, size, skip, sampling, adapter)  → path
   ├─ if snapshot exists  → read rows from JSONL       (source="local", no network)
   └─ else stream+skip+take, write snapshot            (source="download")
        └─ per row: adapter.adapt(row) → [(fields, target)…] → render_prompt → Example
```

### 3.1 `row_adapters.py`

```
RowAdapter(Protocol):
    name: str
    def required_row_fields(spec: TaskSpec, options: Mapping) -> tuple[str, ...]
    def adapt(spec: TaskSpec, row: Mapping, options: Mapping) -> list[tuple[dict, str]]

ROW_ADAPTERS: dict[str, RowAdapter]          # "passthrough" + the four D-015 rules
def resolve(spec) -> RowAdapter              # raises UnresolvedRowAdapterError on None/unknown
```

Registered adapters:

| name | status | behaviour |
|---|---|---|
| `passthrough` | **selected** by `sum`, `qa`, `instr`, `longform` | today's logic verbatim; option `list_target_pick ∈ {first, highest_score}` reduces list/dict targets. `longform` uses `highest_score` (D-015 c) |
| `dialog_last_turn` | **selected** by `dialog` (D-015 a) | `history` = utterances[:-1] joined by the configured separator; `target` = utterances[-1]; a row with < 2 utterances emits nothing |
| `dialog_all_prefixes` | implemented, unselected | one example per prefix ≥ 1 turn; multiplies volume |
| `finqa_pre_table_post` | **selected** by `domain` (D-015 b), `table_format: markdown` | `context` = `pre_text` + serialised `table` + `post_text`; option `table_format ∈ {markdown, pipe_rows}` |
| `finqa_text_only` | implemented, unselected | `context` = `pre_text` + `post_text`, table dropped |

Rejected candidates are kept registered rather than deleted, so the alternative remains reachable
by a one-line config edit if the decision is ever revisited — and so `tests/test_row_adapters.py`
can keep asserting that the registry, not a hardcoded branch, is what selects behaviour.

`passthrough`'s `required_row_fields` is `spec.input_fields + (spec.target_field,)`, which is
why `sum`/`qa`/`instr`/`longform` need no bespoke code — their `input_fields` are literally
upstream column names. That is the existing convention (`{article}`, `{context}`,
`{question}`), and `longform` follows it with `input_fields: [title]` and `{title}` in the
template. **No rename/mapping indirection is added**: introducing one would mean two names
for every field and a second place for the prompt contract to drift.

### 3.2 `snapshots.py`

```
@dataclass(frozen=True) SnapshotSpec:
    hf_id, hf_revision, hf_config, split, size, skip, sampling,
    shuffle_buffer_size, row_adapter, row_adapter_options, required_fields
    def key(self) -> str          # sha256 over canonical JSON, first 16 hex

@dataclass SnapshotManifest:
    spec fields + task_id, license, rows_written, exhausted, created_utc,
    datasets_version, content_sha256

def path_for(cache_dir, task_id, spec) -> Path            # <cache>/<task>/<split>-<key>.jsonl
def read(path) -> list[dict] | None                        # None if absent/manifest mismatch
def write(path, rows, manifest) -> None                    # atomic: tmp + rename
```

Raw rows (restricted to `required_fields`) are stored, not rendered `Example`s, so a
prompt-template edit does not force a re-download. The cost is that the cache cannot prove
which prompts were produced — mitigated because `split_provenance` already records the
template-bearing `TaskSpec` fields and `config_used.json` carries the whole tree.

Alternative considered: store rendered `Example`s (simpler, one object type). Rejected — it
couples an expensive network artefact to a cheap local formatting decision, and would make
every prompt tweak a re-download.

Alternative considered: `datasets`' own `cache_dir` with `streaming=False`. Rejected — it
downloads the *entire* corpus (CNN/DailyMail ≈ 1.3 GB, ELI5 216 k rows), which brief §4
explicitly forbids, for a run that needs 32 examples per split.

## 4. The single-split leak (`instr`) — the load-bearing design decision

**Failure mode.** `prepare_data` draws the training pool from `split="train"` and the test
pool from `volumes.test_split`. dolly-15k has only `train`. With no offset, both calls stream
from row 0 and the test pool is a prefix of the training pool → TEST ⊆ TRAIN.

**Is it caught today?** Yes, loudly. `build_splits` calls `assert_disjoint()`, which compares
`hashes(TEST)` against `hashes(TRAIN|VALIDATION|SUPPORT)` by `example_id =
sha256(task_id|prompt|target)`. Identical rows produce identical ids, so it raises
`LeakageError`. `leakage.audit` would independently mark it `critical`. So the pre-fix state
**aborts the run** rather than silently leaking — good, and T-08 asserts exactly that, which
is what makes the fix trustworthy rather than assumed.

**Chosen mechanism — O1: explicit streaming offset.**
`huggingface_pool(..., skip: int = 0)` calls the iterable's native `.skip(n)` before
consuming. `data.volumes.<task>.test_offset` (rows) is **required** whenever
`volumes.test_split` equals the split the training pool comes from. `prepare_data` then
enforces `test_offset ≥ rows_consumed_by_the_training_pool`, checked against the
**measured** row count returned in the provider's `LoadReport` — not against the configured
example cap, because a row adapter may emit several examples per row and the two counts
differ.

Alternatives considered:

| Option | Why not |
|---|---|
| **O2** non-streaming load for "small" datasets + Arrow slicing / `train_test_split` | Needs a size threshold — a new numeric literal and a new code path that behaves differently per dataset. It also special-cases exactly the dataset it was invented for, so the *general* single-split case stays broken. |
| **O3** `split="train[12000:]"` slice syntax | Verified not supported under streaming (`ValueError: Bad split`), and non-streaming means O2. |
| **O4** deterministic hash-based partition of one stream into pool/test | Would make TEST a *partition of the training pool* rather than an independently sourced split, contradicting `SplitRatios`' documented contract and assumption A-05. Rejected on methodology, not convenience. |

O1 is uniform (works for any dataset, any split), keeps streaming-first (brief §4), adds one
optional parameter and one config key, and leaves `splits.py` untouched.

**Interaction with `sampling: shuffled_buffer` (D-016).** The offset is a disjointness
guarantee only if the two loads see the **same row order**. Pipeline order is therefore fixed as
`load_dataset(streaming=True) → .shuffle(seed, buffer_size) → .skip(n) → iterate`, and the
shuffle seed is derived deterministically from `data.seed` and the task id — never from a clock
or a per-call RNG. `data.shuffle_buffer_size ≥ max(train_pool, test)` is additionally required,
because a window narrower than the sample makes the shuffle a no-op and would misreport the
sampling policy. T-11 asserts identical ordering across two independently constructed streams;
`assert_disjoint` remains the independent check.

**Residual property.** Offset windows are disjoint by *row position*; exact-duplicate rows
across windows would still overlap by content. That is precisely what `assert_disjoint`
catches, and it raises rather than degrading — the correct outcome.

**Capacity check.** dolly has 15 011 rows, so `train_pool + test_offset + test` must fit
inside it, and EC-7's `exhausted` flag surfaces the case where it does not. The `local_smoke`
volumes must satisfy FR-3.5 (`train_pool ≥ cap / 0.15`) with room to spare; concrete numbers
belong in `configs/`, not here.

## 5. Row adaptation — resolved values and what they commit us to

Three tasks need a rule that changes what the adapter learns, so each was registered as a
decision rather than guessed. All three were resolved on 2026-09-22
(`docs/decision_register.md` D-015) and are now configured values:

| task | rule | what it commits the paper to |
|---|---|---|
| `dialog` | `dialog_last_turn` | One example per dialogue (~11 k train), not per turn prefix — so `dialog`'s volume stays comparable to the other tasks, and the context-length distribution is that of full dialogue histories |
| `domain` | `finqa_pre_table_post`, `table_format: markdown` | FinQA's original input (`pre_text` + markdown table + `post_text`). Contexts are long against `max_input_tokens: 768`, so the truncated fraction is **measured** into the snapshot manifest rather than assumed (FR-6.5a) |
| `longform` | `list_target_pick: highest_score` | The target is `answers.text[argmax(score)]`, **not** `text[0]`. Any earlier synthetic-data run is not comparable at the target level |

The **mechanism is unchanged by the resolution**: the rule is still named per task in config,
still has no code default, and an unset or unknown name still raises
`UnresolvedRowAdapterError`. Resolution supplied values; it did not remove the guard, and
`tests/test_prepare_data_blocked.py` keeps the skip-and-report path under test using fixture
specs rather than by leaving a shipped task unconfigured. Rejected candidates stay implemented
and unselected (§3.1).

Deliberately **not** done: hiding any of these in code as a "sensible default", or deleting the
alternatives so the comparison becomes unreachable.

## 6. Scientific validity

- **Which splits may this component read?** `prepare_data` is upstream of every guard — it
  *constructs* the splits. It hands out no handles; consumers still receive
  `SplitAccessGuard`s. No new reader of TEST is introduced, and `scripts/prepare_data.py`
  reports only *counts*, never content.
- **Could any value reach a decision from TEST?** No. Snapshot keys and manifests contain
  dataset coordinates and counts, never examples' content beyond the raw rows they cache, and
  the manager never reads `data/raw`. The one genuine new risk — `instr`'s TEST window
  overlapping its training pool — is closed by FR-4.4 and independently caught by
  `assert_disjoint` / `audit` (§4).
- **Budget comparability across arms.** Untouched: this feature changes no training budget.
  It does change *which examples* a `local_smoke` run sees, but `local_smoke` is
  non-reportable, so no arm comparison depends on it.
- **Reportability containment.** Nothing here relaxes `is_reportable`. `actual` remains blocked
  by (i) `data.volumes: null` at base level — and LoRA-3's `idea.md` §0.2 range is no longer
  achievable at all, since dolly-15k has 15 011 rows, so that volume must be set deliberately
  and its inequality with the other tasks disclosed; (ii) D-002 / D-005 / D-011 / D-012 still
  `OPEN`; (iii) D-008 (no CUDA host). D-016's resolution removed one of the former gates, which
  makes (i) the load-bearing one — it must not be filled in casually.
- **Licensing.** `instr` is CC BY-SA 3.0 (share-alike reaches derivatives), `dialog` is
  CC BY-NC-SA 4.0 (non-commercial), `longform`'s mirror declares none over Reddit-derived
  content. All three strings are carried into every snapshot manifest so a release audit is
  mechanical rather than archaeological.

## 7. 7B feasibility / performance

- Nothing here materialises a dense `ΔW`; no fusion or training code is touched.
- `BackboneCache` stays single-slot; `fetch_backbone.py` loads at most one backbone and
  releases it before exiting, so no second instance becomes resident (`I1.4`).
- Memory: `local_smoke` loads 1.5 B in bf16 on MPS (≈ 3.1 GB) — already the configured tier.
- Disk: ~3.1 GB backbone under `data/models/`, a few MB of capped JSONL under `data/raw/`.
  Both gitignored.
- Download volume is bounded by `data.volumes` and never by corpus size; `.skip(n)` streams
  past rows without materialising them.
- The leakage audit over 6 tasks × 4 splits × ≤ 32 examples is microseconds.

## 8. Errors, and defects found but not fixed here

New/used error types: `UnresolvedRowAdapterError(DataError)` (row adapter unset/unknown),
`DataError` (missing required upstream field, non-existent upstream split, zero examples,
empty target, offset violation), `ConfigError` (missing `data.volumes.*`, missing
`data.sampling`, `fetch_backbone --run-mode dummy`), `LeakageError` (raised by existing
machinery), `QuantizationUnavailableError` (EC-2).

Out-of-scope defects observed during investigation, named rather than fixed:

1. `requirements.txt` pins `datasets==4.5.1` but `5.0.1` is installed. Reproducing this
   environment from `requirements.txt` would restore script-loading and change which datasets
   load. Needs a deliberate re-pin.
2. `docs/system_specification.md` M2 describes `TaskSpec` as carrying `volumes` and
   `build_splits(task_spec, seed, volumes)`; neither matches the code. This feature updates
   the M2 config-key list; the signature prose stays stale.
3. `scripts/run_smoke.py` step 7 is labelled "six adapters trained independently" but
   iterates over whatever tasks are available. Correct for the expected six-task run, but the
   label would silently read wrong if a task ever became blocked.
4. `scripts/run_smoke.py` step 16 hardcodes `adapter_id="seed-sum"` while evaluating
   `tasks[0]`; correct only while `sum` is first and available.

## 9. Testing strategy

Every new test runs on the `dummy` tier, offline, with the HF loader monkeypatched — no test
may touch the network (C-6). The real-network verification lives in the AC commands, not in
`pytest`.

| Test | Asserts |
|---|---|
| `tests/test_row_adapters.py` | `passthrough` byte-identical to pre-change output on a fixed row set; `list_target_pick` both values, with `highest_score` selecting a non-index-0 maximum; dolly-style `context == ""` renders; ELI5-style empty `answers.text` → `DataError`; `dialog_last_turn` history/response split and its <2-utterance no-op; `finqa_pre_table_post` markdown rendering and empty-table tolerance; every rejected candidate still registered with declared `required_row_fields`; `null`/unknown → `UnresolvedRowAdapterError` citing D-015 |
| `tests/test_data_snapshots.py` | key changes on each of 9 content-determining inputs; matching snapshot served with loader monkeypatched to raise; round-trip exact; manifest mismatch → miss |
| `tests/test_single_split_carving.py` | offset-carved pools disjoint by `example_id`; **`test_offset = 0` raises `LeakageError`** (fails against pre-fix code); `test_offset < rows_consumed` raises `DataError`; under `shuffled_buffer`, two independently constructed streams for one task yield identical row order |
| `tests/test_prepare_data_blocked.py` | with a **fixture** spec whose `row_adapter` is blanked and another whose `hf_id` is `null`: both reported, loader never called; a loader `RuntimeError` for an available task propagates. Uses fixtures, not the shipped curriculum, which now configures all six tasks |
| `tests/test_experiment_preconditions.py` (extended) | each missing `data.volumes` key → `ConfigError` naming it; `test_offset` required when `test_split == "train"` |
| `tests/test_data_registry.py` (modified) | unpinned set is now empty; every `hf_revision` matches `^[0-9a-f]{40}$|^refs/convert/parquet$`; licenses still recorded, `dialog` still flagged non-commercial |
| `tests/test_config.py` (extended) | `config_used.json → backbone.revision` present; `model.revision` override honoured |

`tests/test_split_isolation.py` and `tests/test_no_test_leakage_into_decisions.py` must keep
passing untouched — they are the guarantee that this feature did not widen any guard.

## 10. Key implementation decisions

1. **Upstream column names are used verbatim as prompt-template slots.** `longform` becomes
   `input_fields: [title]` + `{title}`, not a `question ← title` mapping. Matches the existing
   convention for `sum`/`qa`; avoids a second naming layer that could drift from the template.
2. **Row adaptation is a named registry with no default**, not a per-task lambda or a
   `if task_id == …` branch. It made D-015 a config decision, and keeps `providers.py`
   task-agnostic now that the decision is resolved — the resolved values live in
   `configs/datasets/curriculum.yaml`, not in code.
3. **Snapshots cache raw rows, not rendered examples** (§3.2).
4. **Streaming offset, not non-streaming slicing**, for single-split corpora (§4).
5. **The DailyDialog pin moves to `refs/convert/parquet`, not to a different repo.** A
   mechanical format conversion of the pinned dataset preserves identity; a mirror does not.
6. **`revision` lives in `BACKBONE_REGISTRY`, not in a script.** M0 states the registry is the
   only place a checkpoint id appears; its revision belongs with it, and D-001 asks for the
   tag to be pinned.
7. **`data.volumes` is `null` at base level.** Making it required-but-absent means an
   `actual` run fails loudly on a value `idea.md` §0.2 only gives as a range — which is the
   correct state until a human fixes the volume. With D-016 resolved this is now the *principal*
   remaining data-side gate on a reportable run (§6).
8. **Shuffle before skip, with a seed derived from `data.seed`.** D-016's `shuffled_buffer`
   would otherwise silently invalidate the offset-based TEST carving of §4, because two
   differently-ordered streams make `skip(n)` meaningless as a disjointness argument.
9. **Blocked ≠ failed, and the path stays live even with nothing blocked.** Only
   `unpinned_dataset` and `unresolved_row_adapter` are skippable; everything else propagates, so
   a transient network error can never masquerade as an open decision. With every decision
   resolved no shipped task is blocked, so the path is kept under test with fixture specs (§9)
   rather than deleted — it is the guard against a future re-pin or added task being guessed at.

"""M2 -- dataset providers.

Two providers:

- ``huggingface``: loads a *pinned* dataset (id + revision).  Refuses unpinned rows
  (D-006).  Never downloads more than the configured volume: we stream and stop, because
  brief S4 explicitly forbids blindly processing entire datasets.
- ``synthetic``: deterministic offline pseudo-data with the right *shape* per task type.
  This is what lets ``pytest`` and the ``dummy`` tier exercise the full pipeline with no
  network, which idea.md S14 requires before any expensive run.

The synthetic provider produces **structurally** realistic data (a context and a
question-shaped target, of plausible lengths) but its content is meaningless.  It is a
correctness fixture, never a source of a reported number.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

from amaun.data import snapshots
from amaun.data.registry import TaskSpec, TaskType
from amaun.data.row_adapters import whitespace_token_count
from amaun.data.splits import Example
from amaun.errors import ConfigError, DataError
from amaun.utils.logging import get_logger

log = get_logger("data.providers")

_VOCAB = (
    "market revenue quarter policy patient dataset gradient adapter summary dialogue "
    "question answer finance report growth decline analysis model token sequence layer "
    "client server round federated continual forgetting utility fusion threshold"
).split()


def _rng_for(task_id: str, index: int, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{task_id}|{index}|{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _words(rng: random.Random, count: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(count))


def _synthetic_example(spec: TaskSpec, index: int, seed: int) -> Example:
    """One deterministic pseudo-example whose shape matches the task type.

    The generated values are positional and are bound to the task's **declared**
    ``input_fields``, not to hardcoded names.  Upstream column names are the template slots
    (design S10.1), so a re-pin that renames a slot -- ``instr``'s ``input`` -> dolly's
    ``context``, ``longform``'s ``question`` -> ELI5's ``title`` -- must not silently break the
    offline ``dummy`` tier.
    """
    rng = _rng_for(spec.task_id, index, seed)
    if spec.task_type is TaskType.SUMMARIZATION:
        values = [_words(rng, rng.randint(60, 120))]
        target = _words(rng, rng.randint(10, 20))
    elif spec.task_type is TaskType.QUESTION_ANSWERING:
        context = _words(rng, rng.randint(40, 80))
        answer = rng.choice(context.split())
        values = [context, f"what is {answer} about"]
        target = answer
    elif spec.task_type is TaskType.INSTRUCTION_FOLLOWING:
        values = [f"explain {_words(rng, 3)}", _words(rng, rng.randint(5, 20))]
        target = _words(rng, rng.randint(15, 30))
    elif spec.task_type is TaskType.DIALOGUE:
        values = [" | ".join(_words(rng, rng.randint(4, 10)) for _ in range(rng.randint(2, 4)))]
        target = _words(rng, rng.randint(6, 15))
    elif spec.task_type is TaskType.LONG_FORM_GENERATION:
        values = [f"why does {_words(rng, 3)} happen"]
        target = _words(rng, rng.randint(40, 90))
    else:  # DOMAIN_GENERATION
        values = [_words(rng, rng.randint(30, 60)), f"compute {_words(rng, 2)}"]
        target = _words(rng, rng.randint(8, 25))

    if len(values) != len(spec.input_fields):
        raise DataError(
            f"task '{spec.task_id}' declares {len(spec.input_fields)} input field(s) "
            f"{list(spec.input_fields)} but the synthetic shape for task_type "
            f"'{spec.task_type.value}' produces {len(values)}; the fixture and the prompt contract "
            f"must agree or the dummy tier would exercise a different task"
        )
    fields = dict(zip(spec.input_fields, values))

    prompt = render_prompt(spec, fields)
    return Example(prompt=prompt, target=target, task_id=spec.task_id,
                   meta={"provider": "synthetic", "index": index, **fields})


def render_prompt(spec: TaskSpec, fields: dict[str, Any]) -> str:
    """Fill the task's prompt template.

    Missing fields fail loudly rather than rendering an empty slot, because a silently
    truncated prompt changes the task the adapter learns.
    """
    try:
        return spec.prompt_template.format(**fields)
    except KeyError as exc:
        raise DataError(
            f"task '{spec.task_id}' prompt_template needs field {exc} but the example "
            f"provides {sorted(fields)}"
        ) from exc


def synthetic_pool(spec: TaskSpec, size: int, *, seed: int, offset: int = 0) -> list[Example]:
    return [_synthetic_example(spec, offset + i, seed) for i in range(size)]


SAMPLING_POLICIES = ("head_of_stream", "shuffled_buffer")


@dataclass
class LoadReport:
    """What one provider load actually did.  Reported, never inferred.

    ``rows_consumed`` (rows read from the stream *after* ``skip``) is deliberately distinct from
    ``n_examples``: a row adapter may emit several examples per row, or none, so the offset
    arithmetic that keeps a single-split corpus's TEST window disjoint (FR-4.4) must be done in
    rows and never in examples.
    """

    task_id: str
    hf_id: str
    hf_revision: str | None
    hf_revision_effective: str | None
    upstream_split: str
    requested_size: int
    skip: int
    sampling: str
    shuffle_buffer_size: int | None
    row_adapter: str
    rows_consumed: int
    n_examples: int
    exhausted: bool
    source: str                      # "download" | "local"
    snapshot_key: str
    rows_yielding_no_examples: int
    over_length_fraction: float
    over_length_metric: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_shuffle_seed(data_seed: int, task_id: str) -> int:
    """A deterministic per-task shuffle seed (FR-3.8, D-016 condition 2).

    It must be a pure function of ``(data.seed, task_id)`` and stable across processes, because
    ``skip``-based TEST carving is a disjointness guarantee *only* if both loads of a task shuffle
    the stream into the same order.  ``hash()`` is salted per process and would silently break
    that; sha256 does not.
    """
    digest = hashlib.sha256(f"{data_seed}|{task_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _effective_revision(spec: TaskSpec) -> str | None:
    """The revision actually requested from the Hub.

    Two of the pinned datasets (``daily_dialog``, ``finqa``) ship only a loading script on
    ``main``, which ``datasets`` >= 4 cannot execute, so they are pinned to
    ``refs/convert/parquet`` *together with* that ref's resolved commit (FR-2.1/FR-2.6).  The
    resolved commit is what we load: a symbolic ref would silently follow a future re-conversion,
    whereas a commit either resolves to the same bytes or fails loudly (EC-11).
    """
    return spec.hf_revision_resolved or spec.hf_revision


def _example_from(spec: TaskSpec, fields: dict[str, str], target: str, meta: dict[str, Any]) -> Example:
    return Example(
        prompt=render_prompt(spec, fields),
        target=str(target).strip(),
        task_id=spec.task_id,
        meta=meta,
    )


def huggingface_pool(
    spec: TaskSpec,
    size: int,
    *,
    split: str,
    row_adapter: Any,
    data_seed: int,
    sampling: str,
    shuffle_buffer_size: int | None,
    offline_snapshots: bool,
    cache_dir: str | None = None,
    skip: int = 0,
) -> tuple[list[Example], LoadReport]:
    """Load at most ``size`` examples from a pinned HF dataset split.

    Always streaming, so a full corpus is never materialised (brief S4: "Do NOT blindly download
    or process entire datasets").  The pipeline order is fixed and load-bearing:

        ``load_dataset(streaming=True) -> .shuffle(seed, buffer) -> .skip(n) -> iterate``

    Shuffling *before* skipping is what keeps ``skip`` meaningful as a disjointness argument for a
    single-split corpus: two loads of the same task shuffle into the same order (the seed is
    derived from ``data.seed`` and the task id), so disjoint row windows stay disjoint.  Skipping
    before shuffling would sample two differently-ordered streams and the windows could overlap.

    Every experiment-affecting parameter is supplied by the caller from ``configs/``; this
    function invents none of them.
    """
    spec.require_pinned()
    if sampling not in SAMPLING_POLICIES:
        raise ConfigError(
            f"data.sampling must be one of {list(SAMPLING_POLICIES)}, got {sampling!r} "
            f"(docs/decision_register.md D-016)"
        )
    if sampling == "shuffled_buffer" and not shuffle_buffer_size:
        raise ConfigError(
            "data.sampling=shuffled_buffer requires data.shuffle_buffer_size; it has no default "
            "because the buffer size determines how near-random the sample is (D-016 condition 1)"
        )

    options = dict(spec.row_adapter_options)
    required_fields = tuple(row_adapter.required_row_fields(spec, options))
    effective_revision = _effective_revision(spec)
    snapshot_spec = snapshots.SnapshotSpec(
        hf_id=str(spec.hf_id),
        hf_revision=effective_revision,
        hf_config=spec.hf_config,
        split=split,
        size=int(size),
        skip=int(skip),
        sampling=sampling,
        shuffle_buffer_size=shuffle_buffer_size if sampling == "shuffled_buffer" else None,
        row_adapter=str(row_adapter.name),
        row_adapter_options=options,
        required_fields=required_fields,
    )
    snapshot_path = (
        snapshots.path_for(cache_dir, spec.task_id, snapshot_spec) if cache_dir else None
    )

    rows: list[dict[str, Any]] | None = None
    source = "download"
    exhausted = False
    if offline_snapshots and snapshot_path is not None:
        cached = snapshots.read(snapshot_path, snapshot_spec)
        if cached is not None:
            rows, manifest = cached
            source = "local"
            exhausted = bool(manifest.exhausted)
            log.info(
                "task %s: reusing local snapshot %s (%d rows, no network)",
                spec.task_id, snapshot_path.name, len(rows),
            )

    if rows is None:
        rows, exhausted = _stream_rows(
            spec, size=size, split=split, skip=skip, sampling=sampling,
            shuffle_buffer_size=shuffle_buffer_size, data_seed=data_seed,
            row_adapter=row_adapter, options=options, required_fields=required_fields,
        )

    # Adapt the (raw, field-restricted) rows identically whether they came from the Hub or from
    # disk, so an offline re-run reproduces the online run exactly.
    examples: list[Example] = []
    rows_consumed = 0
    rows_without_examples = 0
    over_length = 0
    for row in rows:
        if len(examples) >= size:
            break
        rows_consumed += 1
        produced = row_adapter.adapt(spec, row, options)
        if not produced:
            rows_without_examples += 1
        for fields, target in produced:
            if len(examples) >= size:
                break
            meta = {
                "provider": "huggingface",
                "hf_id": spec.hf_id,
                "revision": spec.hf_revision,
                "revision_effective": effective_revision,
                "upstream_split": split,
                "row_adapter": row_adapter.name,
                "source": source,
            }
            example = _example_from(spec, fields, target, meta)
            if whitespace_token_count(example.prompt) > spec.max_input_tokens:
                over_length += 1
            examples.append(example)

    if not examples:
        raise DataError(
            f"loaded zero examples for '{spec.task_id}' from {spec.hf_id}:{split} "
            f"(rows_consumed={rows_consumed}, skip={skip})"
        )

    if source == "download" and offline_snapshots and snapshot_path is not None:
        snapshots.write(
            snapshot_path, rows, spec=snapshot_spec, task_id=spec.task_id,
            license=spec.license, exhausted=exhausted,
            extra={
                "over_length_fraction": over_length / len(examples),
                "over_length_metric": _OVER_LENGTH_METRIC,
                "max_input_tokens": spec.max_input_tokens,
                "n_examples": len(examples),
                "rows_yielding_no_examples": rows_without_examples,
            },
        )

    report = LoadReport(
        task_id=spec.task_id,
        hf_id=str(spec.hf_id),
        hf_revision=spec.hf_revision,
        hf_revision_effective=effective_revision,
        upstream_split=split,
        requested_size=int(size),
        skip=int(skip),
        sampling=sampling,
        shuffle_buffer_size=snapshot_spec.shuffle_buffer_size,
        row_adapter=str(row_adapter.name),
        rows_consumed=rows_consumed,
        n_examples=len(examples),
        exhausted=exhausted,
        source=source,
        snapshot_key=snapshot_spec.key(),
        rows_yielding_no_examples=rows_without_examples,
        over_length_fraction=over_length / len(examples),
        over_length_metric=_OVER_LENGTH_METRIC,
    )
    log.info(
        "task %-10s %s:%s -> %d examples from %d rows (source=%s, skip=%d, exhausted=%s)",
        spec.task_id, spec.hf_id, split, report.n_examples, report.rows_consumed,
        report.source, report.skip, report.exhausted,
    )
    return examples, report


# Named so no consumer can mistake the over-length fraction for a real tokenizer measurement:
# the data layer sits below `models` and has no tokenizer (docs/architecture.md).
_OVER_LENGTH_METRIC = "whitespace_tokens_vs_max_input_tokens (proxy: no tokenizer below M1)"


def _stream_rows(
    spec: TaskSpec,
    *,
    size: int,
    split: str,
    skip: int,
    sampling: str,
    shuffle_buffer_size: int | None,
    data_seed: int,
    row_adapter: Any,
    options: dict[str, Any],
    required_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], bool]:
    """Stream, shuffle, skip, and collect just enough raw rows to fill ``size`` examples."""
    from datasets import load_dataset

    log.info(
        "loading %s (revision=%s, config=%s) split=%s cap=%d skip=%d sampling=%s buffer=%s",
        spec.hf_id, _effective_revision(spec), spec.hf_config, split, size, skip, sampling,
        shuffle_buffer_size,
    )
    dataset = load_dataset(
        spec.hf_id,
        spec.hf_config,
        split=split,
        revision=_effective_revision(spec),
        streaming=True,
    )
    if sampling == "shuffled_buffer":
        dataset = dataset.shuffle(
            seed=derive_shuffle_seed(data_seed, spec.task_id),
            buffer_size=int(shuffle_buffer_size),
        )
    if skip:
        dataset = dataset.skip(int(skip))

    rows: list[dict[str, Any]] = []
    produced = 0
    exhausted = True
    for row in dataset:
        rows.append(snapshots.restrict(row, required_fields))
        produced += len(row_adapter.adapt(spec, row, options))
        if produced >= size:
            exhausted = False       # we stopped at the cap, not at the end of the corpus
            break
    return rows, exhausted

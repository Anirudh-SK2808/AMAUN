"""M2 -- named row->example rules, selected by config with **no default** (D-015).

An upstream dataset row and a task's prompt contract rarely line up one-to-one:

- DailyDialog rows carry ``dialog: list[str]``; the task wants ``history`` and ``response``.
- FinQA rows carry ``pre_text`` / ``table`` / ``post_text``; the task wants a single ``context``.
- ELI5 rows carry ``answers: {a_id, score, text}``; the task wants *one* target string.

Which rule is applied changes **what the adapter learns**, so no rule may be chosen by code.
Every task names its rule in ``configs/datasets/curriculum.yaml`` (``row_adapter`` plus
``row_adapter_options``); an unset or unknown name raises :class:`UnresolvedRowAdapterError`
and the task is skipped and reported, never guessed at.

Rejected D-015 candidates (``dialog_all_prefixes``, ``finqa_text_only``) stay **registered but
unselected** so the alternative remains reachable by a one-line config edit, and so the tests
can keep asserting that the registry -- not a hardcoded branch -- is what selects behaviour.

Layer note (docs/architecture.md): this module imports only ``errors`` and ``data``; it never
touches ``models``, so no tokenizer is available here.  Length accounting against
``max_input_tokens`` therefore uses a whitespace-token *proxy*, named as such wherever it is
reported (see :func:`whitespace_token_count`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from amaun.data.registry import TaskSpec
from amaun.errors import DataError, UnresolvedRowAdapterError

# One adapted row: the prompt-template fields it fills, and the target string.
AdaptedExample = tuple[dict[str, str], str]

_D015 = "docs/decision_register.md D-015"


def whitespace_token_count(text: str) -> int:
    """Whitespace-token count, used only as a *proxy* for tokenised length.

    The data layer sits below ``models`` (docs/architecture.md), so the real tokenizer is not
    importable here.  Every figure derived from this function is labelled as a proxy at the
    point it is reported, so an over-length fraction is never mistaken for a measured token
    count.
    """
    return len(text.split())


class RowAdapter(Protocol):
    """Maps one upstream row to zero or more ``(fields, target)`` pairs."""

    name: str

    def required_row_fields(self, spec: TaskSpec, options: Mapping[str, Any]) -> tuple[str, ...]:
        """Upstream column names this adapter reads.  Used for snapshot restriction + keying."""

    def adapt(
        self, spec: TaskSpec, row: Mapping[str, Any], options: Mapping[str, Any]
    ) -> list[AdaptedExample]:
        """Zero or more examples from one row.  Zero is legal; a missing field is not."""


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _require_option(adapter: str, options: Mapping[str, Any], key: str, allowed: Sequence[str]) -> str:
    value = options.get(key)
    if value is None:
        raise UnresolvedRowAdapterError(
            f"row_adapter '{adapter}' requires row_adapter_options.{key} (one of "
            f"{list(allowed)}); AMAUN provides no default because the choice changes what the "
            f"adapter learns ({_D015})"
        )
    value = str(value)
    if value not in allowed:
        raise UnresolvedRowAdapterError(
            f"row_adapter '{adapter}' got row_adapter_options.{key}={value!r}, expected one of "
            f"{list(allowed)} ({_D015})"
        )
    return value


def _require_field(spec: TaskSpec, row: Mapping[str, Any], name: str) -> Any:
    """Read an upstream field, aborting if it is *absent*.

    A field that is present but empty (dolly's ``context``, FinQA's ``table``) is valid input,
    not an error: FR-6.7.  Skipping rows instead of aborting would silently change the corpus.
    """
    if name not in row:
        raise DataError(
            f"task '{spec.task_id}': dataset {spec.hf_id} row lacks required upstream field "
            f"'{name}' (row has {sorted(row)}). The row adapter aborts rather than skipping, "
            f"because a partially-adapted corpus silently changes the task."
        )
    return row[name]


def _flatten_text(value: Any) -> str:
    """Render a text field that upstream may store as a string or a list of sentences."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(part).strip() for part in value if str(part).strip())
    return str(value)


def _reduce_target(spec: TaskSpec, raw_target: Any, pick: str) -> str:
    """Reduce a possibly list/dict-valued target to one string under an explicit policy.

    ``first`` reproduces the pre-change provider exactly; ``highest_score`` uses the parallel
    ``score`` list the dataset provides (D-015 (c)).  There is no fallback between them: a
    ``highest_score`` request against a target with no scores is a configuration error, not an
    occasion to quietly take element 0.
    """
    target = raw_target
    scores: Sequence[Any] | None = None
    if isinstance(target, Mapping):
        if "text" not in target:
            raise DataError(
                f"task '{spec.task_id}': target field '{spec.target_field}' is a mapping with "
                f"keys {sorted(target)} but no 'text' entry to reduce"
            )
        scores = target.get("score")
        target = target["text"]
    if isinstance(target, (list, tuple)):
        if not target:
            raise DataError(
                f"task '{spec.task_id}': target field '{spec.target_field}' is an empty list; a "
                f"zero-length target would train on nothing, so the row is refused rather than "
                f"silently dropped"
            )
        if pick == "first":
            target = target[0]
        else:  # highest_score
            if not scores or len(scores) != len(target):
                raise DataError(
                    f"task '{spec.task_id}': list_target_pick=highest_score needs a 'score' list "
                    f"parallel to 'text', got scores={None if not scores else len(scores)} for "
                    f"{len(target)} candidate(s) ({_D015} (c))"
                )
            best = max(range(len(target)), key=lambda i: float(scores[i]))
            target = target[best]
    text = str(target).strip()
    if not text:
        raise DataError(
            f"task '{spec.task_id}': reduced target for field '{spec.target_field}' is empty; "
            f"a zero-length target would train on nothing"
        )
    return text


# --------------------------------------------------------------------------------------
# passthrough -- today's provider logic, with the hidden list choice made explicit
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Passthrough:
    """Upstream column names *are* the prompt-template slots (``sum``, ``qa``, ``instr``, ``longform``).

    This is the existing convention, not a new one: ``{article}``, ``{context}``, ``{question}``
    are already upstream names.  The single behaviour change versus the pre-change provider is
    that reducing a list-valued target now requires an explicit ``list_target_pick``, because
    the old code silently took element 0.
    """

    name: str = "passthrough"

    def required_row_fields(self, spec: TaskSpec, options: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(spec.input_fields) + (spec.target_field,)

    def adapt(
        self, spec: TaskSpec, row: Mapping[str, Any], options: Mapping[str, Any]
    ) -> list[AdaptedExample]:
        pick = _require_option(self.name, options, "list_target_pick", ("first", "highest_score"))
        fields = {name: _flatten_text(_require_field(spec, row, name)) for name in spec.input_fields}
        raw_target = _require_field(spec, row, spec.target_field)
        if raw_target is None:
            raise DataError(
                f"task '{spec.task_id}': dataset {spec.hf_id} row has target field "
                f"'{spec.target_field}' set to null"
            )
        return [(fields, _reduce_target(spec, raw_target, pick))]


# --------------------------------------------------------------------------------------
# dialog -- DailyDialog turn segmentation (D-015 (a))
# --------------------------------------------------------------------------------------


_DIALOG_SOURCE_FIELD = "dialog"   # DailyDialog's upstream column of utterances


def _utterances(spec: TaskSpec, row: Mapping[str, Any]) -> list[str]:
    raw = _require_field(spec, row, _DIALOG_SOURCE_FIELD)
    if isinstance(raw, str):
        raise DataError(
            f"task '{spec.task_id}': upstream field '{_DIALOG_SOURCE_FIELD}' is a string, but the "
            f"dialogue adapters need a list of utterances"
        )
    return [str(part).strip() for part in raw if str(part).strip()]


_HISTORY_SLOT = "history"


def _dialog_fields(
    spec: TaskSpec, row: Mapping[str, Any], history: Sequence[str], separator: str
) -> dict[str, str]:
    """Fill the ``history`` slot from the segmented utterances; any other slot comes from the row.

    Filling an unknown slot with an empty string would silently truncate the prompt, so slots
    the adapter does not produce are read from upstream and abort if absent.
    """
    fields: dict[str, str] = {}
    for name in spec.input_fields:
        if name == _HISTORY_SLOT:
            fields[name] = separator.join(history)
        else:
            fields[name] = _flatten_text(_require_field(spec, row, name))
    return fields


@dataclass(frozen=True)
class _DialogLastTurn:
    """SELECTED by ``dialog`` (D-015 (a)): one example per dialogue.

    ``history`` is every utterance but the last, ``target`` the last.  A row with fewer than two
    utterances has no (history, response) pair at all and emits nothing -- counted, not hidden.
    """

    name: str = "dialog_last_turn"

    def required_row_fields(self, spec: TaskSpec, options: Mapping[str, Any]) -> tuple[str, ...]:
        return (_DIALOG_SOURCE_FIELD,)

    def adapt(
        self, spec: TaskSpec, row: Mapping[str, Any], options: Mapping[str, Any]
    ) -> list[AdaptedExample]:
        separator = options.get("separator")
        if separator is None:
            raise UnresolvedRowAdapterError(
                f"row_adapter '{self.name}' requires row_adapter_options.separator (the string "
                f"joining history utterances); it is part of the rendered prompt and has no code "
                f"default ({_D015} (a))"
            )
        utterances = _utterances(spec, row)
        if len(utterances) < 2:
            return []
        return [(_dialog_fields(spec, row, utterances[:-1], str(separator)), utterances[-1])]


@dataclass(frozen=True)
class _DialogAllPrefixes:
    """REGISTERED BUT UNSELECTED (D-015 (a) rejected option): one example per turn prefix.

    Kept implemented so the comparison stays reachable by a config edit; ~7x more examples,
    which shifts both the context-length distribution and this task's effective volume relative
    to every other task.
    """

    name: str = "dialog_all_prefixes"

    def required_row_fields(self, spec: TaskSpec, options: Mapping[str, Any]) -> tuple[str, ...]:
        return (_DIALOG_SOURCE_FIELD,)

    def adapt(
        self, spec: TaskSpec, row: Mapping[str, Any], options: Mapping[str, Any]
    ) -> list[AdaptedExample]:
        separator = options.get("separator")
        if separator is None:
            raise UnresolvedRowAdapterError(
                f"row_adapter '{self.name}' requires row_adapter_options.separator ({_D015} (a))"
            )
        utterances = _utterances(spec, row)
        out: list[AdaptedExample] = []
        for index in range(1, len(utterances)):
            out.append(
                (_dialog_fields(spec, row, utterances[:index], str(separator)), utterances[index])
            )
        return out


# --------------------------------------------------------------------------------------
# domain -- FinQA context composition (D-015 (b))
# --------------------------------------------------------------------------------------


_FINQA_PRE = "pre_text"
_FINQA_POST = "post_text"
_FINQA_TABLE = "table"
_CONTEXT_SLOT = "context"


def _serialise_table(table: Any, table_format: str) -> str:
    """Render FinQA's ``list[list[str]]`` table.  An empty table renders as the empty string."""
    if not table:
        return ""
    rows = [[str(cell).strip() for cell in row] for row in table]
    if table_format == "pipe_rows":
        return "\n".join(" | ".join(row) for row in rows)
    # markdown: first row is the header, then the alignment rule, then the body.
    header, *body = rows
    width = len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in body:
        padded = list(row) + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded[:width]) + " |")
    return "\n".join(lines)


def _finqa_fields(
    spec: TaskSpec, row: Mapping[str, Any], context: str
) -> dict[str, str]:
    fields: dict[str, str] = {}
    for name in spec.input_fields:
        if name == _CONTEXT_SLOT:
            fields[name] = context
        else:
            fields[name] = _flatten_text(_require_field(spec, row, name))
    return fields


@dataclass(frozen=True)
class _FinqaPreTablePost:
    """SELECTED by ``domain`` (D-015 (b)): FinQA's original input.

    ``context`` = ``pre_text`` + serialised ``table`` + ``post_text``.  Contexts are long against
    ``max_input_tokens``, so the provider measures the over-length fraction rather than assuming
    truncation is rare (FR-6.5a).
    """

    name: str = "finqa_pre_table_post"

    def required_row_fields(self, spec: TaskSpec, options: Mapping[str, Any]) -> tuple[str, ...]:
        extra = tuple(n for n in spec.input_fields if n != _CONTEXT_SLOT)
        return (_FINQA_PRE, _FINQA_TABLE, _FINQA_POST) + extra + (spec.target_field,)

    def adapt(
        self, spec: TaskSpec, row: Mapping[str, Any], options: Mapping[str, Any]
    ) -> list[AdaptedExample]:
        table_format = _require_option(
            self.name, options, "table_format", ("markdown", "pipe_rows")
        )
        parts = [
            _flatten_text(_require_field(spec, row, _FINQA_PRE)),
            _serialise_table(_require_field(spec, row, _FINQA_TABLE), table_format),
            _flatten_text(_require_field(spec, row, _FINQA_POST)),
        ]
        context = "\n\n".join(part for part in parts if part)
        raw_target = _require_field(spec, row, spec.target_field)
        if not str(raw_target).strip():
            # ~1.8% of FinQA rows carry only `gold_evidence` (a narrative justification) with no
            # extracted final `answer` -- a real, present-but-unusable target, not a missing field.
            # Zero examples is the documented contract for this case (RowAdapter.adapt docstring),
            # not a reason to abort the whole stream over ~2% of rows.
            return []
        target = _reduce_target(spec, raw_target, "first")
        return [(_finqa_fields(spec, row, context), target)]


@dataclass(frozen=True)
class _FinqaTextOnly:
    """REGISTERED BUT UNSELECTED (D-015 (b) rejected option): the table is dropped.

    Shorter inputs, but many FinQA answers are unanswerable without the table, so the task
    becomes materially easier/incoherent.  Kept reachable by config for the comparison.
    """

    name: str = "finqa_text_only"

    def required_row_fields(self, spec: TaskSpec, options: Mapping[str, Any]) -> tuple[str, ...]:
        extra = tuple(n for n in spec.input_fields if n != _CONTEXT_SLOT)
        return (_FINQA_PRE, _FINQA_POST) + extra + (spec.target_field,)

    def adapt(
        self, spec: TaskSpec, row: Mapping[str, Any], options: Mapping[str, Any]
    ) -> list[AdaptedExample]:
        parts = [
            _flatten_text(_require_field(spec, row, _FINQA_PRE)),
            _flatten_text(_require_field(spec, row, _FINQA_POST)),
        ]
        context = "\n\n".join(part for part in parts if part)
        raw_target = _require_field(spec, row, spec.target_field)
        if not str(raw_target).strip():
            return []
        target = _reduce_target(spec, raw_target, "first")
        return [(_finqa_fields(spec, row, context), target)]


# --------------------------------------------------------------------------------------
# label_to_text -- integer class labels (IMDb) rendered as a target word
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _LabelToText:
    """Classification rows whose target is an integer label (IMDb: ``label`` in {0, 1}).

    The label->word mapping is ``row_adapter_options.labels`` and has no default: which words
    the model must generate is part of the task definition, and a label missing from the mapping
    aborts rather than being dropped (a silently filtered class would change the class balance).
    """

    name: str = "label_to_text"

    def required_row_fields(self, spec: TaskSpec, options: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(spec.input_fields) + (spec.target_field,)

    def adapt(
        self, spec: TaskSpec, row: Mapping[str, Any], options: Mapping[str, Any]
    ) -> list[AdaptedExample]:
        labels = options.get("labels")
        if not isinstance(labels, Mapping) or not labels:
            raise UnresolvedRowAdapterError(
                f"row_adapter '{self.name}' requires row_adapter_options.labels, a mapping from "
                f"upstream label value to target text ({_D015})"
            )
        fields = {name: _flatten_text(_require_field(spec, row, name)) for name in spec.input_fields}
        raw = _require_field(spec, row, spec.target_field)
        key = str(raw)
        mapping = {str(k): str(v) for k, v in labels.items()}
        if key not in mapping:
            raise DataError(
                f"task '{spec.task_id}': label {raw!r} has no entry in row_adapter_options.labels "
                f"{sorted(mapping)}"
            )
        return [(fields, mapping[key])]


# --------------------------------------------------------------------------------------
# conll_entities -- token/BIO-tag rows (CoNLL-2003) as sentence -> "TYPE: span" lines
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ConllEntities:
    """CoNLL-style NER rows: ``tokens`` and integer BIO ``ner_tags`` (D-021).

    Input slot ``sentence`` = tokens joined by single spaces.  Target = one ``TYPE: span`` line per
    entity in sentence order, or ``row_adapter_options.empty_target`` when there is none.  The tag
    names (index -> ``B-PER`` ...) are an option with no default, because they are the dataset's
    ClassLabel order.  Rows whose tokens are exactly ``row_adapter_options.skip_tokens`` (CoNLL's
    ``-DOCSTART-`` separators) emit nothing.  An ``I-`` tag that does not continue an entity of the
    same type starts a new one (the conlleval convention).
    """

    name: str = "conll_entities"

    def required_row_fields(self, spec: TaskSpec, options: Mapping[str, Any]) -> tuple[str, ...]:
        return ("tokens", "ner_tags")

    def adapt(
        self, spec: TaskSpec, row: Mapping[str, Any], options: Mapping[str, Any]
    ) -> list[AdaptedExample]:
        names = options.get("tag_names")
        empty = options.get("empty_target")
        if not names or empty is None:
            raise UnresolvedRowAdapterError(
                f"row_adapter '{self.name}' requires row_adapter_options.tag_names (the dataset's "
                f"ClassLabel names in index order) and empty_target ({_D015})"
            )
        tokens = [str(t) for t in _require_field(spec, row, "tokens")]
        tags = list(_require_field(spec, row, "ner_tags"))
        if len(tokens) != len(tags):
            raise DataError(f"task '{spec.task_id}': {len(tokens)} tokens but {len(tags)} tags")
        if tokens == list(options.get("skip_tokens") or []) or not tokens:
            return []
        entities: list[tuple[str, list[str]]] = []
        open_kind: str | None = None          # type of the entity the previous token belongs to
        for token, tag in zip(tokens, tags):
            label = tag if isinstance(tag, str) else str(names[int(tag)])
            if label == "O":
                open_kind = None
                continue
            prefix, _, kind = label.partition("-")
            if prefix == "I" and open_kind == kind:
                entities[-1][1].append(token)
            else:
                entities.append((kind, [token]))
            open_kind = kind
        fields = {name: " ".join(tokens) for name in spec.input_fields}
        target = "\n".join(f"{kind}: {' '.join(span)}" for kind, span in entities) or str(empty)
        return [(fields, target)]


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


ROW_ADAPTERS: dict[str, RowAdapter] = {
    adapter.name: adapter
    for adapter in (
        _Passthrough(),
        _DialogLastTurn(),
        _DialogAllPrefixes(),
        _FinqaPreTablePost(),
        _FinqaTextOnly(),
        _LabelToText(),
        _ConllEntities(),
    )
}


def resolve(spec: TaskSpec) -> RowAdapter:
    """Look up the task's configured rule.  Unset or unknown -> :class:`UnresolvedRowAdapterError`."""
    name = spec.row_adapter
    if not name:
        raise UnresolvedRowAdapterError(
            f"task '{spec.task_id}' has no row_adapter configured. Set "
            f"datasets.tasks.{spec.task_id}.row_adapter to one of {sorted(ROW_ADAPTERS)} in "
            f"configs/datasets/; AMAUN provides no default because the rule changes what the "
            f"adapter learns ({_D015})"
        )
    if name not in ROW_ADAPTERS:
        raise UnresolvedRowAdapterError(
            f"task '{spec.task_id}' names unknown row_adapter '{name}'; registered adapters are "
            f"{sorted(ROW_ADAPTERS)} ({_D015})"
        )
    return ROW_ADAPTERS[name]

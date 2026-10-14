"""M2 -- row->example adaptation (FR-6, AC-14).  Offline: no network, no tokenizer, no backbone.

The load-bearing assertion here is the first one: ``passthrough`` must reproduce the
pre-change provider *exactly*, so that introducing the registry did not silently alter
``sum``/``qa``/``instr``.  The reference implementation below is the pre-change
``huggingface_pool`` row loop copied verbatim from commit f894b99.
"""

from __future__ import annotations

import pytest

from amaun.data.providers import render_prompt
from amaun.data.registry import task_spec_from_config
from amaun.data.row_adapters import ROW_ADAPTERS, resolve
from amaun.errors import DataError, UnresolvedRowAdapterError


# --------------------------------------------------------------------------------------
# Fixtures: task specs shaped like the real curriculum rows, built without touching configs/
# --------------------------------------------------------------------------------------


def _spec(task_id: str, **overrides):
    base = {
        "task_name": task_id,
        "task_type": "summarization",
        "primary_score": "rougeL",
        "metrics": ["rougeL"],
        "input_fields": ["article"],
        "target_field": "highlights",
        "prompt_template": "Summarize:\n{article}\n\nSummary:",
        "max_input_tokens": 64,
        "max_target_tokens": 16,
        "hf_id": "fixture/dataset",
        "hf_revision": "0" * 40,
        "row_adapter": "passthrough",
        "row_adapter_options": {"list_target_pick": "first"},
    }
    base.update(overrides)
    return task_spec_from_config(task_id, base)


SUM_SPEC = _spec("sum")
QA_SPEC = _spec(
    "qa",
    task_type="question_answering",
    input_fields=["context", "question"],
    target_field="answers",
    prompt_template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
)
INSTR_SPEC = _spec(
    "instr",
    task_type="instruction_following",
    input_fields=["instruction", "context"],
    target_field="response",
    prompt_template="{instruction}\n\n{context}\n\nResponse:",
)
LONGFORM_SPEC = _spec(
    "longform",
    task_type="long_form_generation",
    input_fields=["title"],
    target_field="answers",
    prompt_template="Explain in detail.\n\nQuestion: {title}\n\nExplanation:",
    row_adapter_options={"list_target_pick": "highest_score"},
)
DIALOG_SPEC = _spec(
    "dialog",
    task_type="dialogue",
    input_fields=["history"],
    target_field="response",
    prompt_template="Continue the conversation naturally.\n\n{history}\n\nResponse:",
    row_adapter="dialog_last_turn",
    row_adapter_options={"separator": " | "},
)
DOMAIN_SPEC = _spec(
    "domain",
    task_type="domain_generation",
    input_fields=["context", "question"],
    target_field="answer",
    prompt_template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    row_adapter="finqa_pre_table_post",
    row_adapter_options={"table_format": "markdown"},
)

SUM_ROWS = [
    {"article": "a long article about markets", "highlights": "markets moved", "id": "1"},
    {"article": "another article", "highlights": "second summary", "id": "2"},
]
QA_ROWS = [
    {
        "context": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "answers": {"text": ["Paris", "Paris, France"], "answer_start": [0, 0]},
        "id": "q1",
    }
]


def _legacy_examples(spec, rows):
    """The pre-change provider's row loop, verbatim (commit f894b99 providers.py:122-145)."""
    out = []
    for row in rows:
        fields = {name: row[name] for name in spec.input_fields if name in row}
        if len(fields) != len(spec.input_fields):
            missing = set(spec.input_fields) - set(fields)
            raise DataError(f"dataset {spec.hf_id} row lacks input fields {sorted(missing)}")
        target = row.get(spec.target_field)
        if target is None:
            raise DataError(f"dataset {spec.hf_id} row lacks target field '{spec.target_field}'")
        if isinstance(target, dict) and "text" in target:
            texts = target["text"]
            target = texts[0] if texts else ""
        if isinstance(target, list):
            target = target[0] if target else ""
        out.append((render_prompt(spec, fields), str(target).strip()))
    return out


def _adapted_examples(spec, rows):
    adapter = resolve(spec)
    out = []
    for row in rows:
        for fields, target in adapter.adapt(spec, row, spec.row_adapter_options):
            out.append((render_prompt(spec, fields), target))
    return out


# --------------------------------------------------------------------------------------
# FR-6.3 -- passthrough documents existing behaviour rather than changing it
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec,rows", [(SUM_SPEC, SUM_ROWS), (QA_SPEC, QA_ROWS)])
def test_passthrough_is_byte_identical_to_the_pre_change_provider(spec, rows):
    assert _adapted_examples(spec, rows) == _legacy_examples(spec, rows)


def test_passthrough_requires_an_explicit_list_target_pick():
    """FR-6.4: the old code silently took element 0; that choice is now config-declared."""
    spec = _spec("qa_nopick", input_fields=["context"], target_field="answers",
                 prompt_template="{context}", row_adapter_options={})
    with pytest.raises(UnresolvedRowAdapterError, match="list_target_pick"):
        resolve(spec).adapt(spec, {"context": "c", "answers": {"text": ["a"], "score": [1]}}, {})


# --------------------------------------------------------------------------------------
# FR-6.4 / D-015 (c) -- list_target_pick
# --------------------------------------------------------------------------------------


ELI5_ROW = {
    "title": "why is the sky blue",
    "answers": {
        "a_id": ["a0", "a1", "a2"],
        # The maximum is deliberately NOT at index 0: otherwise the two policies would be
        # indistinguishable and the test would prove nothing.
        "score": [2, 91, 7],
        "text": ["short answer", "the best answer", "middling answer"],
    },
}


def test_list_target_pick_first_takes_element_zero():
    spec = _spec("lf_first", input_fields=["title"], target_field="answers",
                 prompt_template="{title}", row_adapter_options={"list_target_pick": "first"})
    (_, target), = resolve(spec).adapt(spec, ELI5_ROW, spec.row_adapter_options)
    assert target == "short answer"


def test_list_target_pick_highest_score_takes_argmax_not_index_zero():
    """D-015 (c): the target is text[argmax(score)]."""
    (_, target), = resolve(LONGFORM_SPEC).adapt(
        LONGFORM_SPEC, ELI5_ROW, LONGFORM_SPEC.row_adapter_options
    )
    assert target == "the best answer"
    assert target != ELI5_ROW["answers"]["text"][0]


def test_highest_score_without_parallel_scores_is_an_error_not_a_fallback():
    row = {"title": "t", "answers": {"text": ["a", "b"]}}
    with pytest.raises(DataError, match="highest_score"):
        resolve(LONGFORM_SPEC).adapt(LONGFORM_SPEC, row, LONGFORM_SPEC.row_adapter_options)


def test_empty_answer_list_raises_rather_than_dropping_the_row():
    """EC-10: a zero-length target would train on nothing."""
    row = {"title": "t", "answers": {"a_id": [], "score": [], "text": []}}
    with pytest.raises(DataError):
        resolve(LONGFORM_SPEC).adapt(LONGFORM_SPEC, row, LONGFORM_SPEC.row_adapter_options)


# --------------------------------------------------------------------------------------
# FR-6.7 / EC-9 -- present-but-empty is valid; absent is fatal
# --------------------------------------------------------------------------------------


def test_dolly_style_empty_context_renders_without_error():
    """EC-9: dolly's `context` is empty for a large share of rows -- valid, not missing."""
    row = {"instruction": "Explain gravity.", "context": "", "response": "It attracts mass."}
    (prompt, target), = _adapted_examples(INSTR_SPEC, [row])
    assert "Explain gravity." in prompt
    assert target == "It attracts mass."


def test_absent_required_field_aborts():
    row = {"instruction": "Explain gravity.", "response": "It attracts mass."}   # no 'context'
    with pytest.raises(DataError, match="context"):
        _adapted_examples(INSTR_SPEC, [row])


# --------------------------------------------------------------------------------------
# D-015 (a) -- dialogue segmentation
# --------------------------------------------------------------------------------------


DIALOG_ROW = {"dialog": ["hello there", "hi, how are you?", "fine thanks"], "act": [1, 2, 3]}


def test_dialog_last_turn_splits_history_and_response():
    (fields, target), = resolve(DIALOG_SPEC).adapt(
        DIALOG_SPEC, DIALOG_ROW, DIALOG_SPEC.row_adapter_options
    )
    assert fields["history"] == "hello there | hi, how are you?"
    assert target == "fine thanks"


def test_dialog_last_turn_emits_nothing_for_a_single_utterance_row():
    """EC-12: no (history, response) pair exists, so the row yields no example."""
    assert resolve(DIALOG_SPEC).adapt(
        DIALOG_SPEC, {"dialog": ["hello"]}, DIALOG_SPEC.row_adapter_options
    ) == []


def test_dialog_all_prefixes_is_registered_and_yields_more_examples():
    """The rejected D-015 (a) candidate stays reachable by config alone."""
    spec = _spec("dialog_alt", task_type="dialogue", input_fields=["history"],
                 target_field="response", prompt_template="{history}",
                 row_adapter="dialog_all_prefixes", row_adapter_options={"separator": " | "})
    produced = resolve(spec).adapt(spec, DIALOG_ROW, spec.row_adapter_options)
    assert len(produced) == 2                       # one per prefix, vs 1 for last_turn
    assert produced[0][1] == "hi, how are you?"


def test_dialog_adapters_require_an_explicit_separator():
    spec = _spec("dialog_nosep", task_type="dialogue", input_fields=["history"],
                 target_field="response", prompt_template="{history}",
                 row_adapter="dialog_last_turn", row_adapter_options={})
    with pytest.raises(UnresolvedRowAdapterError, match="separator"):
        resolve(spec).adapt(spec, DIALOG_ROW, {})


# --------------------------------------------------------------------------------------
# D-015 (b) -- FinQA context composition
# --------------------------------------------------------------------------------------


FINQA_ROW = {
    "pre_text": ["the table below shows revenue", "in millions"],
    "post_text": ["revenue grew year over year"],
    "table": [["year", "revenue"], ["2018", "100"], ["2019", "120"]],
    "question": "what was the change in revenue?",
    "answer": "20",
}


def test_finqa_pre_table_post_renders_a_markdown_table_between_the_text_parts():
    (fields, target), = resolve(DOMAIN_SPEC).adapt(
        DOMAIN_SPEC, FINQA_ROW, DOMAIN_SPEC.row_adapter_options
    )
    context = fields["context"]
    assert "| year | revenue |" in context
    assert "| --- | --- |" in context
    assert "| 2019 | 120 |" in context
    assert context.index("the table below shows revenue") < context.index("| year | revenue |")
    assert context.index("| year | revenue |") < context.index("revenue grew year over year")
    assert fields["question"] == FINQA_ROW["question"]
    assert target == "20"


def test_finqa_pre_table_post_tolerates_an_empty_table():
    """EC-13: text-only rendering is valid, not an error."""
    row = {**FINQA_ROW, "table": []}
    (fields, _), = resolve(DOMAIN_SPEC).adapt(DOMAIN_SPEC, row, DOMAIN_SPEC.row_adapter_options)
    assert "|" not in fields["context"]
    assert "the table below shows revenue" in fields["context"]


def test_finqa_pipe_rows_format_differs_from_markdown():
    spec = _spec("domain_pipe", task_type="domain_generation",
                 input_fields=["context", "question"], target_field="answer",
                 prompt_template="{context}\n{question}", row_adapter="finqa_pre_table_post",
                 row_adapter_options={"table_format": "pipe_rows"})
    (fields, _), = resolve(spec).adapt(spec, FINQA_ROW, spec.row_adapter_options)
    assert "year | revenue" in fields["context"]
    assert "| --- |" not in fields["context"]


def test_finqa_text_only_is_registered_and_drops_the_table():
    """The rejected D-015 (b) candidate stays reachable by config alone."""
    spec = _spec("domain_alt", task_type="domain_generation",
                 input_fields=["context", "question"], target_field="answer",
                 prompt_template="{context}\n{question}", row_adapter="finqa_text_only",
                 row_adapter_options={})
    (fields, _), = resolve(spec).adapt(spec, FINQA_ROW, spec.row_adapter_options)
    assert "2019" not in fields["context"]
    assert "revenue grew year over year" in fields["context"]


def test_finqa_requires_an_explicit_table_format():
    spec = _spec("domain_nofmt", task_type="domain_generation",
                 input_fields=["context", "question"], target_field="answer",
                 prompt_template="{context}\n{question}", row_adapter="finqa_pre_table_post",
                 row_adapter_options={})
    with pytest.raises(UnresolvedRowAdapterError, match="table_format"):
        resolve(spec).adapt(spec, FINQA_ROW, {})


# --------------------------------------------------------------------------------------
# FR-6.1 / FR-6.6 -- the registry is the selection mechanism, and it has no default
# --------------------------------------------------------------------------------------


def test_every_d015_candidate_is_registered_and_declares_required_row_fields():
    expected = {
        "passthrough", "dialog_last_turn", "dialog_all_prefixes",
        "finqa_pre_table_post", "finqa_text_only",
    }
    assert expected <= set(ROW_ADAPTERS)
    for name in expected:
        adapter = ROW_ADAPTERS[name]
        spec = DOMAIN_SPEC if name.startswith("finqa") else DIALOG_SPEC
        fields = adapter.required_row_fields(spec, spec.row_adapter_options)
        assert isinstance(fields, tuple) and fields, name
        assert all(isinstance(f, str) for f in fields), name


def test_unset_row_adapter_raises_citing_d015():
    spec = _spec("unset", row_adapter=None)
    with pytest.raises(UnresolvedRowAdapterError) as info:
        resolve(spec)
    assert "D-015" in str(info.value)


def test_unknown_row_adapter_raises_citing_d015():
    spec = _spec("unknown", row_adapter="invent_something")
    with pytest.raises(UnresolvedRowAdapterError) as info:
        resolve(spec)
    assert "D-015" in str(info.value)
    assert "invent_something" in str(info.value)

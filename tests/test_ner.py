"""NER for l4_full_study (D-021): the CoNLL row adapter and entity-level F1."""

import dataclasses

import pytest

from amaun.data.registry import TaskType
from amaun.data.row_adapters import resolve
from amaun.errors import UnresolvedRowAdapterError
from amaun.evaluation.task_metrics import compute_with_examples, entity_f1, entity_f1_micro, parse_entities

NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
OPTIONS = {"tag_names": NAMES, "empty_target": "none", "skip_tokens": ["-DOCSTART-"]}


@pytest.fixture
def ner_spec(tasks):
    return dataclasses.replace(tasks[0], task_id="ner", task_type=TaskType.NAMED_ENTITY_RECOGNITION,
                               input_fields=("sentence",), target_field="ner_tags",
                               row_adapter="conll_entities", row_adapter_options=OPTIONS)


def test_conll_rows_become_typed_span_lines(ner_spec):
    adapter = resolve(ner_spec)
    row = {"tokens": ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
           "ner_tags": [3, 0, 7, 0, 0, 0, 7, 0, 0]}
    [(fields, target)] = adapter.adapt(ner_spec, row, OPTIONS)
    assert fields == {"sentence": "EU rejects German call to boycott British lamb ."}
    assert target == "ORG: EU\nMISC: German\nMISC: British"
    # Multi-token entity; an I- tag of another type starts a new entity (conlleval convention).
    [(_, target)] = adapter.adapt(ner_spec, {"tokens": ["Peter", "Blackburn", "Paris"], "ner_tags": [1, 2, 6]}, OPTIONS)
    assert target == "PER: Peter Blackburn\nLOC: Paris"
    [(_, target)] = adapter.adapt(ner_spec, {"tokens": ["no", "entities"], "ner_tags": [0, 0]}, OPTIONS)
    assert target == "none"
    assert adapter.adapt(ner_spec, {"tokens": ["-DOCSTART-"], "ner_tags": [0]}, OPTIONS) == []


def test_conll_adapter_requires_its_options(ner_spec):
    with pytest.raises(UnresolvedRowAdapterError):
        resolve(ner_spec).adapt(ner_spec, {"tokens": ["a"], "ner_tags": [0]}, {})


def test_entity_f1_is_exact_type_and_span():
    reference = "PER: Peter Blackburn\nLOC: Paris"
    assert entity_f1(reference, reference) == 1.0
    assert entity_f1("per:  Peter   Blackburn\nLOC: Paris", reference) == 1.0     # type case, spacing
    assert entity_f1("PER: Peter\nLOC: Paris", reference) == pytest.approx(0.5)   # partial span = miss
    assert entity_f1("ORG: Paris", "LOC: Paris") == 0.0                           # wrong type = miss
    assert entity_f1("none", "none") == 1.0 and entity_f1("none", "LOC: Paris") == 0.0
    assert parse_entities("garbage without colon\nnone") == {}


def test_micro_f1_and_per_example_scores():
    predictions = ["PER: A\nLOC: B", "none"]
    references = ["PER: A", "ORG: C"]
    assert entity_f1_micro(predictions, references) == pytest.approx(2 * 0.5 * 0.5 / 1.0)
    metrics, per_example = compute_with_examples(["entity_f1", "entity_f1_micro"], predictions, references)
    assert per_example["entity_f1"] == [pytest.approx(2 / 3), 0.0]
    assert metrics["entity_f1"] == pytest.approx(1 / 3)
    assert "entity_f1_micro" not in per_example               # corpus-level only

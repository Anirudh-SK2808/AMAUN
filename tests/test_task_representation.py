"""M5 -- task representation from SUPPORT only (criterion E1)."""

import pytest

from amaun.data.splits import SplitName, evaluator_guard, manager_guard, trainer_guard
from amaun.errors import ManagerError
from amaun.manager.task_representation import build_task_representation, support_set_statistics


def _build(task, splits, backbone, **overrides):
    config = {"encoder": "backbone_mean_hidden", "max_support_examples": 4,
              "pooling": "mean", "normalise": True, "max_encoder_tokens": 64}
    config.update(overrides)
    return build_task_representation(
        task.task_id, manager_guard(splits[task.task_id]), config=config, backbone=backbone
    )


def test_backbone_encoder_produces_a_normalised_embedding(tasks, splits, backbone):
    representation = _build(tasks[0], splits, backbone)
    assert len(representation.embedding) == backbone.hidden_size
    norm = sum(v * v for v in representation.embedding) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-5)
    assert representation.provenance["split_read"] == "support"


def test_token_stats_encoder_needs_no_model(tasks, splits):
    representation = build_task_representation(
        tasks[0].task_id, manager_guard(splits[tasks[0].task_id]),
        config={"encoder": "token_stats", "max_support_examples": 4, "normalise": False},
    )
    assert representation.embedding
    assert representation.encoder == "token_stats"


def test_representation_requires_a_support_permitting_guard(tasks, splits):
    """The manager may read SUPPORT only; a trainer guard must be refused."""
    for guard_factory in (trainer_guard, evaluator_guard):
        with pytest.raises(ManagerError):
            build_task_representation(
                tasks[0].task_id, guard_factory(splits[tasks[0].task_id]),
                config={"encoder": "token_stats", "max_support_examples": 4},
            )


def test_sentence_encoder_is_refused_with_a_reason(tasks, splits, backbone):
    """Not silently substituted: a second model would need holding constant across arms."""
    with pytest.raises(ManagerError, match="not implemented"):
        _build(tasks[0], splits, backbone, encoder="sentence_encoder")


def test_representation_is_deterministic(tasks, splits, backbone):
    first = _build(tasks[0], splits, backbone)
    second = _build(tasks[0], splits, backbone)
    assert first.embedding == second.embedding


def test_different_tasks_give_different_representations(tasks, splits, backbone):
    a = _build(tasks[0], splits, backbone)
    b = _build(tasks[1], splits, backbone)
    assert a.embedding != b.embedding


def test_support_statistics_describe_the_set(tasks, splits):
    examples = splits[tasks[0].task_id][SplitName.SUPPORT]
    stats = support_set_statistics(examples)
    assert stats["n_examples"] == len(examples)
    for key in ("prompt_words_mean", "target_words_mean", "type_token_ratio", "vocabulary_size"):
        assert key in stats
    assert 0 < stats["type_token_ratio"] <= 1


def test_hybrid_encoder_concatenates(tasks, splits, backbone):
    plain = _build(tasks[0], splits, backbone, encoder="backbone_mean_hidden")
    hybrid = _build(tasks[0], splits, backbone, encoder="hybrid")
    assert len(hybrid.embedding) > len(plain.embedding)

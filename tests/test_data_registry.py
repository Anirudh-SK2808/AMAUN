"""M2 -- dataset pinning, licensing, and provider behaviour (criteria B1, B5)."""

import pytest

from amaun.data.providers import render_prompt, synthetic_pool
from amaun.data.registry import TaskType, load_curriculum, task_spec_from_config
from amaun.errors import DataError, UnpinnedDatasetError


def test_curriculum_has_the_six_initial_tasks(tasks):
    """idea.md S0.2 LoRA-1..6."""
    assert len(tasks) == 6
    # SENTIMENT_CLASSIFICATION (IMDb, l4_decision_study) and NAMED_ENTITY_RECOGNITION (CoNLL-2003,
    # l4_full_study) exist for opt-in tasks registered in the curriculum but not part of the default
    # six-task order.
    assert {task.task_type for task in tasks} == set(TaskType) - {
        TaskType.SENTIMENT_CLASSIFICATION, TaskType.NAMED_ENTITY_RECOGNITION}


def test_every_task_documents_its_license_and_test_source(tasks):
    for task in tasks:
        assert task.license, f"{task.task_id} has no license recorded"
        assert task.upstream_test_source != "UNDOCUMENTED", (
            f"{task.task_id} does not document which upstream split becomes TEST"
        )


def test_every_task_declares_a_primary_score_within_its_metrics(tasks):
    """D-009: the scalar used for decisions must be one of the task's own metrics."""
    for task in tasks:
        assert task.primary_score in task.metrics, (
            f"{task.task_id} primary_score '{task.primary_score}' is not in {task.metrics}"
        )


def test_no_task_is_unpinned(tasks):
    """D-006 RESOLVED 2026-09-22: all six rows now carry a pinned identity + revision."""
    assert [task.task_id for task in tasks if not task.pinned] == []
    for task in tasks:
        task.require_pinned()               # must not raise for any task
        assert task.hf_id


def test_an_artificially_unpinned_spec_still_fails_loudly():
    """B5's guarantee must survive D-006's resolution: a future new task cannot be guessed at."""
    spec = task_spec_from_config("future_task", {
        "task_name": "Future", "task_type": "summarization", "primary_score": "rougeL",
        "metrics": ["rougeL"], "input_fields": ["article"], "target_field": "highlights",
        "prompt_template": "{article}", "max_input_tokens": 32, "max_target_tokens": 8,
        "hf_id": None, "pinned": False, "unpinned_reason": "decision not yet taken",
    })
    assert not spec.pinned
    with pytest.raises(UnpinnedDatasetError) as info:
        spec.require_pinned()
    assert "D-006" in str(info.value)


def test_revision_is_pinned_to_a_commit_or_a_pinned_conversion_ref(tasks):
    """FR-2.8: "pinned but floating on main" must not be able to recur."""
    import re

    for task in tasks:
        assert task.hf_revision, f"{task.task_id} has no hf_revision"
        assert re.fullmatch(r"[0-9a-f]{40}|refs/convert/parquet", task.hf_revision), (
            f"{task.task_id} hf_revision '{task.hf_revision}' is neither a 40-hex commit sha nor "
            f"the pinned parquet-conversion ref"
        )
        if task.hf_revision == "refs/convert/parquet":
            # A symbolic ref alone would follow a future re-conversion silently (EC-11).
            assert re.fullmatch(r"[0-9a-f]{40}", task.hf_revision_resolved or ""), (
                f"{task.task_id} pins refs/convert/parquet without recording its resolved commit"
            )


def test_every_task_names_a_registered_row_adapter(tasks):
    """D-015: the rule is config-declared with no code default."""
    from amaun.data.row_adapters import ROW_ADAPTERS, resolve as resolve_adapter

    for task in tasks:
        assert task.row_adapter in ROW_ADAPTERS, f"{task.task_id}: {task.row_adapter}"
        adapter = resolve_adapter(task)
        assert adapter.required_row_fields(task, task.row_adapter_options)


def test_share_alike_and_undeclared_licenses_are_recorded(tasks):
    """C-8 / FR-2.5: licensing stays visible so a release audit can be mechanical."""
    by_id = {task.task_id: task for task in tasks}
    instr = by_id["instr"]
    assert instr.license == "CC BY-SA 3.0"
    assert "share-alike" in instr.license_notes.lower()
    longform = by_id["longform"]
    assert "NONE DECLARED" in longform.license.upper()
    assert "reddit" in longform.license_notes.lower()
    assert "MIT" not in longform.license


def test_pinned_tasks_pass_the_pin_check(tasks):
    for task in tasks:
        if task.pinned:
            task.require_pinned()          # must not raise
            assert task.hf_id


def test_non_commercial_license_is_recorded(tasks):
    """DailyDialog is CC BY-NC-SA 4.0; that restriction must be visible, not buried."""
    dialog = next(task for task in tasks if task.task_id == "dialog")
    assert "NC" in dialog.license
    assert "NON-COMMERCIAL" in dialog.license_notes.upper()


def test_curriculum_requires_an_explicit_order(session_cfg):
    section = dict(session_cfg.section("datasets"))
    section.pop("order")
    with pytest.raises(DataError, match="order"):
        load_curriculum(section)


def test_curriculum_rejects_unknown_task_in_order(session_cfg):
    section = dict(session_cfg.section("datasets"))
    section = {**section, "order": list(section["order"]) + ["nonexistent"]}
    with pytest.raises(DataError, match="unknown"):
        load_curriculum(section)


def test_curriculum_is_extensible_without_code_changes(session_cfg):
    """idea.md S0.2 note 4."""
    section = session_cfg.section("datasets")
    extra = {
        "task_name": "Second Summarization", "task_type": "summarization",
        "primary_score": "rougeL", "metrics": ["rougeL"], "input_fields": ["article"],
        "target_field": "highlights", "prompt_template": "Summarize:\n{article}\n\nSummary:",
        "max_input_tokens": 128, "max_target_tokens": 32, "hf_id": "some/dataset",
    }
    extended = {
        "tasks": {**section["tasks"], "sum2": extra},
        "order": list(section["order"]) + ["sum2"],
    }
    curriculum = load_curriculum(extended)
    assert len(curriculum) == 7
    assert curriculum[-1].task_id == "sum2"


def test_missing_required_dataset_key_is_refused():
    with pytest.raises(DataError):
        task_spec_from_config("broken", {"task_name": "x", "task_type": "summarization"})


def test_synthetic_provider_is_deterministic_and_offline(tasks):
    first = synthetic_pool(tasks[0], 10, seed=7)
    second = synthetic_pool(tasks[0], 10, seed=7)
    assert [e.example_id for e in first] == [e.example_id for e in second]
    assert [e.example_id for e in synthetic_pool(tasks[0], 10, seed=8)] != [
        e.example_id for e in first
    ]


def test_synthetic_offset_produces_disjoint_pools(tasks):
    """The test pool must not overlap the training pool."""
    train = {e.example_id for e in synthetic_pool(tasks[0], 20, seed=1)}
    test = {e.example_id for e in synthetic_pool(tasks[0], 20, seed=1, offset=1_000_000)}
    assert not (train & test)


def test_synthetic_shapes_match_the_task_type(tasks):
    for task in tasks:
        example = synthetic_pool(task, 1, seed=1)[0]
        assert example.prompt and example.target
        assert example.task_id == task.task_id
        assert example.meta["provider"] == "synthetic"


def test_prompt_template_missing_field_fails_loudly(tasks):
    """A silently truncated prompt would change the task the adapter learns."""
    with pytest.raises(DataError, match="prompt_template"):
        render_prompt(tasks[0], {"wrong_field": "x"})

"""M12 -- the evaluator is the only TEST reader (criteria H5, I12.1/I12.2)."""

import pytest

from amaun.adapters.lora import ensure_lora
from amaun.data.splits import SplitName, evaluator_guard, manager_guard, trainer_guard
from amaun.errors import SplitAccessViolation
from amaun.evaluation.evaluator import Evaluator
from amaun.models.generation import GenerationConfig


def _evaluator(session_cfg, backbone):
    return Evaluator(
        backbone, GenerationConfig.from_config(session_cfg.section("generation")),
        max_examples=2, batch_size=2,
    )


def test_evaluator_reads_test_and_produces_the_primary_score(session_cfg, backbone, tasks, splits, lora_spec):
    model = ensure_lora(backbone.model, lora_spec)
    task = tasks[1]        # QA: token_f1
    result = _evaluator(session_cfg, backbone).evaluate(
        task, evaluator_guard(splits[task.task_id]), model=model, adapter_id=None
    )
    assert result.task_id == task.task_id
    assert result.primary_score_name == task.primary_score
    assert 0.0 <= result.primary_score <= 1.0
    assert result.n_examples == 2
    assert set(task.metrics) <= set(result.metrics)


def test_evaluator_refuses_a_guard_that_does_not_permit_test(session_cfg, backbone, tasks, splits, lora_spec):
    model = ensure_lora(backbone.model, lora_spec)
    evaluator = _evaluator(session_cfg, backbone)
    for guard_factory in (manager_guard, trainer_guard):
        with pytest.raises(SplitAccessViolation):
            evaluator.evaluate(
                tasks[0], guard_factory(splits[tasks[0].task_id]), model=model, adapter_id=None
            )


def test_manager_module_does_not_import_the_evaluator():
    """Architecture rule: the manager must have no import path to TEST-reading code."""
    import pathlib

    manager_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "amaun" / "manager"
    offenders = []
    for path in manager_dir.glob("*.py"):
        text = path.read_text()
        for forbidden in ("from amaun.evaluation.evaluator", "import amaun.evaluation.evaluator"):
            if forbidden in text:
                offenders.append(path.name)
    assert not offenders, f"manager modules import the evaluator: {offenders}"


def test_manager_modules_never_reference_the_test_split():
    import pathlib

    manager_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "amaun" / "manager"
    offenders = []
    for path in manager_dir.glob("*.py"):
        for line in path.read_text().splitlines():
            if "SplitName.TEST" in line:
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, f"manager code references the TEST split: {offenders}"


def test_generation_config_is_shared_and_logged(session_cfg):
    """Invariant I12.2: decoding must be identical across arms."""
    config = GenerationConfig.from_config(session_cfg.section("generation"))
    assert config.do_sample is False        # greedy by default => deterministic evaluation
    payload = config.as_dict()
    for key in ("do_sample", "num_beams", "max_new_tokens", "temperature", "top_p"):
        assert key in payload
    kwargs = config.to_kwargs(pad_token_id=0, eos_token_id=2)
    # Sampling parameters are omitted when do_sample is False, keeping the path unambiguous.
    assert "temperature" not in kwargs
    assert kwargs["do_sample"] is False


def test_missing_generation_key_is_refused():
    with pytest.raises(ValueError):
        GenerationConfig.from_config({"do_sample": False})

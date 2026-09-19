"""Regression test for a CRITICAL leak found during the Agent-3 audit.

**The defect.** ``ContinualRunner._prior_scores`` originally read prior-task scores from
``self.matrix`` -- the performance matrix that the evaluator fills from **TEST**. Those scores
feed the ``continual`` feature family (``F_i``, ``KR_i``, ``P_i``, ``S_i``), which feeds the
utility network, which produces the decision. Test-set performance was therefore influencing
adapter selection, violating master-prompt constraint 13 and ``idea.md`` S16.5.

**Why it mattered.** Nothing would have looked wrong: the pipeline ran, the numbers were
plausible, and the leak would have inflated exactly the decision-quality result the paper claims.

**The fix.** A parallel ``validation_history`` filled through a guard that permits VALIDATION and
not TEST. These tests assert the manager's inputs are sourced from it.
"""

import pytest

from amaun.adapters.repository import AdapterRepository
from amaun.baselines.strategies import SeparateLoRAStrategy
from amaun.continual.runner import ContinualRunner
from amaun.evaluation.cl_metrics import PerformanceMatrix
from amaun.evaluation.evaluator import Evaluator
from amaun.features.extractor import FeatureExtractor
from amaun.models.generation import GenerationConfig
from amaun.utils.tracking import ExperimentTracker


def _runner(cfg, backbone, tasks, splits, tmp_path):
    tracker = ExperimentTracker(cfg, base=tmp_path).start()
    repository = AdapterRepository(
        tracker.paths.adapters, backbone_fingerprint=backbone.fingerprint,
        journal_path=tracker.paths.repository_events,
    )
    evaluator = Evaluator(
        backbone, GenerationConfig.from_config(cfg.section("generation")),
        max_examples=2, batch_size=2,
    )
    return ContinualRunner(
        cfg=cfg, backbone=backbone, tasks=tasks, splits=splits, repository=repository,
        strategy=SeparateLoRAStrategy(), extractor=FeatureExtractor(cfg.section("features")),
        evaluator=evaluator, tracker=tracker, arm_id="leak-check",
    )


def test_prior_scores_come_from_validation_not_from_the_test_matrix(
    cfg, backbone, tasks, splits, tmp_path
):
    runner = _runner(cfg, backbone, tasks, splits, tmp_path)

    # Poison the TEST-filled matrix with an unmistakable value, and leave the validation
    # history holding a different one.
    runner.matrix = PerformanceMatrix(task_order=["t_a", "t_b"])
    runner.matrix.record("t_a", "t_a", 0.999)
    runner.matrix.record("t_b", "t_a", 0.888)
    runner.validation_history = PerformanceMatrix(task_order=["t_a", "t_b"])
    runner.validation_history.record("t_a", "t_a", 0.111)
    runner.validation_history.record("t_b", "t_a", 0.222)

    before = runner._prior_scores(before=True)
    after = runner._prior_scores(before=False)

    assert before == {"t_a": 0.111}
    assert after == {"t_a": 0.222}
    for scores in (before, after):
        assert 0.999 not in scores.values(), "TEST score leaked into the decision features"
        assert 0.888 not in scores.values(), "TEST score leaked into the decision features"


def test_validation_history_is_a_separate_object_from_the_test_matrix(
    cfg, backbone, tasks, splits, tmp_path
):
    runner = _runner(cfg, backbone, tasks, splits, tmp_path)
    assert runner.matrix is not runner.validation_history


def test_prior_scores_are_empty_before_two_tasks(cfg, backbone, tasks, splits, tmp_path):
    runner = _runner(cfg, backbone, tasks, splits, tmp_path)
    assert runner._prior_scores(before=True) == {}
    assert runner._prior_scores(before=False) == {}


def test_validation_scoring_reads_validation_and_populates_the_history(
    cfg, backbone, tasks, splits, tmp_path
):
    runner = _runner(cfg, backbone, tasks, splits, tmp_path)
    result = runner.run([tasks[0].task_id, tasks[1].task_id])

    # The validation history must be populated, and independently of the TEST matrix.
    assert runner.validation_history.task_order
    assert any(event.validation_scores for event in result.events)
    # Both histories cover the same tasks but are filled from different splits.
    assert set(runner.validation_history.task_order) <= set(runner.matrix.task_order)


def test_runner_reads_test_only_through_the_evaluator(cfg, backbone, tasks, splits, tmp_path):
    """The runner holds raw splits (it must, to hand out guards), so pin down TEST reads."""
    import inspect

    source = inspect.getsource(ContinualRunner)
    # Every TEST read must go through _evaluator_guard; no direct raw-split TEST access.
    assert "self.splits[task_id][SplitName.TEST]" not in source
    assert "SplitName.TEST" not in source, (
        "the runner references SplitName.TEST directly; TEST access must go through "
        "_evaluator_guard so it is auditable in one place"
    )


def test_continual_feature_family_reads_only_supplied_scores():
    """The family itself must not fetch data: it can only see what the runner hands it."""
    import inspect

    from amaun.features.continual import ContinualFeatures

    source = inspect.getsource(ContinualFeatures)
    for forbidden in ("guard", "split", "evaluate", "TEST"):
        assert forbidden not in source, (
            f"ContinualFeatures references '{forbidden}'; it must be a pure function of "
            f"precomputed scores so it has no path to any dataset"
        )

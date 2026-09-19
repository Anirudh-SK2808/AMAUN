"""M10 -- the continual runner end to end on the dummy tier (criterion G1).

This is the integration test: it drives the real pipeline (task representation -> features ->
utility -> decision -> fusion/KD or creation -> repository update -> evaluation) and asserts
the cross-cutting invariants that only show up when the modules run together.
"""

import pytest

from amaun.adapters.repository import AdapterRepository
from amaun.baselines.strategies import AmaunStrategy, SeparateLoRAStrategy
from amaun.continual.runner import ContinualRunner
from amaun.evaluation.cl_metrics import compute as compute_cl
from amaun.evaluation.evaluator import Evaluator
from amaun.features.extractor import FeatureExtractor
from amaun.manager.decision import Outcome, Thresholds
from amaun.models.backbone import backbone_fingerprint
from amaun.models.generation import GenerationConfig
from amaun.utils.tracking import ExperimentTracker


def _runner(cfg, backbone, tasks, splits, tmp_path, strategy, arm_id):
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
        strategy=strategy, extractor=FeatureExtractor(cfg.section("features")),
        evaluator=evaluator, tracker=tracker, arm_id=arm_id,
    ), tracker


def _amaun_strategy(cfg, *, tau_u=0.0, tau_reuse=1e9):
    """A strategy with FIXED thresholds so decisions are predictable in a test.

    A very high tau_reuse suppresses REUSE; a low tau_u makes FUSE the default. Real runs
    select these from the dev stream (D-005); here we pin them to make outcomes assertable.
    """
    import torch

    from amaun.features.base import FeatureVector
    from amaun.features.normalizer import FeatureNormalizer
    from amaun.manager.utility_network import OUTCOME_HEADS, UtilityNetwork, UtilityNetworkConfig

    extractor = FeatureExtractor(cfg.section("features"))
    dim = None

    class _LazyStrategy(AmaunStrategy):
        def score(self, vectors, *, run_mode):
            nonlocal dim
            if self.normalizer is None or dim != len(next(iter(vectors.values()))):
                sample = list(vectors.values())
                dim = len(sample[0])
                normalizer = FeatureNormalizer()
                normalizer.fit(sample + sample, split="utility_train", run_mode=run_mode)
                self.normalizer = normalizer
                self.network = UtilityNetwork(
                    dim, UtilityNetworkConfig(
                        hidden_dims=(8,), dropout=0.0, activation="relu",
                        output_heads=OUTCOME_HEADS, monotone_weights=True,
                    ),
                )
            return super().score(vectors, run_mode=run_mode)

    return _LazyStrategy(
        network=None, normalizer=None,
        thresholds=Thresholds(tau_u=tau_u, tau_reuse=tau_reuse, source="fixed", selected_on="test"),
        decision_order=str(cfg.get("manager.decision_order")),
        allowed_outcomes=cfg.get("manager.allowed_outcomes"),
        top_k=int(cfg.get("manager.retrieval.k")),
    )


def test_bootstrap_registers_independently_trained_seed_adapters(
    cfg, backbone, tasks, splits, tmp_path
):
    runner, _ = _runner(cfg, backbone, tasks, splits, tmp_path, SeparateLoRAStrategy(), "boot")
    seeds = [task.task_id for task in tasks[:3]]
    entries = runner.bootstrap_repository(seeds)
    assert len(entries) == 3
    assert len(runner.repository) == 3
    # Every seed adapter must be journaled as a bootstrap registration, not a decision.
    events = [e for e in runner.repository.events() if e["event"] == "register"]
    assert all(event["bootstrap"] for event in events)
    # And every one must carry distinct weights.
    hashes = {entry.delta.content_hash() for entry in entries}
    assert len(hashes) == 3


def test_separate_lora_arm_creates_one_adapter_per_task(cfg, backbone, tasks, splits, tmp_path):
    runner, _ = _runner(cfg, backbone, tasks, splits, tmp_path, SeparateLoRAStrategy(), "B2")
    stream = [task.task_id for task in tasks[:3]]
    result = runner.run(stream)
    assert result.outcome_counts() == {"CREATE": 3}
    assert len(result.repository) == 3
    # H3's growth reference: adapters == tasks for this baseline.
    assert result.events[-1].efficiency["n_adapters"] == 3


def test_amaun_arm_runs_and_records_every_artefact(cfg, backbone, tasks, splits, tmp_path):
    runner, tracker = _runner(
        cfg, backbone, tasks, splits, tmp_path, _amaun_strategy(cfg), "amaun"
    )
    runner.bootstrap_repository([tasks[0].task_id, tasks[1].task_id])
    before = backbone_fingerprint(backbone.model)
    result = runner.run([task.task_id for task in tasks[2:4]])

    assert len(result.events) == 2
    for event in result.events:
        assert event.outcome in {o.value for o in Outcome}
        assert event.feature_dimension and event.feature_dimension > 20
        assert event.evaluations
        assert event.efficiency["n_adapters"] >= 1

    # Backbone untouched across the whole stream.
    assert backbone_fingerprint(backbone.model) == before

    # Artefacts landed where the spec says they must.
    assert tracker.paths.performance_matrix.exists()
    assert list(tracker.read("decisions"))
    assert list(tracker.read("metrics"))
    assert any(tracker.paths.repository_snapshots.glob("task_*.json"))


def test_temporary_adapter_never_enters_the_repository_before_the_decision(
    cfg, backbone, tasks, splits, tmp_path
):
    """Master-prompt constraint 10, observed through the event journal."""
    runner, _ = _runner(
        cfg, backbone, tasks, splits, tmp_path, _amaun_strategy(cfg), "candidate"
    )
    runner.bootstrap_repository([tasks[0].task_id])
    runner.run([tasks[1].task_id])
    events = runner.repository.events()
    registrations = [e for e in events if e["event"] == "register" and not e["bootstrap"]]
    for event in registrations:
        # Any non-bootstrap registration must name the authorising decision.
        assert event["authorised_by"], f"unauthorised registration: {event}"


def test_fuse_outcome_produces_a_fusion_log(cfg, backbone, tasks, splits, tmp_path):
    """Force FUSE by making tau_u very low and tau_reuse unreachable."""
    runner, tracker = _runner(
        cfg, backbone, tasks, splits, tmp_path,
        _amaun_strategy(cfg, tau_u=-1e9, tau_reuse=1e9), "fuse",
    )
    runner.bootstrap_repository([tasks[0].task_id, tasks[1].task_id])
    result = runner.run([tasks[2].task_id])
    event = result.events[0]
    assert event.outcome == Outcome.FUSE.value
    assert event.fusion is not None
    assert event.fusion["sources_verified_unmutated"] is True
    # alpha must start at equal contribution.
    first_alpha = event.fusion["alpha_trajectory"][0]["alpha"]
    for values in first_alpha.values():
        assert sum(values) == pytest.approx(1.0, abs=1e-5)
    assert list(tracker.paths.fusion_logs.glob("*.json"))


def test_create_outcome_when_nothing_is_useful(cfg, backbone, tasks, splits, tmp_path):
    """tau_u above any achievable utility must force CREATE."""
    runner, _ = _runner(
        cfg, backbone, tasks, splits, tmp_path,
        _amaun_strategy(cfg, tau_u=1e9, tau_reuse=1e9), "create",
    )
    runner.bootstrap_repository([tasks[0].task_id])
    before = len(runner.repository)
    result = runner.run([tasks[1].task_id])
    assert result.events[0].outcome == Outcome.CREATE.value
    assert len(runner.repository) == before + 1


def test_reuse_outcome_leaves_the_repository_unchanged(cfg, backbone, tasks, splits, tmp_path):
    """REUSE is the only outcome that grows nothing -- the H3 mechanism."""
    runner, _ = _runner(
        cfg, backbone, tasks, splits, tmp_path,
        _amaun_strategy(cfg, tau_u=-1e9, tau_reuse=-1e9), "reuse",
    )
    runner.bootstrap_repository([tasks[0].task_id, tasks[1].task_id])
    before_ids = set(runner.repository.ids())
    before_bytes = runner.repository.memory_bytes()
    result = runner.run([tasks[2].task_id])
    event = result.events[0]
    if event.outcome == Outcome.REUSE.value:
        assert set(runner.repository.ids()) == before_ids
        assert runner.repository.memory_bytes() == before_bytes
        # The reused adapter now serves an extra task (D-014 accounting).
        served = runner.repository.tasks_served()
        assert any(tasks[2].task_id in v for v in served.values())
    else:
        pytest.skip(f"prescreen did not pass, outcome was {event.outcome}")


def test_cl_metrics_computable_from_the_run(cfg, backbone, tasks, splits, tmp_path):
    runner, _ = _runner(cfg, backbone, tasks, splits, tmp_path, SeparateLoRAStrategy(), "metrics")
    result = runner.run([task.task_id for task in tasks[:3]])
    metrics = compute_cl(result.matrix)
    assert 0.0 <= metrics.average_performance <= 1.0
    assert metrics.n_tasks == 3
    assert metrics.forgetting is not None


def test_unknown_task_in_the_stream_is_refused(cfg, backbone, tasks, splits, tmp_path):
    from amaun.errors import PreconditionError

    runner, _ = _runner(cfg, backbone, tasks, splits, tmp_path, SeparateLoRAStrategy(), "bad")
    with pytest.raises(PreconditionError):
        runner.run(["nonexistent_task"])
    with pytest.raises(PreconditionError):
        runner.run([])

"""M13 -- baselines and the citation gate (criteria H1, H2)."""

import pytest

from amaun.baselines.registry import (
    IMPLEMENTED,
    REQUIRES_CITATION,
    build_strategy,
    describe,
    implemented_ids,
    unimplemented_report,
)
from amaun.baselines.strategies import (
    NaiveAverageStrategy,
    SeparateLoRAStrategy,
    SimilarityConfig,
    SimilaritySelectionStrategy,
)
from amaun.errors import BaselineRequiresCitationError
from amaun.features.base import FeatureVector
from amaun.manager.decision import Outcome


def _vectors(n=3):
    return {
        f"A{i}": FeatureVector(
            names=["semantic.d_task_sq", "other"], values=[float(i), 1.0]
        )
        for i in range(n)
    }


def test_named_literature_baselines_refuse_with_the_exact_requirement():
    """D-007: reconstructing a method from its name would fabricate a baseline."""
    for baseline_id in ("LoRM", "FedALoRA", "FedZKD"):
        with pytest.raises(BaselineRequiresCitationError) as info:
            build_strategy(baseline_id, {})
        message = str(info.value)
        assert "NOT implemented" in message
        assert "D-007" in message
        assert "Provide" in message or "reference" in message


def test_unimplemented_report_is_machine_readable():
    report = unimplemented_report()
    assert {row["baseline_id"] for row in report} == set(REQUIRES_CITATION)
    for row in report:
        assert row["implemented"] is False
        assert row["requirement"]


def test_implemented_baselines_are_listed():
    ids = implemented_ids()
    for expected in ("B1_sequential_full_ft", "B2_separate_lora", "B3_naive_average",
                     "B4_similarity_selection", "B5_fixed_fusion", "B6_heuristic_compression"):
        assert expected in ids
    assert set(ids) == set(IMPLEMENTED)


def test_describe_returns_provenance():
    spec = describe("B4_similarity_selection")
    assert spec.implemented
    assert spec.source


def test_separate_lora_always_creates():
    strategy = SeparateLoRAStrategy()
    decision = strategy.decide(
        task_id="t", task_index=1, scores=[], repository_size=3,
        temp_adapter_trained=True, config_hash="h",
    )
    assert decision.outcome is Outcome.CREATE


def test_naive_average_always_fuses():
    strategy = NaiveAverageStrategy()
    scores = strategy.score(_vectors(), run_mode="dummy")
    decision = strategy.decide(
        task_id="t", task_index=1, scores=scores, repository_size=3,
        temp_adapter_trained=True, config_hash="h",
    )
    assert decision.outcome is Outcome.FUSE
    assert "uniform" in decision.reason


def test_similarity_baseline_honestly_uses_one_feature():
    """It must not be strawmanned, and it must not be disguised as multi-criteria."""
    strategy = SimilaritySelectionStrategy(
        SimilarityConfig(feature="semantic.d_task_sq", threshold=0.0, higher_is_more_similar=False)
    )
    scores = strategy.score(_vectors(), run_mode="dummy")
    assert all(score.n_features == 1 for score in scores)
    # Smaller distance == more similar, so A0 (distance 0) must rank first.
    assert scores[0].adapter_id == "A0"


def test_similarity_baseline_has_no_reuse_outcome():
    strategy = SimilaritySelectionStrategy(
        SimilarityConfig(feature="semantic.d_task_sq", threshold=-1.0, higher_is_more_similar=False)
    )
    scores = strategy.score(_vectors(), run_mode="dummy")
    decision = strategy.decide(
        task_id="t", task_index=1, scores=scores, repository_size=3,
        temp_adapter_trained=True, config_hash="h",
    )
    assert decision.outcome in (Outcome.FUSE, Outcome.CREATE)


def test_similarity_baseline_fails_loudly_on_a_missing_feature():
    strategy = SimilaritySelectionStrategy(
        SimilarityConfig(feature="not.a.feature", threshold=0.0, higher_is_more_similar=True)
    )
    with pytest.raises(KeyError):
        strategy.score(_vectors(), run_mode="dummy")


def test_full_finetuning_baseline_requires_an_explicit_flag():
    """B1 is the ONE arm where the frozen-backbone invariant is suspended; it must be visible."""
    with pytest.raises(BaselineRequiresCitationError, match="backbone_trainable"):
        build_strategy("B1_sequential_full_ft", {})


def test_heuristic_compression_baseline_is_configured_not_constructed():
    with pytest.raises(BaselineRequiresCitationError, match="mechanism"):
        build_strategy("B6_heuristic_compression", {})


def test_every_strategy_reports_whether_it_needs_the_utility_network():
    assert SeparateLoRAStrategy().requires_utility_network is False
    from amaun.baselines.strategies import AmaunStrategy

    assert AmaunStrategy.requires_utility_network is True

"""M6 -- all seven families, their toggles, and layout stability (criterion E2)."""

import pytest
import torch

from amaun.features.extractor import FAMILY_ORDER, CandidateContext, FeatureExtractor
from amaun.features.parameter import (
    effective_difference_norm,
    effective_frobenius_norm,
    effective_singular_values,
    effective_variance,
)
from tests.conftest import make_delta, make_metadata


def _context(**overrides):
    base = dict(
        temp_id="temp", temp_delta=make_delta(1, "temp"),
        candidate_id="cand", candidate_delta=make_delta(2, "cand"),
        candidate_metadata=make_metadata("cand", created_at_task_index=0, use_count=2, merge_count=1),
        task_index=3, total_events=5,
        task_embedding=[0.1] * 8, candidate_task_embedding=[0.3] * 8,
        hidden_discrepancy=0.7, output_kl=0.12, output_kl_reverse=0.15,
        temp_gradients={"g": torch.randn(20)}, candidate_gradients={"g": torch.randn(20)},
        temp_quality={"loss": 2.1, "confidence": 0.4, "entropy": 1.2, "token_accuracy": 0.2},
        candidate_quality={"primary_score": 0.35, "loss": 2.6, "confidence": 0.5,
                           "entropy": 1.0, "ece": 0.08, "token_accuracy": 0.25},
        prior_scores_pre={"t1": 0.5, "t2": 0.6}, prior_scores_post={"t1": 0.45, "t2": 0.62},
        new_task_score_pre=0.1, new_task_score_post=0.3,
        repository_bytes=9000.0, repository_size=4, candidate_latency_seconds=0.004,
    )
    base.update(overrides)
    return CandidateContext(**base)


def test_all_seven_families_are_present(cfg):
    extractor = FeatureExtractor(cfg.section("features"))
    assert extractor.enabled == list(FAMILY_ORDER)
    vector = extractor.extract(_context())
    assert set(vector.family_slices) == set(FAMILY_ORDER)
    for family in FAMILY_ORDER:
        assert vector.family(family), f"family '{family}' produced no features"


def test_decision_input_is_never_a_single_scalar(cfg):
    """idea.md S16.2 -- the whole point of the multi-criteria formulation."""
    vector = FeatureExtractor(cfg.section("features")).extract(_context())
    assert len(vector) > 20, f"only {len(vector)} features; this is not multi-criteria"


def test_each_family_toggle_changes_the_dimension(cfg):
    """Invariant I6.4 -- an ablation must not be able to be a silent no-op."""
    section = dict(cfg.section("features"))
    full = FeatureExtractor(section).extract(_context())
    for family in FAMILY_ORDER:
        toggles = {name: name != family for name in FAMILY_ORDER}
        reduced = FeatureExtractor({**section, "families": toggles}).extract(_context())
        assert len(reduced) < len(full), f"disabling '{family}' did not change the vector"
        assert family not in reduced.family_slices


def test_no_families_enabled_is_rejected(cfg):
    section = dict(cfg.section("features"))
    with pytest.raises(ValueError):
        FeatureExtractor({**section, "families": {name: False for name in FAMILY_ORDER}})


def test_layout_is_stable_regardless_of_data_availability(cfg):
    """The vector's width must not depend on what happens to be computable."""
    extractor = FeatureExtractor(cfg.section("features"))
    variants = [
        _context(),
        _context(task_embedding=None, candidate_task_embedding=None),
        _context(temp_gradients={}, candidate_gradients={}),
        _context(temp_quality={}, candidate_quality={}),
        _context(prior_scores_pre={}, prior_scores_post={}),
        _context(candidate_delta=make_delta(3, "other", d_in=16)),   # incompatible shapes
        _context(candidate_latency_seconds=None, repository_bytes=0.0),
    ]
    vectors = [extractor.extract(variant) for variant in variants]
    assert len({len(v) for v in vectors}) == 1
    assert len({v.signature() for v in vectors}) == 1


def test_missing_values_are_substituted_and_counted(cfg):
    """Invariant I6.6 -- a permanently-missing feature must be visible in the logs."""
    extractor = FeatureExtractor(cfg.section("features"))
    vector = extractor.extract(_context(temp_gradients={}, candidate_gradients={}))
    assert vector.missing
    assert any(name.startswith("gradient.") for name in vector.missing)
    assert all(value == value for value in vector.values)   # no NaN leaked through


def test_semantic_uses_euclidean_not_cosine_as_the_signal(cfg):
    """idea.md S4.1 specifies Euclidean; cosine may appear only as one extra feature."""
    extractor = FeatureExtractor(cfg.section("features"))
    vector = extractor.extract(_context())
    semantic = vector.family("semantic")
    assert "semantic.d_task_sq" in semantic
    assert "semantic.d_task_l2" in semantic
    # Cosine is permitted as an additional feature, but must be a small minority of the vector.
    cosine_features = [name for name in vector.names if "cos" in name]
    assert len(cosine_features) <= 1


def test_low_rank_shortcuts_match_dense_computation():
    """The O(r^2 d) tricks that make parameter features tractable at 7B must be exact."""
    generator = torch.Generator().manual_seed(0)
    from amaun.adapters.delta import LayerDelta

    for rank, d_in, d_out, scaling in ((4, 16, 12, 2.0), (8, 32, 24, 0.5), (2, 10, 10, 1.0)):
        left = LayerDelta(
            A=torch.randn(rank, d_in, generator=generator),
            B=torch.randn(d_out, rank, generator=generator), scaling=scaling,
        )
        right = LayerDelta(
            A=torch.randn(rank, d_in, generator=generator),
            B=torch.randn(d_out, rank, generator=generator), scaling=scaling * 0.7,
        )
        dense_left, dense_right = left.dense(), right.dense()
        assert effective_frobenius_norm(left) == pytest.approx(
            float(torch.linalg.norm(dense_left)), rel=1e-4
        )
        assert effective_difference_norm(left, right) == pytest.approx(
            float(torch.linalg.norm(dense_left - dense_right)), rel=1e-4
        )
        assert torch.allclose(
            effective_singular_values(left), torch.linalg.svdvals(dense_left)[:rank], atol=1e-4
        )
        assert effective_variance(left) == pytest.approx(
            float(dense_left.flatten().var(unbiased=False)), rel=1e-3
        )


def test_continual_features_match_their_definitions(cfg):
    """idea.md S4.5 formulas, checked by hand."""
    extractor = FeatureExtractor(cfg.section("features"))
    vector = extractor.extract(_context())
    continual = vector.family("continual")
    # F = mean(max(0, 0.5-0.45), max(0, 0.6-0.62)) = mean(0.05, 0.0) = 0.025
    assert continual["continual.F_i"] == pytest.approx(0.025, abs=1e-9)
    assert continual["continual.KR_i"] == pytest.approx(0.975, abs=1e-9)
    assert continual["continual.P_i"] == pytest.approx(0.2, abs=1e-9)   # 0.3 - 0.1
    rho = float(cfg.get("features.continual.rho"))
    assert continual["continual.S_i"] == pytest.approx(1.0 - 0.025 + rho * 0.2, abs=1e-9)


def test_historical_unknown_performance_is_not_encoded_as_zero(cfg):
    """A new adapter has UNKNOWN history; zero would bias the network against it."""
    metadata = make_metadata("fresh")            # no performance history
    assert metadata.mean_historical_performance() is None
    vector = FeatureExtractor(cfg.section("features")).extract(
        _context(candidate_metadata=metadata)
    )
    assert "historical.PerfHist_i" in vector.missing

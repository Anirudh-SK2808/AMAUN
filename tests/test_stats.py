"""M12 -- statistical protocol (criterion H7, D-012)."""

import pytest

from amaun.errors import InsufficientSeedsError
from amaun.evaluation.stats import MIN_SEEDS_FOR_COMPARISON, aggregate_seeds, compare_arms


def test_single_seed_reports_no_dispersion():
    aggregate = aggregate_seeds([0.5])
    assert aggregate.std is None
    assert "no dispersion" in aggregate.format()


def test_multi_seed_reports_mean_and_std():
    aggregate = aggregate_seeds([0.50, 0.52, 0.48])
    assert aggregate.mean == pytest.approx(0.5)
    assert aggregate.std is not None and aggregate.std > 0
    assert aggregate.n == 3


def test_comparison_below_minimum_seeds_is_refused():
    """A one-seed difference is not evidence and must not be presentable as a gain."""
    a = {("t0", 0): 0.7, ("t1", 0): 0.8}
    b = {("t0", 0): 0.5, ("t1", 0): 0.6}
    with pytest.raises(InsufficientSeedsError, match="minimum"):
        compare_arms("A", "B", a, b, n_bootstrap=200)


def test_real_difference_is_detected_across_seeds():
    a = {(f"t{t}", s): 0.60 + 0.01 * t + 0.003 * s for t in range(4) for s in range(3)}
    b = {(f"t{t}", s): 0.50 + 0.01 * t + 0.003 * s for t in range(4) for s in range(3)}
    result = compare_arms("A", "B", a, b, n_bootstrap=2000, seed=1)
    assert result.mean_difference == pytest.approx(0.10, abs=1e-6)
    assert result.significant
    assert result.n_seeds == 3
    assert "higher" in result.interpretation


def test_no_difference_is_reported_as_no_difference():
    base = {(f"t{t}", s): 0.5 + 0.02 * t for t in range(5) for s in range(3)}
    noisy = {key: value + (0.01 if (hash(key) % 2) else -0.01) for key, value in base.items()}
    result = compare_arms("A", "B", base, noisy, n_bootstrap=2000, seed=2)
    assert not result.significant or abs(result.mean_difference) < 0.01
    if not result.significant:
        assert "contains zero" in result.interpretation


def test_effect_size_is_always_reported():
    a = {(f"t{t}", s): 0.6 + 0.01 * s for t in range(4) for s in range(3)}
    b = {(f"t{t}", s): 0.4 + 0.01 * s for t in range(4) for s in range(3)}
    result = compare_arms("A", "B", a, b, n_bootstrap=500, seed=3)
    assert result.effect_size_d is not None
    assert 0.0 <= result.p_value <= 1.0


def test_no_shared_pairs_is_refused():
    with pytest.raises(InsufficientSeedsError):
        compare_arms("A", "B", {("t0", 0): 0.5}, {("t1", 1): 0.4}, n_bootstrap=100, min_seeds=1)


def test_minimum_seed_constant_matches_the_decision_register():
    assert MIN_SEEDS_FOR_COMPARISON == 3      # D-012 recommendation

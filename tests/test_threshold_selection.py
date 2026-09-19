"""M7 -- threshold selection provenance (criterion E12, D-005)."""

import pytest

from amaun.errors import ThresholdNotSelectedError
from amaun.manager.threshold import select_thresholds


def test_selection_procedure_is_required():
    with pytest.raises(ThresholdNotSelectedError):
        select_thresholds({})


def test_fixed_source_is_labelled_as_unjustified():
    """An honest but unjustified threshold must say so in its own provenance."""
    thresholds = select_thresholds(
        {"selection": "fixed", "fixed_tau_u": 0.3, "fixed_tau_reuse": 0.7}
    )
    assert thresholds.source == "fixed"
    assert "NOT experimentally justified" in thresholds.notes


def test_fixed_source_requires_both_values():
    with pytest.raises(ThresholdNotSelectedError):
        select_thresholds({"selection": "fixed", "fixed_tau_u": 0.3})


def test_dev_stream_requires_observations():
    """Selecting tau with no data would be inventing it."""
    with pytest.raises(ThresholdNotSelectedError):
        select_thresholds({"selection": "dev_stream"})


def test_dev_stream_separates_beneficial_from_harmful():
    thresholds = select_thresholds(
        {"selection": "dev_stream"},
        observed_utilities=[0.10, 0.15, 0.20, 0.70, 0.80, 0.90],
        beneficial_flags=[False, False, False, True, True, True],
    )
    # The cut must separate the two populations perfectly here.
    assert 0.20 < thresholds.tau_u <= 0.70
    assert thresholds.tau_reuse >= thresholds.tau_u
    assert thresholds.source == "dev_stream"
    assert "balanced accuracy" in thresholds.notes


def test_mismatched_observation_lengths_are_refused():
    with pytest.raises(ThresholdNotSelectedError):
        select_thresholds(
            {"selection": "dev_stream"}, observed_utilities=[0.1, 0.2], beneficial_flags=[True]
        )


def test_one_class_case_falls_back_to_a_stated_quantile():
    thresholds = select_thresholds(
        {"selection": "dev_stream"},
        observed_utilities=[0.2, 0.4, 0.6],
        beneficial_flags=[True, True, True],
    )
    assert thresholds.tau_u == pytest.approx(0.4)      # median
    assert "median" in thresholds.notes


def test_reuse_threshold_is_at_least_the_fuse_threshold():
    thresholds = select_thresholds(
        {"selection": "historical_cv", "reuse_quantile": 0.9},
        observed_utilities=[0.1, 0.5, 0.9],
        beneficial_flags=[False, True, True],
    )
    assert thresholds.tau_reuse >= thresholds.tau_u

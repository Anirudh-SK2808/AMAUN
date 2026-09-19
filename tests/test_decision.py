"""M7.3 -- the REUSE / FUSE / CREATE decision rule (criteria E3, E6-E9)."""

import pytest

from amaun.errors import ManagerError, ThresholdNotSelectedError
from amaun.manager.decision import CandidateScore, DecisionOrder, Outcome, Thresholds, decide


def _score(adapter_id, utility, n_features=54):
    return CandidateScore(
        adapter_id=adapter_id, utility=utility, outcomes={},
        feature_signature="sig", n_features=n_features,
    )


def _thresholds():
    return Thresholds(tau_u=0.2, tau_reuse=0.6, source="fixed", selected_on="config")


def _decide(scores, *, prescreen=True, repository_size=None, allowed=None, top_k=2):
    return decide(
        task_id="t", task_index=1, scores=scores, thresholds=_thresholds(),
        decision_order=DecisionOrder.TRAIN_THEN_DECIDE,
        allowed_outcomes=allowed or ["REUSE", "FUSE", "CREATE"],
        repository_size=repository_size if repository_size is not None else len(scores),
        temp_adapter_trained=True, top_k=top_k, reuse_prescreen_passed=prescreen,
    )


def test_reuse_when_utility_clears_tau_reuse_and_prescreen_passes():
    decision = _decide([_score("A1", 0.9), _score("A2", 0.3)])
    assert decision.outcome is Outcome.REUSE
    assert decision.selected_adapter_ids == ["A1"]


def test_fuse_when_utility_clears_tau_u_only():
    decision = _decide([_score("A1", 0.4), _score("A2", 0.3)])
    assert decision.outcome is Outcome.FUSE
    assert decision.selected_adapter_ids == ["A1", "A2"]   # top_k = 2


def test_create_when_no_candidate_clears_tau_u():
    decision = _decide([_score("A1", 0.05), _score("A2", 0.01)])
    assert decision.outcome is Outcome.CREATE
    assert decision.selected_adapter_ids == []


def test_failed_prescreen_downgrades_reuse_to_fuse():
    """REUSE requires support-set evidence, not just a high predicted utility."""
    decision = _decide([_score("A1", 0.9)], prescreen=False)
    assert decision.outcome is Outcome.FUSE


def test_empty_repository_creates_without_consulting_the_network():
    """Invariant I7.5 / idea.md S5.4 lines 2-4."""
    decision = decide(
        task_id="t", task_index=0, scores=[], thresholds=None,
        decision_order="train_then_decide", allowed_outcomes=["REUSE", "FUSE", "CREATE"],
        repository_size=0, temp_adapter_trained=True,
    )
    assert decision.outcome is Outcome.CREATE
    assert decision.threshold_source == "not_applicable"
    assert "empty" in decision.reason


def test_decision_without_thresholds_is_refused():
    """D-005: no default tau exists, so a decision cannot be made before selection."""
    with pytest.raises(ThresholdNotSelectedError):
        decide(
            task_id="t", task_index=1, scores=[_score("A1", 0.9)], thresholds=None,
            decision_order="train_then_decide", allowed_outcomes=["FUSE", "CREATE"],
            repository_size=1, temp_adapter_trained=True,
        )


def test_multi_criteria_inputs_are_recorded():
    """Criterion E3: the record must show the decision saw many features, not one."""
    decision = _decide([_score("A1", 0.4, n_features=54)])
    assert decision.ranked_candidates[0].n_features == 54
    payload = decision.as_dict()
    assert payload["ranked_candidates"][0]["n_features"] == 54
    assert payload["tau_u"] == 0.2 and payload["tau_reuse"] == 0.6


def test_candidates_are_ranked_by_utility():
    decision = _decide([_score("low", 0.1), _score("high", 0.5), _score("mid", 0.3)], top_k=3)
    assert [c.adapter_id for c in decision.ranked_candidates] == ["high", "mid", "low"]


def test_determinism():
    """Invariant I7.3."""
    scores = [_score("A1", 0.4), _score("A2", 0.3)]
    first, second = _decide(scores), _decide(scores)
    assert first.outcome == second.outcome
    assert first.selected_adapter_ids == second.selected_adapter_ids


def test_ablation_restricted_outcomes_are_honoured():
    """A9 reuse-only, A10 fuse-only, A11 create-always."""
    high = [_score("A1", 0.9)]
    assert _decide(high, allowed=["REUSE"]).outcome is Outcome.REUSE
    assert _decide(high, allowed=["FUSE"]).outcome is Outcome.FUSE
    assert _decide(high, allowed=["CREATE"]).outcome is Outcome.CREATE
    low = [_score("A1", 0.01)]
    assert _decide(low, allowed=["FUSE"]).outcome is Outcome.FUSE   # forced despite low utility


def test_tau_reuse_below_tau_u_is_rejected():
    """Reusing outright is a stronger claim than fusing, so it needs at least as much utility."""
    with pytest.raises(ManagerError):
        Thresholds(tau_u=0.8, tau_reuse=0.2, source="fixed", selected_on="config")


def test_unknown_threshold_source_is_rejected():
    with pytest.raises(ThresholdNotSelectedError):
        Thresholds(tau_u=0.1, tau_reuse=0.2, source="vibes", selected_on="nowhere")


def test_empty_repository_without_create_allowed_is_an_error():
    with pytest.raises(ManagerError):
        decide(
            task_id="t", task_index=0, scores=[], thresholds=None,
            decision_order="train_then_decide", allowed_outcomes=["FUSE"],
            repository_size=0, temp_adapter_trained=True,
        )

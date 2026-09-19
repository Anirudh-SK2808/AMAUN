"""M7.2 -- retrieval modes (criterion E5)."""

import pytest
import torch

from amaun.errors import ManagerError
from amaun.features.base import FeatureVector
from amaun.features.normalizer import FeatureNormalizer
from amaun.manager.decision import CandidateScore
from amaun.manager.retrieval import RetrievalMode, score_candidates, select
from amaun.manager.utility_network import OUTCOME_HEADS, UtilityNetwork, UtilityNetworkConfig


def _vectors(n=4, dim=6):
    return {
        f"A{i}": FeatureVector(names=[f"f{j}" for j in range(dim)],
                               values=[float(i + j) for j in range(dim)])
        for i in range(n)
    }


def test_select_single_returns_the_top_candidate():
    scores = [CandidateScore(f"A{i}", utility=float(i), outcomes={}, feature_signature="s", n_features=6)
              for i in range(4)]
    scores.sort(key=lambda s: s.utility, reverse=True)
    assert [s.adapter_id for s in select(scores, mode=RetrievalMode.SINGLE, k=3)] == ["A3"]


def test_select_top_k_returns_k_candidates():
    scores = [CandidateScore(f"A{i}", utility=float(i), outcomes={}, feature_signature="s", n_features=6)
              for i in range(5)]
    scores.sort(key=lambda s: s.utility, reverse=True)
    picked = select(scores, mode="top_k", k=3)
    assert [s.adapter_id for s in picked] == ["A4", "A3", "A2"]


def test_invalid_k_is_rejected():
    scores = [CandidateScore("A0", 0.1, {}, "s", 6)]
    with pytest.raises(ManagerError):
        select(scores, mode="top_k", k=0)


def test_scoring_ranks_by_utility_not_by_similarity():
    """Ranking must come from the network, so a raw feature cannot act as the rule."""
    torch.manual_seed(0)
    vectors = _vectors()
    normalizer = FeatureNormalizer()
    normalizer.fit(list(vectors.values()), split="utility_train", run_mode="dummy")
    network = UtilityNetwork(
        6, UtilityNetworkConfig(hidden_dims=(8,), dropout=0.0, activation="relu",
                                output_heads=OUTCOME_HEADS, monotone_weights=True)
    )
    scores = score_candidates(
        network=network, normalizer=normalizer, vectors=vectors, run_mode="dummy"
    )
    assert len(scores) == len(vectors)
    utilities = [s.utility for s in scores]
    assert utilities == sorted(utilities, reverse=True)
    for score in scores:
        assert set(score.outcomes) == set(OUTCOME_HEADS)
        assert score.n_features == 6


def test_scoring_empty_candidate_set():
    assert score_candidates(
        network=None, normalizer=None, vectors={}, run_mode="dummy"
    ) == []

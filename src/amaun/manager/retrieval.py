"""M7.2 -- candidate enumeration and utility-ranked retrieval (idea.md S5.4 lines 5-11).

Ranking is **always** by predicted utility, never by raw similarity.  The ``single`` and
``top_k`` modes differ only in how many of the ranked candidates are handed to fusion, so a
top-1 AMAUN run and a top-k run share the identical scoring path -- which is what makes the
"is FUSE better than picking the best adapter?" comparison (falsification question 6) a fair
one rather than two different pipelines.
"""

from __future__ import annotations

import enum
from typing import Sequence

import torch

from amaun.errors import ManagerError
from amaun.features.base import FeatureVector
from amaun.features.normalizer import FeatureNormalizer
from amaun.manager.decision import CandidateScore
from amaun.manager.utility_network import UtilityNetwork


class RetrievalMode(str, enum.Enum):
    SINGLE = "single"
    TOP_K = "top_k"


def score_candidates(
    *,
    network: UtilityNetwork,
    normalizer: FeatureNormalizer,
    vectors: dict[str, FeatureVector],
    run_mode: str,
) -> list[CandidateScore]:
    """Normalise each candidate's features and predict its utility and outcome vector."""
    if not vectors:
        return []
    network.eval()
    scores: list[CandidateScore] = []
    adapter_ids = sorted(vectors)
    rows = [normalizer.transform(vectors[adapter_id], run_mode=run_mode) for adapter_id in adapter_ids]
    batch = torch.tensor(rows, dtype=torch.float32)
    with torch.no_grad():
        utilities, outcomes = network(batch)
    for index, adapter_id in enumerate(adapter_ids):
        vector = vectors[adapter_id]
        scores.append(
            CandidateScore(
                adapter_id=adapter_id,
                utility=float(utilities[index]),
                outcomes={
                    head: float(outcomes[index, head_index])
                    for head_index, head in enumerate(network.heads)
                },
                feature_signature=vector.signature(),
                n_features=len(vector),
                missing_features=list(vector.missing),
            )
        )
    return sorted(scores, key=lambda s: s.utility, reverse=True)


def select(scores: Sequence[CandidateScore], *, mode: RetrievalMode | str, k: int) -> list[CandidateScore]:
    """Take the top-1 or top-k utility-ranked candidates."""
    mode = RetrievalMode(mode)
    if not scores:
        return []
    if mode is RetrievalMode.SINGLE:
        return [scores[0]]
    if k < 1:
        raise ManagerError("retrieval.k must be >= 1")
    return list(scores[:k])

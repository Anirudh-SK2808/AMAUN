"""M13 -- baseline management strategies.

Every baseline is a *scorer/decider* plugged into the **same** continual runner, so it shares
the identical training loop, fusion implementation, metric code, and evaluation protocol as
AMAUN.  That is what makes the comparison fair by construction rather than by review: a
baseline cannot accidentally receive a different ROUGE implementation or a different decoding
config, because there is only one of each.

What differs per strategy is exactly the management policy, which is the variable under study.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from amaun.features.base import FeatureVector
from amaun.manager.decision import CandidateScore, Decision, Outcome, Thresholds, decide


class ManagementStrategy(abc.ABC):
    """Scores candidates and chooses an outcome.  The runner executes the outcome."""

    strategy_id: str = "abstract"
    description: str = ""
    #: Whether this strategy needs a trained utility network.
    requires_utility_network: bool = False

    @abc.abstractmethod
    def score(self, vectors: Mapping[str, FeatureVector], *, run_mode: str) -> list[CandidateScore]:
        ...

    @abc.abstractmethod
    def decide(
        self,
        *,
        task_id: str,
        task_index: int,
        scores: Sequence[CandidateScore],
        repository_size: int,
        temp_adapter_trained: bool,
        config_hash: str,
    ) -> Decision:
        ...

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "description": self.description,
            "requires_utility_network": self.requires_utility_network,
        }


def _forced_decision(
    *,
    outcome: Outcome,
    task_id: str,
    task_index: int,
    scores: Sequence[CandidateScore],
    repository_size: int,
    temp_adapter_trained: bool,
    reason: str,
    strategy_id: str,
    top_k: int = 1,
    config_hash: str = "",
) -> Decision:
    """Build a Decision for a strategy whose policy is fixed rather than learned."""
    ranked = sorted(scores, key=lambda s: s.utility, reverse=True)
    if repository_size == 0 or not ranked:
        outcome, selected = Outcome.CREATE, []
        reason = "repository empty: create"
    elif outcome is Outcome.CREATE:
        selected = []
    elif outcome is Outcome.FUSE:
        selected = [candidate.adapter_id for candidate in ranked[: max(1, top_k)]]
    else:
        selected = [ranked[0].adapter_id]
    return Decision(
        decision_id=f"{strategy_id}-{task_index:04d}",
        outcome=outcome,
        task_id=task_id,
        task_index=task_index,
        selected_adapter_ids=selected,
        ranked_candidates=list(ranked),
        tau_u=None,
        tau_reuse=None,
        threshold_source="not_applicable",
        decision_order="train_then_decide",
        reason=reason,
        temp_adapter_trained=temp_adapter_trained,
        config_hash=config_hash,
        repository_size_before=repository_size,
        n_candidates_scored=len(ranked),
    )


# --------------------------------------------------------------------------------------
# B2 -- separate LoRA per task
# --------------------------------------------------------------------------------------
class SeparateLoRAStrategy(ManagementStrategy):
    """One adapter per task, forever.  No forgetting, unbounded growth.

    The growth reference for H3: if AMAUN does not store fewer adapters than this, its
    growth-control claim fails.
    """

    strategy_id = "B2_separate_lora"
    description = "Train and keep a new LoRA adapter for every task (brief S14 item 2)"

    def score(self, vectors, *, run_mode):  # no scoring needed
        return []

    def decide(self, *, task_id, task_index, scores, repository_size, temp_adapter_trained, config_hash):
        return _forced_decision(
            outcome=Outcome.CREATE, task_id=task_id, task_index=task_index, scores=scores,
            repository_size=max(repository_size, 1) if repository_size else 0,
            temp_adapter_trained=temp_adapter_trained,
            reason="policy: always create a new adapter", strategy_id=self.strategy_id,
            config_hash=config_hash,
        )


# --------------------------------------------------------------------------------------
# B3 -- naive uniform delta averaging
# --------------------------------------------------------------------------------------
class NaiveAverageStrategy(ManagementStrategy):
    """Always fuse with uniform coefficients and no learning.

    Coefficients are fixed at ``1/m`` and never optimised, so this isolates "does *learning*
    the coefficients matter?" from "does combining adapters at all matter?".
    Averaging is still at the effective-delta level -- averaging A/B factors is forbidden
    everywhere (idea.md S16.3), including for baselines.
    """

    strategy_id = "B3_naive_average"
    description = "Uniform-coefficient delta averaging with the most recent adapter (brief S14 item 3)"

    def score(self, vectors, *, run_mode):
        # Arbitrary but deterministic order; the policy does not rank.
        return [
            CandidateScore(adapter_id=adapter_id, utility=0.0, outcomes={},
                           feature_signature=vector.signature(), n_features=len(vector))
            for adapter_id, vector in sorted(vectors.items())
        ]

    def decide(self, *, task_id, task_index, scores, repository_size, temp_adapter_trained, config_hash):
        return _forced_decision(
            outcome=Outcome.FUSE, task_id=task_id, task_index=task_index, scores=scores,
            repository_size=repository_size, temp_adapter_trained=temp_adapter_trained,
            reason="policy: always fuse with uniform, non-learnable coefficients",
            strategy_id=self.strategy_id, config_hash=config_hash,
        )


# --------------------------------------------------------------------------------------
# Always merge -- decision-quality baseline for the L4 decision study
# --------------------------------------------------------------------------------------
class AlwaysMergeStrategy(ManagementStrategy):
    """Always FUSE, with the candidate nearest by one configured feature.

    Differs from B3 only in *which* candidate and *how*: B3 fixes uniform, non-learnable
    coefficients and picks by id order; this policy uses the same learnable fusion + KD as
    AMAUN, so a comparison against it isolates the merge-or-not decision itself.
    """

    strategy_id = "always_merge"
    description = "Always fuse with the nearest candidate by a single feature (decision baseline)"

    def __init__(self, feature: str, *, higher_is_more_similar: bool) -> None:
        self.feature = feature
        self.higher_is_more_similar = higher_is_more_similar

    def score(self, vectors, *, run_mode):
        scores: list[CandidateScore] = []
        for adapter_id, vector in sorted(vectors.items()):
            values = vector.as_dict()
            if self.feature not in values or self.feature in vector.missing:
                raise KeyError(
                    f"always_merge needs a measured '{self.feature}' for candidate '{adapter_id}'; "
                    f"it is {'missing' if self.feature in vector.missing else 'absent'}"
                )
            raw = values[self.feature]
            oriented = raw if self.higher_is_more_similar else -raw
            scores.append(CandidateScore(
                adapter_id=adapter_id, utility=float(oriented), outcomes={self.feature: float(raw)},
                feature_signature=vector.signature(), n_features=1,
            ))
        return sorted(scores, key=lambda s: s.utility, reverse=True)

    def decide(self, *, task_id, task_index, scores, repository_size, temp_adapter_trained, config_hash):
        return _forced_decision(
            outcome=Outcome.FUSE, task_id=task_id, task_index=task_index, scores=scores,
            repository_size=repository_size, temp_adapter_trained=temp_adapter_trained,
            reason=f"policy: always fuse with the nearest candidate by '{self.feature}'",
            strategy_id=self.strategy_id, config_hash=config_hash,
        )


# --------------------------------------------------------------------------------------
# B4 -- similarity-based selection (the direct H1 comparison)
# --------------------------------------------------------------------------------------
@dataclass
class SimilarityConfig:
    feature: str          # which single feature acts as the similarity signal
    threshold: float
    higher_is_more_similar: bool


class SimilaritySelectionStrategy(ManagementStrategy):
    """Select and fuse by a **single** similarity metric, thresholded.

    This is the method AMAUN claims to beat (H1), so it must be implemented honestly: it uses
    exactly one feature, which is what "similarity-based selection" means, and it receives the
    *same* feature extraction pipeline so it is not handicapped by worse inputs.

    The threshold is a real weakness of the approach rather than a strawman: idea.md S16.2
    objects to single-metric rules precisely because one favourable metric can license an
    unrelated merge, and this baseline exhibits that behaviour by design.
    """

    strategy_id = "B4_similarity_selection"
    description = "Single-metric similarity selection with a fixed threshold (brief S14 item 4)"

    def __init__(self, config: SimilarityConfig) -> None:
        self.config = config

    def score(self, vectors, *, run_mode):
        scores: list[CandidateScore] = []
        for adapter_id, vector in sorted(vectors.items()):
            values = vector.as_dict()
            if self.config.feature not in values:
                raise KeyError(
                    f"similarity baseline needs feature '{self.config.feature}', which is not in "
                    f"the extracted vector. Available: {sorted(values)[:10]}..."
                )
            raw = values[self.config.feature]
            # Orient so larger always means "more similar", then map to a bounded score so
            # it is comparable against a threshold.
            oriented = raw if self.config.higher_is_more_similar else -raw
            scores.append(
                CandidateScore(
                    adapter_id=adapter_id,
                    utility=float(math.tanh(oriented)),
                    outcomes={self.config.feature: float(raw)},
                    feature_signature=vector.signature(),
                    n_features=1,   # honestly reported: this rule uses ONE feature
                )
            )
        return sorted(scores, key=lambda s: s.utility, reverse=True)

    def decide(self, *, task_id, task_index, scores, repository_size, temp_adapter_trained, config_hash):
        thresholds = Thresholds(
            tau_u=self.config.threshold,
            tau_reuse=max(self.config.threshold, 1.0),  # this baseline has no REUSE outcome
            source="fixed",
            selected_on="config",
            notes="single-metric similarity threshold",
        )
        return decide(
            task_id=task_id, task_index=task_index, scores=scores, thresholds=thresholds,
            decision_order="train_then_decide", allowed_outcomes=["FUSE", "CREATE"],
            repository_size=repository_size, temp_adapter_trained=temp_adapter_trained,
            top_k=1, reuse_prescreen_passed=False, config_hash=config_hash,
        )


# --------------------------------------------------------------------------------------
# B5 -- fixed-coefficient fusion (ablation A3 as a standalone baseline)
# --------------------------------------------------------------------------------------
class FixedFusionStrategy(SimilaritySelectionStrategy):
    """Select as AMAUN would, but fuse with non-learnable coefficients.

    Implemented as a selection strategy plus ``fusion.coefficient.learning_rate = 0`` in the
    arm's config, so the *only* difference from AMAUN is coefficient learnability. That is the
    clean test of H2.
    """

    strategy_id = "B5_fixed_fusion"
    description = "Fusion with fixed (non-learnable) coefficients (idea.md S12.4 'Fixed fusion')"


# --------------------------------------------------------------------------------------
# AMAUN itself
# --------------------------------------------------------------------------------------
class AmaunStrategy(ManagementStrategy):
    """The method: learned multi-criteria utility over all enabled families."""

    strategy_id = "amaun"
    description = "Learned utility network over 7 feature families; REUSE / FUSE / CREATE"
    requires_utility_network = True

    def __init__(
        self,
        *,
        network: Any,
        normalizer: Any,
        thresholds: Thresholds,
        decision_order: str,
        allowed_outcomes: Sequence[str],
        top_k: int,
        reuse_gate: Any | None = None,
    ) -> None:
        self.network = network
        self.normalizer = normalizer
        self.thresholds = thresholds
        self.decision_order = decision_order
        self.allowed_outcomes = list(allowed_outcomes)
        self.top_k = top_k
        #: Stage-1 REUSE gate (amaun.manager.reuse_gate.ReuseGate), consulted before training when
        #: decision_order == prescreen_then_train.  None keeps the post-training pre-screen only.
        self.reuse_gate = reuse_gate
        self._prescreen_passed: bool = False

    @property
    def gates_before_training(self) -> bool:
        """True when REUSE is decided before A_temp is trained (D-003 O1)."""
        return (
            self.decision_order == "prescreen_then_train"
            and self.reuse_gate is not None
            and "REUSE" in self.allowed_outcomes
        )

    def set_prescreen(self, passed: bool) -> None:
        """Record whether the support-set reuse pre-screen passed (D-003 option O1)."""
        self._prescreen_passed = passed

    def score(self, vectors, *, run_mode):
        from amaun.manager.retrieval import score_candidates

        return score_candidates(
            network=self.network, normalizer=self.normalizer, vectors=vectors, run_mode=run_mode
        )

    def decide(self, *, task_id, task_index, scores, repository_size, temp_adapter_trained, config_hash):
        return decide(
            task_id=task_id, task_index=task_index, scores=scores, thresholds=self.thresholds,
            decision_order=self.decision_order, allowed_outcomes=self.allowed_outcomes,
            repository_size=repository_size, temp_adapter_trained=temp_adapter_trained,
            top_k=self.top_k, reuse_prescreen_passed=self._prescreen_passed,
            config_hash=config_hash,
        )


# --------------------------------------------------------------------------------------
# A2 -- AMAUN with a fixed selection rule instead of the learned network
# --------------------------------------------------------------------------------------
class FixedRuleStrategy(SimilaritySelectionStrategy):
    """Ablation A2: all seven families are extracted, but selection uses a fixed rule.

    Isolates "does *learning* the utility help?" from "do the features help?", which a
    single-feature baseline cannot separate.
    """

    strategy_id = "A2_fixed_rule"
    description = "Fixed selection rule over extracted features (ablation A2)"

"""M7.3 -- the REUSE / FUSE / CREATE decision (idea.md S5.2, generalised per D-003).

    i*     = argmax_i U_i
    REUSE  if U_i* >= tau_reuse  and the reuse pre-screen meets the support-set bar
    FUSE   if U_i* >= tau_u
    CREATE if max_i U_i < tau_u

``idea.md`` specifies two outcomes (MERGE / CREATE-NEW); the brief requires three.  D-003
records this as an OPEN methodology addition, so **both decision orders are implemented and
neither is the default**:

- ``train_then_decide`` -- faithful to idea.md S5.4: always train ``A_temp`` first, then choose
  among the three outcomes.  REUSE here saves storage but not training cost.
- ``prescreen_then_train`` -- REUSE is decided *before* training ``A_temp``, so it saves
  training cost too.  This is the stronger efficiency claim and the larger deviation.

Reporting both is what separates "the mechanism helped" from "this arm simply trained less"
(falsification question 14), which is why the ``Decision`` record carries the budget-relevant
fact of whether a temporary adapter was trained at all.
"""

from __future__ import annotations

import enum
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from amaun.errors import ManagerError, ThresholdNotSelectedError


class Outcome(str, enum.Enum):
    REUSE = "REUSE"
    FUSE = "FUSE"
    CREATE = "CREATE"


class DecisionOrder(str, enum.Enum):
    PRESCREEN_THEN_TRAIN = "prescreen_then_train"
    TRAIN_THEN_DECIDE = "train_then_decide"


@dataclass
class CandidateScore:
    """One candidate's predicted utility and outcome vector."""

    adapter_id: str
    utility: float
    outcomes: dict[str, float]
    feature_signature: str
    n_features: int
    missing_features: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "utility": self.utility,
            "outcomes": self.outcomes,
            "feature_signature": self.feature_signature,
            "n_features": self.n_features,
            "n_missing_features": len(self.missing_features),
        }


@dataclass
class Decision:
    """The decision record.  Also the capability token authorising registration (I3.2)."""

    decision_id: str
    outcome: Outcome
    task_id: str
    task_index: int
    selected_adapter_ids: list[str]
    ranked_candidates: list[CandidateScore]
    tau_u: float | None
    tau_reuse: float | None
    threshold_source: str
    decision_order: str
    reason: str
    temp_adapter_trained: bool
    config_hash: str = ""
    repository_size_before: int = 0
    n_candidates_scored: int = 0

    @property
    def best_utility(self) -> float | None:
        return self.ranked_candidates[0].utility if self.ranked_candidates else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "outcome": self.outcome.value,
            "task_id": self.task_id,
            "task_index": self.task_index,
            "selected_adapter_ids": list(self.selected_adapter_ids),
            "ranked_candidates": [c.as_dict() for c in self.ranked_candidates],
            "best_utility": self.best_utility,
            "tau_u": self.tau_u,
            "tau_reuse": self.tau_reuse,
            "threshold_source": self.threshold_source,
            "decision_order": self.decision_order,
            "reason": self.reason,
            "temp_adapter_trained": self.temp_adapter_trained,
            "repository_size_before": self.repository_size_before,
            "n_candidates_scored": self.n_candidates_scored,
            "config_hash": self.config_hash,
        }


@dataclass
class Thresholds:
    """Selected thresholds plus the provenance of *how* they were selected (D-005)."""

    tau_u: float
    tau_reuse: float
    source: str
    selected_on: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.source not in {"dev_stream", "historical_cv", "fixed"}:
            raise ThresholdNotSelectedError(
                f"unknown threshold source '{self.source}'; must be one of dev_stream, "
                f"historical_cv, fixed (docs/decision_register.md D-005)"
            )
        if self.tau_reuse < self.tau_u:
            raise ManagerError(
                f"tau_reuse ({self.tau_reuse}) must be >= tau_u ({self.tau_u}): reusing an "
                f"existing adapter outright is a stronger claim than merely fusing with it, so "
                f"it must require at least as much predicted utility"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tau_u": self.tau_u,
            "tau_reuse": self.tau_reuse,
            "source": self.source,
            "selected_on": self.selected_on,
            "notes": self.notes,
        }


def decide(
    *,
    task_id: str,
    task_index: int,
    scores: Sequence[CandidateScore],
    thresholds: Thresholds | None,
    decision_order: DecisionOrder | str,
    allowed_outcomes: Sequence[str],
    repository_size: int,
    temp_adapter_trained: bool,
    top_k: int = 1,
    reuse_prescreen_passed: bool | None = None,
    config_hash: str = "",
) -> Decision:
    """Apply the decision rule to already-scored candidates.

    Scoring is separated from deciding so the rule can be unit-tested on synthetic utilities
    without a model, and so a baseline can substitute its own scorer while reusing the exact
    same rule.
    """
    order = DecisionOrder(decision_order)
    allowed = {Outcome(outcome) for outcome in allowed_outcomes}
    if not allowed:
        raise ManagerError("at least one outcome must be allowed")

    decision_id = f"d-{task_index:04d}-{uuid.uuid4().hex[:8]}"
    ranked = sorted(scores, key=lambda s: s.utility, reverse=True)

    # idea.md S5.4 lines 2-4: an empty repository short-circuits to CREATE without
    # consulting the network at all (invariant I7.5).
    if repository_size == 0 or not ranked:
        if Outcome.CREATE not in allowed:
            raise ManagerError(
                f"repository is empty for task '{task_id}' but CREATE is not an allowed "
                f"outcome; this configuration cannot make progress"
            )
        return Decision(
            decision_id=decision_id,
            outcome=Outcome.CREATE,
            task_id=task_id,
            task_index=task_index,
            selected_adapter_ids=[],
            ranked_candidates=[],
            tau_u=thresholds.tau_u if thresholds else None,
            tau_reuse=thresholds.tau_reuse if thresholds else None,
            threshold_source=thresholds.source if thresholds else "not_applicable",
            decision_order=order.value,
            reason="repository empty: insert A_temp directly (idea.md S5.4 lines 2-4)",
            temp_adapter_trained=temp_adapter_trained,
            config_hash=config_hash,
            repository_size_before=repository_size,
            n_candidates_scored=0,
        )

    if thresholds is None:
        raise ThresholdNotSelectedError(
            "thresholds must be selected from a declared source before any decision is made "
            "(docs/decision_register.md D-005); AMAUN provides no default tau_u"
        )

    best = ranked[0]
    if best.utility >= thresholds.tau_reuse and Outcome.REUSE in allowed and reuse_prescreen_passed:
        outcome = Outcome.REUSE
        selected = [best.adapter_id]
        reason = (
            f"U={best.utility:.4f} >= tau_reuse={thresholds.tau_reuse:.4f} and the support-set "
            f"pre-screen passed: reuse '{best.adapter_id}' without training a new adapter"
        )
    elif best.utility >= thresholds.tau_u and Outcome.FUSE in allowed:
        outcome = Outcome.FUSE
        selected = [candidate.adapter_id for candidate in ranked[: max(1, top_k)]]
        reason = (
            f"U={best.utility:.4f} >= tau_u={thresholds.tau_u:.4f}: fuse with top-"
            f"{len(selected)} candidate(s) {selected}"
        )
    elif Outcome.CREATE in allowed:
        outcome = Outcome.CREATE
        selected = []
        reason = (
            f"max U={best.utility:.4f} < tau_u={thresholds.tau_u:.4f}: no existing adapter is "
            f"predicted useful enough; create a new adapter"
        )
    else:
        # Every preferred outcome is disabled by the ablation config; fall back to whatever
        # remains rather than silently doing nothing.
        outcome = sorted(allowed, key=lambda o: o.value)[0]
        selected = [best.adapter_id] if outcome is not Outcome.CREATE else []
        reason = f"ablation restricts outcomes to {sorted(o.value for o in allowed)}; forced {outcome.value}"

    return Decision(
        decision_id=decision_id,
        outcome=outcome,
        task_id=task_id,
        task_index=task_index,
        selected_adapter_ids=selected,
        ranked_candidates=list(ranked),
        tau_u=thresholds.tau_u,
        tau_reuse=thresholds.tau_reuse,
        threshold_source=thresholds.source,
        decision_order=order.value,
        reason=reason,
        temp_adapter_trained=temp_adapter_trained,
        config_hash=config_hash,
        repository_size_before=repository_size,
        n_candidates_scored=len(ranked),
    )

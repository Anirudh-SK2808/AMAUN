"""M9 -- utility-based repository compression (idea.md S7.2).

    compress while Mem(R_c) >= R_max:
        (i*, j*) = argmax_{i != j} U_merge(A_i, A_j)
        R_c <- (R_c \\ {A_i*, A_j*}) union {A_merge_ij}

idea.md S16.7 forbids replacing this with an age/usage heuristic: the framework's claim is
that *one* utility principle governs creation, merging, and compression.  A heuristic variant
exists only as baseline ``B6``, and it is constructed through a separate, explicitly named
class so it cannot be mistaken for the method.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.errors import ManagerError


class CompressionMechanism(str, enum.Enum):
    UTILITY = "utility"
    HEURISTIC = "heuristic"   # baseline B6 only


@dataclass
class CompressionPlan:
    """A single compression step: which pair to merge, and why."""

    first_id: str
    second_id: str
    utility: float
    mechanism: str
    rationale: str


@dataclass
class CompressionReport:
    triggered: bool
    reason: str
    repository_bytes_before: int
    repository_bytes_after: int
    steps: list[CompressionPlan] = field(default_factory=list)
    pairs_evaluated: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "repository_bytes_before": self.repository_bytes_before,
            "repository_bytes_after": self.repository_bytes_after,
            "n_steps": len(self.steps),
            "pairs_evaluated": self.pairs_evaluated,
            "steps": [
                {
                    "pair": [s.first_id, s.second_id],
                    "utility": s.utility,
                    "mechanism": s.mechanism,
                    "rationale": s.rationale,
                }
                for s in self.steps
            ],
        }


PairUtilityFn = Callable[[AdapterEntry, AdapterEntry], float]


def should_compress(repository: AdapterRepository, max_bytes: int | None) -> tuple[bool, str]:
    """``Mem(R_c) >= R_max`` (idea.md S7.2)."""
    if max_bytes is None:
        return False, "repository.max_bytes is null: compression disabled (ablation A5)"
    current = repository.memory_bytes()
    if current >= max_bytes:
        return True, f"Mem(R_c)={current} >= R_max={max_bytes}"
    return False, f"Mem(R_c)={current} < R_max={max_bytes}"


def plan_compression(
    repository: AdapterRepository,
    *,
    pair_utility: PairUtilityFn,
    mechanism: CompressionMechanism | str,
    min_adapters: int,
) -> tuple[CompressionPlan | None, int]:
    """Choose the highest-utility mergeable pair.  Returns ``(plan, pairs_evaluated)``."""
    mechanism = CompressionMechanism(mechanism)
    entries = repository.entries()
    if len(entries) <= max(2, min_adapters):
        return None, 0

    best: CompressionPlan | None = None
    evaluated = 0
    for index, left in enumerate(entries):
        for right in entries[index + 1 :]:
            evaluated += 1
            try:
                utility = float(pair_utility(left, right))
            except Exception as exc:  # a pair may be structurally incompatible
                continue
            if best is None or utility > best.utility:
                best = CompressionPlan(
                    first_id=left.adapter_id,
                    second_id=right.adapter_id,
                    utility=utility,
                    mechanism=mechanism.value,
                    rationale=(
                        f"argmax_(i!=j) U_merge over {evaluated} pairs"
                        if mechanism is CompressionMechanism.UTILITY
                        else "heuristic score (baseline B6 only)"
                    ),
                )
    return best, evaluated


def heuristic_pair_utility(left: AdapterEntry, right: AdapterEntry, *, task_index: int) -> float:
    """Age/usage heuristic -- **baseline B6 only** (idea.md S16.7 forbids this for AMAUN).

    Prefers merging the pair that is oldest and least used, which is the standard
    ad-hoc policy AMAUN's utility-based compression is claimed to beat (H5).
    """
    score = 0.0
    for entry in (left, right):
        score += entry.metadata.age(task_index)
        score -= 2.0 * entry.metadata.use_count
    return score

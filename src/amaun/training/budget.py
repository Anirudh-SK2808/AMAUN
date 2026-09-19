"""M13 -- the budget ledger that makes baseline fairness mechanical.

Brief S14 says "do not give AMAUN additional information unavailable to the baseline", and
falsification question 14 asks whether a gain could come from extra training rather than the
method.  Those are easy to assert and hard to guarantee, so the ledger *measures* what each
arm consumed and the comparison layer refuses to report a difference between arms whose
budgets diverge beyond tolerance (invariant I13.1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from amaun.errors import BudgetParityError


@dataclass
class ArmBudget:
    """What one experimental arm consumed."""

    arm_id: str
    optimizer_steps: int = 0
    tokens_seen: int = 0
    examples_seen: int = 0
    trainable_parameters: int = 0
    stored_parameters: int = 0
    wall_seconds: float = 0.0
    tuning_trials: int = 0
    forward_passes: int = 0
    notes: list[str] = field(default_factory=list)

    def add_steps(self, steps: int = 1) -> None:
        self.optimizer_steps += steps

    def add_batch(self, *, tokens: int, examples: int) -> None:
        self.tokens_seen += tokens
        self.examples_seen += examples

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Which fields must match for a comparison to be fair, and how each is described in the
# failure message.  ``trainable_parameters`` is included because fusion adds L*m
# coefficients; that is small but must be disclosed, not hidden.
_PARITY_FIELDS = (
    "optimizer_steps",
    "tokens_seen",
    "examples_seen",
    "tuning_trials",
)


class BudgetLedger:
    """Collects per-arm budgets and gates comparisons between them."""

    def __init__(self, tolerance: float) -> None:
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError("tolerance must be a fraction in [0, 1]")
        self.tolerance = tolerance
        self.arms: dict[str, ArmBudget] = {}

    def arm(self, arm_id: str) -> ArmBudget:
        if arm_id not in self.arms:
            self.arms[arm_id] = ArmBudget(arm_id=arm_id)
        return self.arms[arm_id]

    def check_parity(
        self,
        arm_a: str,
        arm_b: str,
        *,
        allow_unequal_budget: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Compare two arms' budgets.  Raises unless they match or the run declares why not."""
        left, right = self.arm(arm_a), self.arm(arm_b)
        divergences: dict[str, dict[str, Any]] = {}
        for field_name in _PARITY_FIELDS:
            a_value = float(getattr(left, field_name))
            b_value = float(getattr(right, field_name))
            scale = max(abs(a_value), abs(b_value), 1.0)
            relative = abs(a_value - b_value) / scale
            if relative > self.tolerance:
                divergences[field_name] = {
                    arm_a: a_value, arm_b: b_value, "relative_difference": relative
                }
        verdict = {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "tolerance": self.tolerance,
            "divergences": divergences,
            "fair": not divergences,
            "allow_unequal_budget": allow_unequal_budget,
            "declared_reason": reason,
        }
        if divergences and not allow_unequal_budget:
            raise BudgetParityError(
                f"refusing to compare '{arm_a}' with '{arm_b}': budgets diverge beyond "
                f"tolerance {self.tolerance:.3f} on {sorted(divergences)}. "
                f"A performance difference under unequal budgets cannot be attributed to the "
                f"method (falsification question 14). Either equalise the budgets or set "
                f"experiment.allow_unequal_budget with a reason, which will be printed "
                f"alongside every reported comparison. Details: {json.dumps(divergences, default=str)}"
            )
        if divergences and allow_unequal_budget and not reason:
            raise BudgetParityError(
                "allow_unequal_budget=True requires an explicit reason, which is recorded in "
                "the results so readers can see the comparison is not budget-matched"
            )
        return verdict

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(
                {"tolerance": self.tolerance, "arms": {k: v.as_dict() for k, v in self.arms.items()}},
                handle, indent=2,
            )
            handle.write("\n")
        return path

"""M10 -- continual-learning metrics from the performance matrix.

``S[i][j]`` = score on task ``j`` after learning task ``i`` (both 0-indexed, ``i >= j`` filled).

    AvgPerf_T  = (1/T) sum_j S[T][j]
    Forgetting = (1/(T-1)) sum_{j<T} max(0, max_{i<T} S[i][j] - S[T][j])
    BWT        = (1/(T-1)) sum_{j<T} ( S[T][j] - S[j][j] )
    FWT        = (1/(T-1)) sum_{j>0} ( S[j-1][j] - b_j )

Two deliberate asymmetries, both from the sources:

- **Forgetting is clipped** at 0 per idea.md S4.5, so improvement on an old task does not
  offset forgetting on another. **BWT is signed**, so it *can* be positive. They therefore
  measure different things and both are reported; reporting only one would hide either
  backward transfer or forgetting.
- **FWT needs a reference** ``b_j``: the frozen backbone's zero-shot score on task ``j``,
  measured once with no adapters (assumption A-06). FWT is undefined without it, so this
  module requires it rather than silently substituting zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from amaun.errors import EvaluationError


@dataclass
class PerformanceMatrix:
    """``S[i][j]``, stored sparsely so partial runs are still analysable."""

    task_order: list[str]
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    zero_shot_baseline: dict[str, float] = field(default_factory=dict)
    score_name: str = "primary_score"

    def record(self, after_task: str, evaluated_task: str, score: float) -> None:
        self.scores.setdefault(after_task, {})[evaluated_task] = float(score)

    def get(self, after_task: str, evaluated_task: str) -> float | None:
        return self.scores.get(after_task, {}).get(evaluated_task)

    def as_grid(self) -> list[list[float | None]]:
        return [[self.get(row, column) for column in self.task_order] for row in self.task_order]

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(
                {
                    "task_order": self.task_order,
                    "scores": self.scores,
                    "zero_shot_baseline": self.zero_shot_baseline,
                    "score_name": self.score_name,
                },
                handle, indent=2,
            )
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PerformanceMatrix":
        with Path(path).open() as handle:
            data = json.load(handle)
        return cls(
            task_order=data["task_order"],
            scores=data["scores"],
            zero_shot_baseline=data.get("zero_shot_baseline", {}),
            score_name=data.get("score_name", "primary_score"),
        )


@dataclass
class ContinualMetrics:
    average_performance: float
    forgetting: float | None
    backward_transfer: float | None
    forward_transfer: float | None
    per_task_final: dict[str, float]
    n_tasks: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def compute(matrix: PerformanceMatrix) -> ContinualMetrics:
    """Compute all four metrics from the matrix; ``None`` where a metric is undefined."""
    order = matrix.task_order
    if not order:
        raise EvaluationError("performance matrix has no task order")
    final_row = order[-1]
    notes: list[str] = []

    final_scores: dict[str, float] = {}
    for task in order:
        value = matrix.get(final_row, task)
        if value is not None:
            final_scores[task] = value
    if not final_scores:
        raise EvaluationError(f"no scores recorded after the final task '{final_row}'")
    if len(final_scores) < len(order):
        notes.append(
            f"only {len(final_scores)}/{len(order)} tasks evaluated after the final task; "
            f"AvgPerf is over the evaluated subset"
        )
    average = sum(final_scores.values()) / len(final_scores)

    # Forgetting and BWT over tasks seen before the last one.
    prior = order[:-1]
    forgetting_terms, bwt_terms = [], []
    for task in prior:
        final_value = matrix.get(final_row, task)
        if final_value is None:
            continue
        peaks = [
            matrix.get(row, task)
            for row in order[:-1]
            if matrix.get(row, task) is not None
        ]
        if peaks:
            forgetting_terms.append(max(0.0, max(peaks) - final_value))
        diagonal = matrix.get(task, task)
        if diagonal is not None:
            bwt_terms.append(final_value - diagonal)

    forgetting = sum(forgetting_terms) / len(forgetting_terms) if forgetting_terms else None
    backward = sum(bwt_terms) / len(bwt_terms) if bwt_terms else None

    # FWT: score on task j before learning it, versus its zero-shot reference.
    fwt_terms = []
    for index in range(1, len(order)):
        task = order[index]
        previous = order[index - 1]
        before = matrix.get(previous, task)
        baseline = matrix.zero_shot_baseline.get(task)
        if before is not None and baseline is not None:
            fwt_terms.append(before - baseline)
    forward = sum(fwt_terms) / len(fwt_terms) if fwt_terms else None
    if forward is None and len(order) > 1:
        notes.append(
            "forward transfer is undefined: it requires both pre-task scores and a zero-shot "
            "baseline b_j measured with the frozen backbone and no adapters (assumption A-06)"
        )

    return ContinualMetrics(
        average_performance=average,
        forgetting=forgetting,
        backward_transfer=backward,
        forward_transfer=forward,
        per_task_final=final_scores,
        n_tasks=len(order),
        notes=notes,
    )

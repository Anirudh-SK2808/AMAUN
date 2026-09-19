"""M12 -- efficiency metrics (brief S15).

Reported jointly, never in isolation.  In particular, D-014: after a FUSE the adapter count
stays flat while one adapter now serves two tasks, so quoting adapter count alone would make
fusion look free.  :func:`snapshot` therefore always emits count, tasks-served-per-adapter,
stored parameters, and storage bytes together, and :func:`format_growth_row` refuses to build
a reporting row that omits any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amaun.adapters.repository import AdapterRepository


@dataclass
class EfficiencySnapshot:
    task_index: int
    n_adapters: int
    n_archived: int
    total_stored_parameters: int
    storage_bytes: int
    tasks_served_total: int
    tasks_served_per_adapter: float
    adapters_per_task: float | None
    trainable_parameters_this_event: int = 0
    communication_bytes_cumulative: int = 0
    training_seconds_cumulative: float = 0.0
    inference_overhead_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def snapshot(
    repository: AdapterRepository,
    *,
    task_index: int,
    n_tasks_seen: int,
    trainable_parameters_this_event: int = 0,
    communication_bytes_cumulative: int = 0,
    training_seconds_cumulative: float = 0.0,
    inference_overhead_seconds: float | None = None,
) -> EfficiencySnapshot:
    served = repository.tasks_served()
    total_served = sum(len(tasks) for tasks in served.values())
    count = len(repository)
    return EfficiencySnapshot(
        task_index=task_index,
        n_adapters=count,
        n_archived=len(repository.archived_ids()),
        total_stored_parameters=repository.total_parameters(),
        storage_bytes=repository.memory_bytes(),
        tasks_served_total=total_served,
        tasks_served_per_adapter=(total_served / count) if count else 0.0,
        adapters_per_task=(count / n_tasks_seen) if n_tasks_seen else None,
        trainable_parameters_this_event=trainable_parameters_this_event,
        communication_bytes_cumulative=communication_bytes_cumulative,
        training_seconds_cumulative=training_seconds_cumulative,
        inference_overhead_seconds=inference_overhead_seconds,
    )


# The four columns that must appear together in any growth-control table (D-014).
REQUIRED_GROWTH_COLUMNS = (
    "n_adapters",
    "tasks_served_per_adapter",
    "total_stored_parameters",
    "average_performance",
)


def format_growth_row(snapshot: EfficiencySnapshot, average_performance: float | None) -> dict[str, Any]:
    """Build a reporting row, refusing to omit the context that makes the count meaningful."""
    row = {
        "task_index": snapshot.task_index,
        "n_adapters": snapshot.n_adapters,
        "tasks_served_per_adapter": snapshot.tasks_served_per_adapter,
        "total_stored_parameters": snapshot.total_stored_parameters,
        "storage_bytes": snapshot.storage_bytes,
        "average_performance": average_performance,
    }
    missing = [column for column in REQUIRED_GROWTH_COLUMNS if row.get(column) is None]
    if missing:
        raise ValueError(
            f"growth-control rows must report {list(REQUIRED_GROWTH_COLUMNS)} together; missing "
            f"{missing}. An adapter-count reduction bought by a performance loss must stay "
            f"visible (docs/decision_register.md D-014)."
        )
    return row

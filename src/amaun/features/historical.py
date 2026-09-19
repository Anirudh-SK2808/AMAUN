"""M6.6 -- historical features (idea.md S4.6).

    Age_i      = t_current - t_created,i
    Freq_i     = N_use,i / N_events
    Merge_i    = N_merge,i
    PerfHist_i = (1/K) sum_k s_{i,k}

Read straight from :class:`~amaun.adapters.metadata.AdapterMetadata`, which is why that
metadata is maintained transactionally by the repository rather than reconstructed later.
``PerfHist`` returns ``None`` for an adapter with no history: encoding "unknown" as zero
would systematically bias the utility network against newly created adapters.
"""

from __future__ import annotations

from typing import Any

from amaun.features.base import FeatureFamily


class HistoricalFeatures(FeatureFamily):
    name = "historical"

    def compute(self, ctx: Any) -> dict[str, float | None]:
        metadata = ctx.candidate_metadata
        task_index = int(getattr(ctx, "task_index", 0))
        total_events = max(1, int(getattr(ctx, "total_events", 1)))

        last_used = metadata.last_used_at_task_index
        return {
            "Age_i": float(metadata.age(task_index)),
            "Freq_i": float(metadata.use_count) / total_events,
            "Merge_i": float(metadata.merge_count),
            "PerfHist_i": self.finite(metadata.mean_historical_performance()),
            "UseCount_i": float(metadata.use_count),
            "TasksServed_i": float(len(metadata.tasks_served)),
            "StepsSinceUse_i": float(task_index - last_used) if last_used is not None else None,
            "IsFused_i": 1.0 if metadata.origin == "fused" else 0.0,
        }

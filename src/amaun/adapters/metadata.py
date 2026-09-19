"""M3 -- adapter metadata.

Carries every field brief S6 requires, plus ``provenance`` and ``tasks_served``, which
exist for reasons the brief implies but does not name:

- ``provenance`` records whether an adapter was trained, fused, or federated-aggregated,
  and from which parents.  Without it, "were the six initial adapters trained
  independently?" (reviewer falsification question 2) is unanswerable from artefacts alone.
- ``tasks_served`` is required by D-014: after a FUSE, one adapter serves several tasks, so
  reporting adapter count alone would make fusion look free.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AdapterOrigin(str, enum.Enum):
    TRAINED = "trained"          # trained from theta_0 on one task's TRAIN split
    FUSED = "fused"              # produced by layer-wise fusion + KD (M8)
    AGGREGATED = "aggregated"    # produced by federated aggregation (M11)
    COMPRESSED = "compressed"    # produced by utility-based repository compression (M9)


@dataclass
class AdapterMetadata:
    """Persisted next to the adapter weights as ``metadata.json``."""

    adapter_id: str
    task_name: str
    task_type: str
    dataset: str
    dataset_revision: str | None
    model_version: str            # backbone model id
    backbone_fingerprint: str     # ties the adapter to the exact frozen weights it was fitted to
    run_mode: str
    lora_rank: int | None
    target_modules: list[str]
    training_configuration: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    task_embedding: list[float] | None = None
    parameter_count: int = 0
    storage_bytes: int = 0
    resource_cost: dict[str, Any] = field(default_factory=dict)
    creation_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # -- provenance and lifecycle ------------------------------------------------
    origin: str = AdapterOrigin.TRAINED.value
    parent_adapter_ids: list[str] = field(default_factory=list)
    tasks_served: list[str] = field(default_factory=list)      # D-014 accounting
    fusion_coefficients_path: str | None = None
    created_at_task_index: int | None = None

    # -- usage history (feature family z_hist, idea.md S4.6) ---------------------
    use_count: int = 0
    merge_count: int = 0
    last_used_at_task_index: int | None = None
    performance_history: list[dict[str, Any]] = field(default_factory=list)
    archived: bool = False
    archived_reason: str | None = None

    content_hash: str | None = None
    schema_version: int = 1

    # -- IO ----------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=False)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AdapterMetadata":
        with Path(path).open() as handle:
            data = json.load(handle)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            # Fail loudly: silently dropping metadata fields would lose provenance.
            raise ValueError(f"unknown metadata fields in {path}: {sorted(unknown)}")
        return cls(**data)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- history helpers ---------------------------------------------------------
    def record_use(self, task_index: int) -> None:
        self.use_count += 1
        self.last_used_at_task_index = task_index

    def record_performance(self, task_id: str, metrics: dict[str, Any], task_index: int) -> None:
        self.performance_history.append(
            {"task_id": task_id, "task_index": task_index, "metrics": metrics}
        )

    def age(self, current_task_index: int) -> int:
        """``Age_i = t_current - t_created,i`` (idea.md S4.6)."""
        if self.created_at_task_index is None:
            return 0
        return max(0, current_task_index - self.created_at_task_index)

    def mean_historical_performance(self, score_key: str = "primary_score") -> float | None:
        """``PerfHist_i = (1/K) sum_k Acc_i,k`` (idea.md S4.6).

        Returns ``None`` rather than 0.0 when there is no history: a brand-new adapter has
        *unknown* historical performance, and encoding that as zero would bias the utility
        network against new adapters.  The feature layer substitutes a configured sentinel
        and counts the substitution.
        """
        scores = [
            entry["metrics"][score_key]
            for entry in self.performance_history
            if score_key in entry.get("metrics", {})
        ]
        return sum(scores) / len(scores) if scores else None

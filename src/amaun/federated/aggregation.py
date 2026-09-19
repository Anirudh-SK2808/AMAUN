"""M11 -- federated aggregation (idea.md S7.3).

    A_global_k = sum_{c in C_k} ( n_c / sum_j n_j ) * A_c,k

**Averaging is at the effective-delta level, never over A/B factors** (idea.md S16.3).  There
is deliberately no ``factor_fedavg`` option: the correct operation is
``mean_c(B_c A_c)``, and ``mean(B_c) @ mean(A_c)`` is a different (and wrong) matrix.

Three methods differ only in how the averaged delta is *stored*:

- ``delta_fedavg``: exact, via factor concatenation. Rank grows to ``sum_c r_c``.
- ``svd_refactor_fedavg``: projects the exact average back to rank ``r``, keeping adapter
  size constant across rounds. Reconstruction error is measured and reported.
- ``component_concat``: keeps all components; exact but grows every round, so it is only
  suitable for short-horizon diagnostics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Sequence

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.metadata import AdapterMetadata, AdapterOrigin
from amaun.adapters.repository import AdapterEntry
from amaun.errors import FederatedError
from amaun.federated.compatibility import CompatibilityGroup
from amaun.fusion.refactorize import FusionStorage, fuse_to_delta


class AggregationMethod(str, enum.Enum):
    DELTA_FEDAVG = "delta_fedavg"
    SVD_REFACTOR_FEDAVG = "svd_refactor_fedavg"
    COMPONENT_CONCAT = "component_concat"


@dataclass
class AggregationResult:
    group_key: str
    aggregated: AdapterEntry
    weights: dict[str, float]
    method: str
    n_clients: int
    total_examples: int
    reconstruction_error: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "aggregated_adapter_id": self.aggregated.adapter_id,
            "weights": self.weights,
            "method": self.method,
            "n_clients": self.n_clients,
            "total_examples": self.total_examples,
            "reconstruction_error": self.reconstruction_error,
            "notes": list(self.notes),
        }


def fedavg_weights(group: CompatibilityGroup) -> dict[str, float]:
    """``n_c / sum_j n_j``, verified to sum to 1 (invariant I11.2)."""
    total = group.total_examples
    if total <= 0:
        raise FederatedError(
            f"compatibility group '{group.key}' reports zero total examples; FedAvg weights "
            f"are undefined. n_c must be each client's real example count (invariant I11.3)."
        )
    weights = {update.client_id: update.n_examples / total for update in group.updates}
    drift = abs(sum(weights.values()) - 1.0)
    if drift > 1e-9:
        raise FederatedError(f"aggregation weights sum to {sum(weights.values())}, not 1.0")
    return weights


def aggregate_group(
    group: CompatibilityGroup,
    *,
    method: AggregationMethod | str,
    round_index: int,
    target_rank: int | None = None,
) -> AggregationResult:
    """Aggregate one compatibility group into a single global adapter."""
    method = AggregationMethod(method)
    weights = fedavg_weights(group)
    notes: list[str] = []

    if len(group.updates) == 1:
        # A singleton group is a legitimate no-op, not an error.
        only = group.updates[0]
        notes.append("single-client group: aggregation is an identity operation")
        return AggregationResult(
            group_key=group.key,
            aggregated=only.entry,
            weights=weights,
            method=method.value,
            n_clients=1,
            total_examples=group.total_examples,
            notes=notes,
        )

    deltas: list[AdapterDelta] = [update.entry.delta for update in group.updates]
    coefficients = [weights[update.client_id] for update in group.updates]

    storage = {
        AggregationMethod.DELTA_FEDAVG: FusionStorage.KEEP_COMPONENTS,
        AggregationMethod.COMPONENT_CONCAT: FusionStorage.KEEP_COMPONENTS,
        AggregationMethod.SVD_REFACTOR_FEDAVG: FusionStorage.REFACTORIZE_SVD,
    }[method]

    aggregated_id = f"global-{group.key.replace('|', '_')}-r{round_index}"
    delta, report = fuse_to_delta(
        deltas, coefficients, storage=storage, target_rank=target_rank, adapter_id=aggregated_id
    )
    if method is AggregationMethod.DELTA_FEDAVG and target_rank is not None:
        notes.append(
            f"delta_fedavg is exact; stored rank is sum of client ranks ({delta.rank}), which "
            f"grows with participation. Use svd_refactor_fedavg to hold rank constant."
        )

    template = group.updates[0].entry.metadata
    metadata = AdapterMetadata(
        adapter_id=aggregated_id,
        task_name=template.task_name,
        task_type=template.task_type,
        dataset=template.dataset,
        dataset_revision=template.dataset_revision,
        model_version=template.model_version,
        backbone_fingerprint=template.backbone_fingerprint,
        run_mode=template.run_mode,
        lora_rank=delta.rank,
        target_modules=list(template.target_modules),
        origin=AdapterOrigin.AGGREGATED.value,
        parent_adapter_ids=[u.adapter_id for u in group.updates],
        tasks_served=sorted({task for u in group.updates for task in u.entry.metadata.tasks_served}),
        parameter_count=delta.parameter_count,
        storage_bytes=delta.storage_bytes(),
        training_configuration={
            "aggregation_method": method.value,
            "round_index": round_index,
            "weights": weights,
        },
    )

    return AggregationResult(
        group_key=group.key,
        aggregated=AdapterEntry(delta=delta, metadata=metadata),
        weights=weights,
        method=method.value,
        n_clients=len(group.updates),
        total_examples=group.total_examples,
        reconstruction_error=(
            report.max_relative_error if method is AggregationMethod.SVD_REFACTOR_FEDAVG else None
        ),
        notes=notes,
    )

"""M11 -- the federated server: one round of compatibility-gated aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from amaun.federated.aggregation import AggregationResult, aggregate_group
from amaun.federated.comm import CommunicationLedger, build_payload
from amaun.federated.compatibility import ClientUpdate, CompatibilityGroup, group_updates
from amaun.utils.logging import get_logger

log = get_logger("federated.server")


@dataclass
class RoundReport:
    round_index: int
    participating_clients: list[str]
    groups: list[dict[str, Any]]
    aggregations: list[dict[str, Any]]
    payload_bytes_total: int
    payload_bytes_per_client: dict[str, int]
    n_incompatible_singletons: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class FederatedServer:
    """Aggregates only compatible adapter states; never sees raw client data."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.ledger = CommunicationLedger()
        self.rounds: list[RoundReport] = []

    def run_round(
        self,
        round_index: int,
        updates: Sequence[ClientUpdate],
        *,
        target_rank: int | None = None,
    ) -> tuple[list[AggregationResult], RoundReport]:
        """Group, measure, aggregate.  Raw data never leaves the client (idea.md S7.3)."""
        if not updates:
            report = RoundReport(
                round_index=round_index, participating_clients=[], groups=[], aggregations=[],
                payload_bytes_total=0, payload_bytes_per_client={},
                n_incompatible_singletons=0, notes=["no participating clients this round"],
            )
            self.rounds.append(report)
            return [], report

        # Measure real payload sizes, and verify no backbone tensor is present.
        per_client: dict[str, int] = {}
        for update in updates:
            payload = build_payload(
                client_id=update.client_id,
                delta=update.entry.delta,
                metadata=update.entry.metadata,
            )
            size = payload.size_bytes()
            per_client[update.client_id] = per_client.get(update.client_id, 0) + size
            self.ledger.record(round_index, update.client_id, size)

        groups = group_updates(updates, config=self.config.get("compatibility") or {})
        method = (self.config.get("aggregation") or {}).get("method", "delta_fedavg")

        results: list[AggregationResult] = []
        for group in groups:
            results.append(
                aggregate_group(group, method=method, round_index=round_index, target_rank=target_rank)
            )

        singletons = sum(1 for group in groups if len(group.updates) == 1)
        report = RoundReport(
            round_index=round_index,
            participating_clients=sorted({u.client_id for u in updates}),
            groups=[group.as_dict() for group in groups],
            aggregations=[result.as_dict() for result in results],
            payload_bytes_total=sum(per_client.values()),
            payload_bytes_per_client=per_client,
            n_incompatible_singletons=singletons,
            notes=(
                [f"{singletons} group(s) had a single client and were not averaged"]
                if singletons else []
            ),
        )
        self.rounds.append(report)
        log.info(
            "round %d: %d clients -> %d compatibility groups, %d bytes transmitted",
            round_index, len(report.participating_clients), len(groups), report.payload_bytes_total,
        )
        return results, report

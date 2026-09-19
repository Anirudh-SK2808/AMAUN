"""M11 -- explicit compatibility gating (idea.md S7.3).

    C_k = { c : client c reports compatible adapter state k }

idea.md is emphatic that the server "must determine compatibility among adapter states
before aggregating -- never blindly average unrelated adapters (this would defeat the purpose
of the whole compatibility/utility mechanism)", but it does not define compatibility.  A-10
records the definition used: identical backbone fingerprint, identical target modules, and
identical rank, plus a configurable grouping key.

The grouping key is the experimental variable.  ``task_id`` groups adapters serving the same
task; ``adapter_lineage`` groups by shared provenance, which is the only sensible key once
fusion has produced adapters that serve several tasks; ``utility_cluster`` defers to a
caller-supplied clustering.
"""

from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from amaun.adapters.repository import AdapterEntry
from amaun.errors import IncompatibleAdapterError


class CompatibilityKey(str, enum.Enum):
    TASK_ID = "task_id"
    ADAPTER_LINEAGE = "adapter_lineage"
    UTILITY_CLUSTER = "utility_cluster"


@dataclass
class ClientUpdate:
    """One client's contribution to a round."""

    client_id: str
    entry: AdapterEntry
    n_examples: int             # n_c, the FedAvg weight numerator
    task_ids: list[str] = field(default_factory=list)
    round_index: int = 0

    @property
    def adapter_id(self) -> str:
        return self.entry.adapter_id


@dataclass
class CompatibilityGroup:
    key: str
    updates: list[ClientUpdate]

    @property
    def total_examples(self) -> int:
        return sum(update.n_examples for update in self.updates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n_clients": len(self.updates),
            "client_ids": [u.client_id for u in self.updates],
            "adapter_ids": [u.adapter_id for u in self.updates],
            "n_examples": [u.n_examples for u in self.updates],
            "total_examples": self.total_examples,
        }


def structural_signature(update: ClientUpdate, *, config: Mapping[str, Any]) -> tuple:
    """The structural part of compatibility: what must match for averaging to be defined."""
    metadata = update.entry.metadata
    delta = update.entry.delta
    parts: list[Any] = []
    if config.get("require_same_backbone", True):
        parts.append(metadata.backbone_fingerprint)
    if config.get("require_same_rank", True):
        parts.append(delta.rank)
    if config.get("require_same_target_modules", True):
        parts.append(tuple(sorted(metadata.target_modules)))
    # Layer key sets must always match: averaging deltas defined over different layers is
    # not a meaningful operation regardless of configuration.
    parts.append(tuple(delta.keys()))
    return tuple(parts)


def grouping_value(
    update: ClientUpdate,
    key: CompatibilityKey,
    *,
    cluster_fn: Callable[[ClientUpdate], str] | None = None,
) -> str:
    if key is CompatibilityKey.TASK_ID:
        return "|".join(sorted(update.entry.metadata.tasks_served)) or update.entry.metadata.task_name
    if key is CompatibilityKey.ADAPTER_LINEAGE:
        lineage = sorted(update.entry.metadata.parent_adapter_ids) or [update.adapter_id]
        return "|".join(lineage)
    if key is CompatibilityKey.UTILITY_CLUSTER:
        if cluster_fn is None:
            raise IncompatibleAdapterError(
                "compatibility.key='utility_cluster' requires a clustering function; none supplied"
            )
        return cluster_fn(update)
    raise IncompatibleAdapterError(f"unknown compatibility key '{key}'")


def group_updates(
    updates: Sequence[ClientUpdate],
    *,
    config: Mapping[str, Any],
    cluster_fn: Callable[[ClientUpdate], str] | None = None,
) -> list[CompatibilityGroup]:
    """Partition client updates into compatibility groups ``C_k``.

    A group of size 1 is returned, not dropped: aggregating a singleton is a legitimate
    no-op, and silently discarding it would lose that client's contribution.
    """
    key = CompatibilityKey(config.get("key", CompatibilityKey.TASK_ID.value))
    buckets: dict[tuple, list[ClientUpdate]] = defaultdict(list)
    for update in updates:
        buckets[(grouping_value(update, key, cluster_fn=cluster_fn), structural_signature(update, config=config))].append(update)

    groups = []
    for (group_value, signature), members in sorted(buckets.items(), key=lambda kv: str(kv[0][0])):
        groups.append(CompatibilityGroup(key=str(group_value), updates=members))
    return groups

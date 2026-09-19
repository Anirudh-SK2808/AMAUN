"""M11 -- federated clients and data partitioning.

The four axes are kept independent (brief S13, D-011).  ``one client = one task`` is never
assumed: a client holds a *shard schedule* mapping task ids to the example indices it owns,
and several clients may hold different shards of the same task, or different task subsets.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from amaun.data.splits import Example, SplitName, TaskSplits
from amaun.errors import FederatedError


class PartitionScheme(str, enum.Enum):
    IID_SHARDS = "iid_shards"                    # same task sequence, disjoint data shards
    DISJOINT_TASK_SUBSETS = "disjoint_task_subsets"   # different task subsets per client
    DIRICHLET = "dirichlet"                      # skewed proportions per client


@dataclass
class ClientState:
    """One federated client: its own repository, task view, and data shards."""

    client_id: str
    task_ids: list[str]
    shards: dict[str, list[int]] = field(default_factory=dict)   # task_id -> example indices
    round_participation: list[int] = field(default_factory=list)

    def n_examples(self, task_id: str) -> int:
        """``n_c`` for FedAvg weighting (invariant I11.3)."""
        return len(self.shards.get(task_id, []))

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "task_ids": list(self.task_ids),
            "shard_sizes": {task: len(indices) for task, indices in self.shards.items()},
            "rounds_participated": list(self.round_participation),
        }


def _dirichlet_proportions(n_clients: int, alpha: float, rng: random.Random) -> list[float]:
    """Dirichlet sample via normalised Gammas (avoids a numpy dependency here)."""
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(n_clients)]
    total = sum(samples) or 1.0
    return [value / total for value in samples]


def partition_clients(
    splits_by_task: Mapping[str, TaskSplits],
    *,
    config: Mapping[str, Any],
    seed: int,
) -> list[ClientState]:
    """Build client states under the declared partition scheme."""
    scheme_name = config.get("partition")
    if not scheme_name:
        raise FederatedError(
            "federated.partition is required and has no default (docs/decision_register.md D-011)"
        )
    scheme = PartitionScheme(scheme_name)
    n_clients = int(config["num_clients"])
    if n_clients < 1:
        raise FederatedError("federated.num_clients must be >= 1")

    rng = random.Random(seed)
    task_ids = sorted(splits_by_task)
    clients = [ClientState(client_id=f"c{i}", task_ids=[]) for i in range(n_clients)]

    if scheme is PartitionScheme.DISJOINT_TASK_SUBSETS:
        # Each client sees a different (possibly overlapping) subset of tasks; every task
        # goes to at least one client so no task is silently dropped from the experiment.
        shuffled = list(task_ids)
        rng.shuffle(shuffled)
        for index, task_id in enumerate(shuffled):
            owner = clients[index % n_clients]
            owner.task_ids.append(task_id)
            owner.shards[task_id] = list(range(len(splits_by_task[task_id][SplitName.TRAIN])))
        for client in clients:
            client.task_ids.sort()
        return clients

    for task_id in task_ids:
        train_size = len(splits_by_task[task_id][SplitName.TRAIN])
        indices = list(range(train_size))
        rng.shuffle(indices)

        if scheme is PartitionScheme.IID_SHARDS:
            proportions = [1.0 / n_clients] * n_clients
        else:  # DIRICHLET
            alpha = float(config.get("dirichlet_alpha", 0.5))
            proportions = _dirichlet_proportions(n_clients, alpha, rng)

        cursor = 0
        for client_index, client in enumerate(clients):
            if client_index == n_clients - 1:
                shard = indices[cursor:]
            else:
                take = int(round(proportions[client_index] * train_size))
                shard = indices[cursor : cursor + take]
                cursor += take
            client.task_ids.append(task_id)
            client.shards[task_id] = sorted(shard)

    return clients


def select_participants(
    clients: Sequence[ClientState], *, participation_rate: float, round_index: int, seed: int
) -> list[ClientState]:
    """Sample participating clients for a round (partial participation)."""
    if not 0.0 < participation_rate <= 1.0:
        raise FederatedError("federated.participation_rate must be in (0, 1]")
    count = max(1, int(round(participation_rate * len(clients))))
    rng = random.Random(f"{seed}:{round_index}")
    chosen = rng.sample(list(clients), k=min(count, len(clients)))
    for client in chosen:
        client.round_participation.append(round_index)
    return sorted(chosen, key=lambda c: c.client_id)

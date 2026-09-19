"""M11 -- federated axes, aggregation, communication (criteria G4-G7)."""

import pytest
import torch

from amaun.adapters.repository import AdapterEntry
from amaun.errors import FederatedError, PayloadContainsBackboneError
from amaun.federated.aggregation import AggregationMethod, aggregate_group, fedavg_weights
from amaun.federated.client import PartitionScheme, partition_clients, select_participants
from amaun.federated.comm import CommunicationLedger, Payload, assert_no_backbone, build_payload
from amaun.federated.compatibility import ClientUpdate, CompatibilityGroup, group_updates
from amaun.federated.server import FederatedServer
from tests.conftest import make_delta, make_metadata


def _update(client_id, adapter_id, *, rank=4, fingerprint="fp", n_examples=100, tasks=("t1",)):
    delta = make_delta(hash(adapter_id) % 1000, adapter_id, rank=rank)
    entry = AdapterEntry(
        delta=delta,
        metadata=make_metadata(
            adapter_id, lora_rank=rank, backbone_fingerprint=fingerprint, tasks_served=list(tasks)
        ),
    )
    return ClientUpdate(client_id=client_id, entry=entry, n_examples=n_examples, task_ids=list(tasks))


# -- compatibility -------------------------------------------------------------------
def test_incompatible_adapters_are_never_averaged():
    """idea.md S7.3: blindly averaging unrelated adapters defeats the whole mechanism."""
    updates = [
        _update("c0", "A0"),
        _update("c1", "A1"),
        _update("c2", "A2", rank=8),                  # different rank
        _update("c3", "A3", fingerprint="other"),     # different backbone
    ]
    groups = group_updates(updates, config={"key": "task_id", "require_same_rank": True,
                                            "require_same_target_modules": True,
                                            "require_same_backbone": True})
    sizes = sorted(len(group.updates) for group in groups)
    assert sizes == [1, 1, 2]     # the two compatible clients group; the others stay alone


def test_disabling_the_gate_merges_everything():
    """F2 control: with the gate off, everything lands in one bucket."""
    updates = [_update("c0", "A0"), _update("c1", "A1", rank=8), _update("c2", "A2", fingerprint="x")]
    groups = group_updates(updates, config={"key": "task_id", "require_same_rank": False,
                                            "require_same_target_modules": False,
                                            "require_same_backbone": False})
    # Layer key sets still have to match -- averaging over different layers is undefined.
    assert len(groups) == 1
    assert len(groups[0].updates) == 3


def test_lineage_key_groups_by_provenance():
    a = _update("c0", "A0", tasks=("t1",))
    b = _update("c1", "A1", tasks=("t2",))
    a.entry.metadata.parent_adapter_ids = ["P"]
    b.entry.metadata.parent_adapter_ids = ["P"]
    groups = group_updates([a, b], config={"key": "adapter_lineage"})
    assert len(groups) == 1 and len(groups[0].updates) == 2


# -- aggregation ---------------------------------------------------------------------
def test_weights_sum_to_one_and_are_proportional_to_example_counts():
    group = CompatibilityGroup("t1", [_update("c0", "A0", n_examples=100),
                                      _update("c1", "A1", n_examples=300)])
    weights = fedavg_weights(group)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["c0"] == pytest.approx(0.25)
    assert weights["c1"] == pytest.approx(0.75)


def test_zero_example_group_is_refused():
    group = CompatibilityGroup("t1", [_update("c0", "A0", n_examples=0)])
    with pytest.raises(FederatedError):
        fedavg_weights(group)


def test_aggregation_is_exact_at_the_delta_level():
    """Never A/B averaging (idea.md S16.3): verify against the weighted dense sum."""
    updates = [_update("c0", "A0", n_examples=100), _update("c1", "A1", n_examples=300)]
    group = CompatibilityGroup("t1", updates)
    result = aggregate_group(group, method=AggregationMethod.DELTA_FEDAVG, round_index=0)
    for key in updates[0].entry.delta.keys():
        expected = (
            result.weights["c0"] * updates[0].entry.delta[key].dense()
            + result.weights["c1"] * updates[1].entry.delta[key].dense()
        )
        assert torch.allclose(result.aggregated.delta[key].dense(), expected, atol=1e-5)


def test_singleton_group_is_an_identity_not_an_error():
    group = CompatibilityGroup("t1", [_update("c0", "A0")])
    result = aggregate_group(group, method="delta_fedavg", round_index=0)
    assert result.n_clients == 1
    assert result.aggregated.adapter_id == "A0"
    assert any("single-client" in note for note in result.notes)


def test_svd_refactor_holds_rank_constant_and_reports_error():
    updates = [_update("c0", "A0"), _update("c1", "A1")]
    group = CompatibilityGroup("t1", updates)
    result = aggregate_group(
        group, method=AggregationMethod.SVD_REFACTOR_FEDAVG, round_index=0, target_rank=4
    )
    assert result.aggregated.delta.rank == 4
    assert result.reconstruction_error is not None and result.reconstruction_error >= 0.0


def test_aggregated_metadata_records_provenance():
    group = CompatibilityGroup("t1", [_update("c0", "A0", tasks=("t1",)),
                                      _update("c1", "A1", tasks=("t2",))])
    result = aggregate_group(group, method="delta_fedavg", round_index=3)
    metadata = result.aggregated.metadata
    assert metadata.origin == "aggregated"
    assert set(metadata.parent_adapter_ids) == {"A0", "A1"}
    assert set(metadata.tasks_served) == {"t1", "t2"}


# -- communication -------------------------------------------------------------------
def test_backbone_tensors_in_a_payload_are_refused():
    """Master-prompt constraint 4."""
    for key in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        with pytest.raises(PayloadContainsBackboneError):
            assert_no_backbone(Payload("a", "c0", {key: torch.zeros(4)}, {}))


def test_oversized_tensor_in_a_payload_is_refused():
    huge = torch.zeros(1)
    huge_numel = type("T", (), {"numel": lambda self: 60_000_000})()
    with pytest.raises(PayloadContainsBackboneError):
        assert_no_backbone(Payload("a", "c0", {"weird": huge_numel}, {}))


def test_payload_contains_only_lora_factors():
    update = _update("c0", "A0")
    payload = build_payload(client_id="c0", delta=update.entry.delta, metadata=update.entry.metadata)
    assert payload.tensors
    for key in payload.tensors:
        assert key.endswith(("::A", "::B", "::scaling"))
    assert payload.size_bytes() > 0


def test_communication_ratio_is_like_for_like():
    """Cumulative LoRA traffic must be compared against the SAME number of transmissions."""
    ledger = CommunicationLedger()
    ledger.conceptual_full_model_bytes = 1000
    for index in range(4):
        ledger.record(0, f"c{index}", 10)
    assert ledger.total_bytes == 40
    assert ledger.n_transmissions == 4
    assert ledger.counterfactual_full_model_bytes() == 4000
    assert ledger.savings_vs_full_model() == pytest.approx(0.01)


# -- axes ----------------------------------------------------------------------------
def test_partition_requires_an_explicit_scheme(splits):
    """D-011: one client = one task is never assumed, and there is no default."""
    with pytest.raises(FederatedError):
        partition_clients(splits, config={"num_clients": 2}, seed=0)


def test_iid_shards_give_every_client_every_task(splits):
    clients = partition_clients(
        splits, config={"partition": "iid_shards", "num_clients": 3}, seed=0
    )
    assert len(clients) == 3
    for client in clients:
        assert set(client.task_ids) == set(splits)          # CLIENT != TASK
    # Shards are disjoint: the union of client shard sizes equals the train pool.
    for task_id in splits:
        total = sum(client.n_examples(task_id) for client in clients)
        from amaun.data.splits import SplitName

        assert total == len(splits[task_id][SplitName.TRAIN])


def test_disjoint_task_subsets_spreads_tasks_without_dropping_any(splits):
    clients = partition_clients(
        splits, config={"partition": "disjoint_task_subsets", "num_clients": 3}, seed=0
    )
    covered = {task for client in clients for task in client.task_ids}
    assert covered == set(splits)


def test_dirichlet_partition_is_skewed_but_complete(splits):
    clients = partition_clients(
        splits, config={"partition": "dirichlet", "num_clients": 4, "dirichlet_alpha": 0.1},
        seed=0,
    )
    from amaun.data.splits import SplitName

    task_id = sorted(splits)[0]
    sizes = [client.n_examples(task_id) for client in clients]
    assert sum(sizes) == len(splits[task_id][SplitName.TRAIN])
    assert len(set(sizes)) > 1, "a low-alpha Dirichlet partition should be uneven"


def test_participation_sampling(splits):
    clients = partition_clients(
        splits, config={"partition": "iid_shards", "num_clients": 4}, seed=0
    )
    chosen = select_participants(clients, participation_rate=0.5, round_index=0, seed=0)
    assert len(chosen) == 2
    with pytest.raises(FederatedError):
        select_participants(clients, participation_rate=0.0, round_index=0, seed=0)


def test_round_axis_is_independent_of_task_and_client(splits):
    """ROUND is its own axis: the same clients can participate in several rounds."""
    server = FederatedServer({"compatibility": {"key": "task_id"},
                              "aggregation": {"method": "delta_fedavg"}})
    updates = [_update("c0", "A0"), _update("c1", "A1")]
    for round_index in range(3):
        _results, report = server.run_round(round_index, updates)
        assert report.round_index == round_index
    assert len(server.rounds) == 3
    assert server.ledger.n_transmissions == 6


def test_empty_round_is_handled():
    server = FederatedServer({"compatibility": {}, "aggregation": {"method": "delta_fedavg"}})
    results, report = server.run_round(0, [])
    assert results == []
    assert report.payload_bytes_total == 0

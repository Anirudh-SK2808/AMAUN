"""M9 -- utility-based repository compression (criterion D5, idea.md S7.2 / S16.7)."""

import pytest

from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.manager.compression import (
    CompressionMechanism,
    heuristic_pair_utility,
    plan_compression,
    should_compress,
)
from tests.conftest import make_delta, make_metadata


class _Decision:
    outcome = "COMPRESS"
    decision_id = "compress-1"


def _repository(tmp_path, n=4):
    repository = AdapterRepository(tmp_path / "repo", backbone_fingerprint="fp")
    for index in range(n):
        repository.register(
            AdapterEntry(
                delta=make_delta(index + 1, f"A{index}"),
                metadata=make_metadata(
                    f"A{index}", tasks_served=[f"t{index}"], created_at_task_index=index,
                    use_count=index,
                ),
            ),
            bootstrap=True, task_index=index,
        )
    return repository


def test_no_trigger_when_budget_is_null(tmp_path):
    """A null R_max disables compression -- that is ablation A5, not an error."""
    repository = _repository(tmp_path)
    triggered, reason = should_compress(repository, None)
    assert not triggered
    assert "A5" in reason


def test_trigger_fires_at_the_memory_budget(tmp_path):
    repository = _repository(tmp_path)
    current = repository.memory_bytes()
    assert should_compress(repository, current)[0] is True        # >= is the condition
    assert should_compress(repository, current + 1)[0] is False


def test_plan_picks_the_argmax_utility_pair(tmp_path):
    repository = _repository(tmp_path)
    # Make one specific pair the clear maximum.
    def pair_utility(left, right):
        return 10.0 if {left.adapter_id, right.adapter_id} == {"A1", "A2"} else 0.0

    plan, evaluated = plan_compression(
        repository, pair_utility=pair_utility, mechanism=CompressionMechanism.UTILITY, min_adapters=2
    )
    assert plan is not None
    assert {plan.first_id, plan.second_id} == {"A1", "A2"}
    assert plan.utility == 10.0
    assert plan.mechanism == "utility"
    assert evaluated == 6      # C(4,2)


def test_plan_respects_the_minimum_adapter_floor(tmp_path):
    """Invariant I9.2: compression must not collapse the repository to a single adapter."""
    repository = _repository(tmp_path, n=2)
    plan, _ = plan_compression(
        repository, pair_utility=lambda a, b: 1.0,
        mechanism=CompressionMechanism.UTILITY, min_adapters=2,
    )
    assert plan is None


def test_compression_strictly_reduces_the_repository(tmp_path):
    repository = _repository(tmp_path, n=4)
    before = len(repository)
    merged = AdapterEntry(
        delta=make_delta(99, "AM"), metadata=make_metadata("AM", tasks_served=[])
    )
    repository.replace_pair_with_merged(
        first_id="A0", second_id="A1", merged_entry=merged, authorised_by=_Decision()
    )
    assert len(repository) == before - 1
    # D-014: the merged adapter must inherit what the pair served.
    assert set(repository.retrieve("AM").metadata.tasks_served) == {"t0", "t1"}
    assert {"A0", "A1"} <= set(repository.archived_ids())


def test_heuristic_is_available_only_as_a_baseline(tmp_path):
    """idea.md S16.7: the heuristic exists for B6, never as AMAUN's mechanism."""
    repository = _repository(tmp_path)
    left, right = repository.retrieve("A0"), repository.retrieve("A3")
    score = heuristic_pair_utility(left, right, task_index=10)
    assert isinstance(score, float)
    plan, _ = plan_compression(
        repository, pair_utility=lambda a, b: heuristic_pair_utility(a, b, task_index=10),
        mechanism=CompressionMechanism.HEURISTIC, min_adapters=2,
    )
    assert plan is not None
    assert plan.mechanism == "heuristic"
    assert "baseline B6" in plan.rationale


def test_heuristic_prefers_old_and_unused_adapters(tmp_path):
    repository = _repository(tmp_path, n=4)
    old_unused = (repository.retrieve("A0"), repository.retrieve("A1"))   # low use_count
    new_used = (repository.retrieve("A2"), repository.retrieve("A3"))     # higher use_count
    assert heuristic_pair_utility(*old_unused, task_index=10) > heuristic_pair_utility(
        *new_used, task_index=10
    )

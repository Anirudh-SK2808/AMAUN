"""M2 -- split isolation and leakage guards (criteria B2, B3, B4)."""

import pytest

from amaun.data.leakage import audit
from amaun.data.providers import synthetic_pool
from amaun.data.splits import (
    SplitName,
    SplitRatios,
    build_splits,
    evaluator_guard,
    manager_guard,
    trainer_guard,
    utility_guard,
)
from amaun.errors import LeakageError, SplitAccessViolation


def test_all_four_splits_are_disjoint_by_content(splits):
    for task_splits in splits.values():
        task_splits.assert_disjoint()          # raises on overlap
        names = list(SplitName)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                assert not (task_splits.hashes(left) & task_splits.hashes(right))


def test_support_never_comes_from_the_test_pool(splits):
    """Assumption A-05: support access must never imply test access."""
    for task_splits in splits.values():
        assert not (task_splits.hashes(SplitName.SUPPORT) & task_splits.hashes(SplitName.TEST))


def test_split_assignment_determinism(session_cfg, tasks):
    ratios = SplitRatios.from_config(session_cfg.get("data.split_ratios"))
    task = tasks[0]
    pool = synthetic_pool(task, 120, seed=5)
    test = synthetic_pool(task, 20, seed=5, offset=10**6)
    first = build_splits(task.task_id, training_pool=pool, test_pool=test, seed=11, ratios=ratios)
    second = build_splits(task.task_id, training_pool=pool, test_pool=test, seed=11, ratios=ratios)
    third = build_splits(task.task_id, training_pool=pool, test_pool=test, seed=12, ratios=ratios)
    assert first.hashes(SplitName.SUPPORT) == second.hashes(SplitName.SUPPORT)
    assert first.hashes(SplitName.SUPPORT) != third.hashes(SplitName.SUPPORT)


def test_split_assignment_is_content_based_not_index_based(session_cfg, tasks):
    """Reshuffling the input must not reassign examples between splits."""
    ratios = SplitRatios.from_config(session_cfg.get("data.split_ratios"))
    task = tasks[0]
    pool = synthetic_pool(task, 80, seed=3)
    test = synthetic_pool(task, 10, seed=3, offset=10**6)
    ordered = build_splits(task.task_id, training_pool=pool, test_pool=test, seed=7, ratios=ratios)
    shuffled = build_splits(
        task.task_id, training_pool=list(reversed(pool)), test_pool=test, seed=7, ratios=ratios
    )
    assert ordered.hashes(SplitName.SUPPORT) == shuffled.hashes(SplitName.SUPPORT)
    assert ordered.hashes(SplitName.TRAIN) == shuffled.hashes(SplitName.TRAIN)


def test_guard_manager_cannot_read_test(splits):
    guard = manager_guard(next(iter(splits.values())))
    assert guard.get(SplitName.SUPPORT)
    for forbidden in (SplitName.TEST, SplitName.TRAIN, SplitName.VALIDATION):
        with pytest.raises(SplitAccessViolation):
            guard.get(forbidden)


def test_guard_trainer_cannot_read_test_or_support(splits):
    guard = trainer_guard(next(iter(splits.values())))
    assert guard.get(SplitName.TRAIN)
    with pytest.raises(SplitAccessViolation):
        guard.get(SplitName.TEST)
    with pytest.raises(SplitAccessViolation):
        guard.get(SplitName.SUPPORT)


def test_guard_evaluator_reads_only_test(splits):
    guard = evaluator_guard(next(iter(splits.values())))
    assert guard.get(SplitName.TEST)
    with pytest.raises(SplitAccessViolation):
        guard.get(SplitName.TRAIN)


def test_guard_utility_cannot_read_test(splits):
    """idea.md S16.6: the utility network must never see the test sequence."""
    guard = utility_guard(next(iter(splits.values())))
    for allowed in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.SUPPORT):
        assert guard.get(allowed) is not None
    with pytest.raises(SplitAccessViolation):
        guard.get(SplitName.TEST)


def test_corpus_wide_audit_is_clean(splits):
    report = audit(splits)
    assert report.clean
    assert report.pairs_checked > 0


def test_audit_detects_injected_test_leak(splits):
    """The audit must catch a leak the per-task check cannot see."""
    tampered = dict(splits)
    first, second = list(tampered)[:2]
    leaked = tampered[second][SplitName.TEST][0]
    tampered[first].splits[SplitName.TRAIN].append(leaked)
    with pytest.raises(LeakageError):
        audit(tampered)


def test_smoke_subsampling_applies_to_all_splits(splits):
    """idea.md S0.2.1: one config key, applied uniformly."""
    task_splits = next(iter(splits.values()))
    reduced = task_splits.subsample(3, seed=1)
    for name in SplitName:
        assert len(reduced[name]) <= 3
    reduced.assert_disjoint()

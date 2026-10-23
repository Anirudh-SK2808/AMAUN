"""D-025: a training-pool row equal to a TEST row of the same task leaves the pool; TEST is kept.

Some corpora repeat rows across their upstream splits (DailyDialog has 9 dialogues in both train
and validation), so the training pool can contain a TEST row.  ``prepare_data`` drops it before
the split, instead of failing the disjointness check, and records how many rows it dropped.
"""

from amaun.data.splits import SplitName
from amaun.experiment import prepare_data

INJECTED = 3


def _inject_pool_rows_into_test(monkeypatch, task_id):
    """Make ``task_id``'s TEST pool contain its first ``INJECTED`` training-pool rows."""
    import amaun.experiment as experiment

    real = experiment.synthetic_pool
    pools = {}

    def patched(task, size, *, seed, offset=0, stream_key=None):
        rows = real(task, size, seed=seed, offset=offset, stream_key=stream_key)
        if task.task_id != task_id:
            return rows
        if offset < 1_000_000:
            pools[task_id] = rows
            return rows
        return list(rows) + list(pools[task_id][:INJECTED])

    monkeypatch.setattr(experiment, "synthetic_pool", patched)
    return pools


def test_pool_rows_found_in_test_leave_the_pool_and_test_is_unchanged(cfg, monkeypatch):
    baseline = prepare_data(cfg)
    task_id = next(iter(baseline.splits))
    pools = _inject_pool_rows_into_test(monkeypatch, task_id)
    data = prepare_data(cfg)

    splits = data.splits[task_id]
    injected = {e.content_hash for e in pools[task_id][:INJECTED]}
    # TEST keeps every row of its pool, injected ones included (the dummy caps then sample it).
    assert splits.provenance["test_pool_size"] == baseline.splits[task_id].provenance["test_pool_size"] + INJECTED
    for name in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.SUPPORT):
        assert not injected & splits.content_hashes(name)
    assert splits.provenance["excluded_pool_rows_in_test"] == INJECTED
    assert data.leakage_report["clean"] is True


def test_tasks_without_overlap_keep_identical_splits(cfg, monkeypatch):
    baseline = prepare_data(cfg)
    task_id = next(iter(baseline.splits))
    _inject_pool_rows_into_test(monkeypatch, task_id)
    data = prepare_data(cfg)

    for other, splits in data.splits.items():
        if other == task_id:
            continue
        assert splits.provenance["excluded_pool_rows_in_test"] == 0
        assert splits.digests() == baseline.splits[other].digests()

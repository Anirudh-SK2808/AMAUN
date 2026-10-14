"""FR-4 -- carving a disjoint TEST pool from a single-split corpus (AC-12).  Offline.

``databricks/databricks-dolly-15k`` has exactly one upstream split (`train`, 15 011 rows), so a
TEST pool drawn from it without an offset is a *prefix of the training pool*: TEST subset TRAIN.
This file proves three separate things, and the middle one is what licenses trusting the fix:

(a) with an offset >= the training pool's measured ``rows_consumed``, the two pools share no
    ``example_id``;
(b) with ``test_offset = 0`` -- the pre-fix behaviour -- ``build_splits`` **raises**
    ``LeakageError``.  This is a property of the existing ``splits.py`` machinery, asserted rather
    than assumed: without it, (a) would prove only that two windows differ, not that a regression
    would be caught;
(c) ``prepare_data`` refuses an offset smaller than the training pool's measured row count;
(d) under ``sampling: shuffled_buffer``, two independently constructed streams for the same task
    are shuffled into **identical row order** -- without which ``skip`` carves nothing (FR-3.8).
"""

from __future__ import annotations

import random

import pytest

from amaun.config import resolve
from amaun.data.providers import derive_shuffle_seed, huggingface_pool
from amaun.data.registry import task_spec_from_config
from amaun.data.row_adapters import resolve as resolve_adapter
from amaun.data.splits import SplitRatios, build_splits
from amaun.errors import DataError, LeakageError
from amaun.experiment import prepare_data

TOTAL_ROWS = 400            # a fixture stand-in for dolly's single 15 011-row split
POOL_ROWS = 60
TEST_ROWS = 20

INSTR_SPEC = task_spec_from_config(
    "instr",
    {
        "task_name": "Instruction Following", "task_type": "instruction_following",
        "primary_score": "rougeL", "metrics": ["rougeL"],
        "input_fields": ["instruction", "context"], "target_field": "response",
        "prompt_template": "{instruction}\n\n{context}\n\nResponse:",
        "max_input_tokens": 64, "max_target_tokens": 32,
        "hf_id": "fixture/dolly", "hf_revision": "c" * 40, "hf_config": "default",
        "row_adapter": "passthrough", "row_adapter_options": {"list_target_pick": "first"},
    },
)

# Every row is uniquely identifiable, so an overlap is unambiguous rather than coincidental.
ROWS = [
    {"instruction": f"instruction {i}", "context": f"context {i}", "response": f"response {i}"}
    for i in range(TOTAL_ROWS)
]


class _FakeIterable:
    """A stand-in for ``datasets.IterableDataset`` implementing only shuffle/skip/iterate.

    ``shuffle`` performs a *real* seeded permutation so the ordering assertion in (d) is
    meaningful: a fake that ignored the seed could not distinguish a deterministic derivation from
    a per-call random one.
    """

    def __init__(self, items):
        self._items = list(items)

    def shuffle(self, seed, buffer_size):
        items = list(self._items)
        random.Random(seed).shuffle(items)
        return _FakeIterable(items[: max(buffer_size, len(items))])

    def skip(self, n):
        return _FakeIterable(self._items[n:])

    def __iter__(self):
        return iter(self._items)


@pytest.fixture
def fake_loader(monkeypatch):
    calls: list[dict] = []

    def loader(*args, **kwargs):
        calls.append(kwargs)
        return _FakeIterable(ROWS)

    monkeypatch.setattr("datasets.load_dataset", loader)
    return calls


def _load(size, *, skip, sampling="shuffled_buffer", tmp_path=None):
    return huggingface_pool(
        INSTR_SPEC, size, split="train", row_adapter=resolve_adapter(INSTR_SPEC), data_seed=1234,
        sampling=sampling, shuffle_buffer_size=TOTAL_ROWS, offline_snapshots=False,
        cache_dir=str(tmp_path) if tmp_path else None, skip=skip,
    )


# --------------------------------------------------------------------------------------
# (d) first: everything else depends on it
# --------------------------------------------------------------------------------------


def test_two_independent_streams_are_shuffled_into_identical_order(fake_loader):
    """FR-3.8 / D-016 condition 2: a per-call random seed would void the carving guarantee."""
    first, _ = _load(POOL_ROWS, skip=0)
    second, _ = _load(POOL_ROWS, skip=0)
    assert [e.example_id for e in first] == [e.example_id for e in second]
    assert derive_shuffle_seed(1234, "instr") == derive_shuffle_seed(1234, "instr")


def test_shuffled_buffer_actually_reorders_the_stream(fake_loader):
    """Otherwise `shuffled_buffer` would be indistinguishable from `head_of_stream` (EC-14)."""
    shuffled, _ = _load(POOL_ROWS, skip=0, sampling="shuffled_buffer")
    head, _ = _load(POOL_ROWS, skip=0, sampling="head_of_stream")
    assert [e.example_id for e in shuffled] != [e.example_id for e in head]


# --------------------------------------------------------------------------------------
# (a) offset-carved windows are disjoint
# --------------------------------------------------------------------------------------


def test_offset_carved_pools_share_no_example_id(fake_loader):
    pool, pool_report = _load(POOL_ROWS, skip=0)
    test, test_report = _load(TEST_ROWS, skip=pool_report.rows_consumed)
    assert pool_report.rows_consumed == POOL_ROWS       # passthrough: one example per row
    assert test_report.skip == POOL_ROWS
    assert not ({e.example_id for e in pool} & {e.example_id for e in test})
    splits = build_splits(
        "instr", training_pool=pool, test_pool=test, seed=1234,
        ratios=SplitRatios(train=0.70, validation=0.15, support=0.15),
    )
    splits.assert_disjoint()                            # must not raise


# --------------------------------------------------------------------------------------
# (b) the pre-fix behaviour is caught, loudly
# --------------------------------------------------------------------------------------


def test_zero_offset_reproduces_the_leak_and_build_splits_raises(fake_loader):
    """The pre-fix path: both loads stream from row 0, so TEST is a prefix of the training pool."""
    pool, _ = _load(POOL_ROWS, skip=0)
    test, _ = _load(TEST_ROWS, skip=0)                  # test_offset = 0 == pre-fix behaviour
    overlap = {e.example_id for e in pool} & {e.example_id for e in test}
    assert len(overlap) == TEST_ROWS                    # the leak is real, not hypothetical
    with pytest.raises(LeakageError) as info:
        build_splits(
            "instr", training_pool=pool, test_pool=test, seed=1234,
            ratios=SplitRatios(train=0.70, validation=0.15, support=0.15),
        )
    assert "test" in str(info.value)


# --------------------------------------------------------------------------------------
# (c) prepare_data enforces the offset against the MEASURED row count
# --------------------------------------------------------------------------------------


def _single_split_config(raw_config, *, test_offset):
    """A dummy-tier config whose only task is the fixture single-split corpus."""
    tree = dict(raw_config)
    tree["datasets"] = {
        "order": ["instr"],
        "tasks": {
            "instr": {
                "task_name": "Instruction Following", "task_type": "instruction_following",
                "primary_score": "rougeL", "metrics": ["rougeL"],
                "input_fields": ["instruction", "context"], "target_field": "response",
                "prompt_template": "{instruction}\n\n{context}\n\nResponse:",
                "max_input_tokens": 64, "max_target_tokens": 32,
                "hf_id": "fixture/dolly", "hf_revision": "c" * 40, "hf_config": "default",
                "upstream_test_source": "offset-carved window of upstream 'train'",
                "row_adapter": "passthrough",
                "row_adapter_options": {"list_target_pick": "first"},
            }
        },
    }
    cfg = resolve(tree, run_mode="dummy", experiment_id="carving-check")
    # Set the data section *after* resolution: the dummy overlay in configs/experiments/base.yaml
    # deliberately forces provider: synthetic (NG3), and this fixture needs the huggingface path
    # with a monkeypatched loader.
    cfg.raw["data"] = {
        **cfg.raw["data"],
        "provider": "huggingface",
        "sampling": "shuffled_buffer",
        "shuffle_buffer_size": TOTAL_ROWS,
        "offline_snapshots": False,
        "smoke_max_examples_per_split": 4,
        "volumes": {
            "instr": {
                "train_pool": POOL_ROWS, "test": TEST_ROWS,
                "test_split": "train", "test_offset": test_offset,
            }
        },
    }
    return cfg


def test_prepare_data_accepts_an_offset_at_or_above_the_measured_row_count(raw_config, fake_loader):
    cfg = _single_split_config(raw_config, test_offset=POOL_ROWS)
    data = prepare_data(cfg)
    assert data.blocked == {}
    assert data.leakage_report["clean"] is True
    counts = data.splits["instr"].counts()
    assert all(count > 0 for count in counts.values())


def test_prepare_data_rejects_an_offset_below_the_measured_row_count(raw_config, fake_loader):
    """FR-4.4: checked against measured rows, not the configured example cap."""
    cfg = _single_split_config(raw_config, test_offset=POOL_ROWS - 1)
    with pytest.raises(DataError, match="test_offset"):
        prepare_data(cfg)


def test_prepare_data_requires_a_test_offset_when_test_split_is_the_training_split(raw_config, fake_loader):
    """FR-4.2: omitting it is exactly the pre-fix leak."""
    from amaun.errors import ConfigError

    cfg = _single_split_config(raw_config, test_offset=POOL_ROWS)
    del cfg.raw["data"]["volumes"]["instr"]["test_offset"]
    with pytest.raises(ConfigError, match="test_offset"):
        prepare_data(cfg)

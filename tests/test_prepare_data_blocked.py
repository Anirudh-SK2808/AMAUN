"""FR-7 -- blocked != failed (AC-16).  Offline, dummy tier, loader monkeypatched.

Two claims, and the second is the reason this file exists:

1. A task blocked by an *open decision* (`unpinned_dataset` D-006, `unresolved_row_adapter`
   D-015) is reason-coded and reported, and is detected **before any loader call** so a block
   costs no network.
2. Any *other* failure -- here a loader ``RuntimeError`` -- propagates. Silent degradation on a
   transient error would be indistinguishable from a real block, so the skip path is deliberately
   narrow.

The shipped curriculum configures all six tasks, so the blocked path is exercised with
**fixture** specs rather than by leaving a real task unconfigured.
"""

from __future__ import annotations

import pytest

from amaun.config import resolve
from amaun.errors import PreconditionError
from amaun.experiment import BLOCK_REASONS, prepare_data

# Distinct rows per upstream split: a fake that served the same rows for `train` and
# `validation` would (correctly) trip the leakage guard and test nothing about blocking.
SPLIT_ROWS = {
    "train": [{"article": f"train article {i}", "highlights": f"train summary {i}"}
              for i in range(40)],
    "validation": [{"article": f"val article {i}", "highlights": f"val summary {i}"}
                   for i in range(20)],
}


class _FakeIterable:
    def __init__(self, items):
        self._items = list(items)

    def shuffle(self, seed, buffer_size):
        return _FakeIterable(self._items)

    def skip(self, n):
        return _FakeIterable(self._items[n:])

    def __iter__(self):
        return iter(self._items)


def _task(**overrides):
    base = {
        "task_name": "T", "task_type": "summarization", "primary_score": "rougeL",
        "metrics": ["rougeL"], "input_fields": ["article"], "target_field": "highlights",
        "prompt_template": "{article}", "max_input_tokens": 32, "max_target_tokens": 8,
        "hf_id": "fixture/dataset", "hf_revision": "e" * 40, "hf_config": "default",
        "upstream_test_source": "fixture validation split",
        "row_adapter": "passthrough", "row_adapter_options": {"list_target_pick": "first"},
    }
    base.update(overrides)
    return base


def _cfg(raw_config, monkeypatch, *, loader):
    monkeypatch.setattr("datasets.load_dataset", loader)
    tree = dict(raw_config)
    tree["datasets"] = {
        "order": ["available", "no_adapter", "no_dataset"],
        "tasks": {
            "available": _task(),
            # D-015: the rule is blanked, exactly as an unconfigured new task would be.
            "no_adapter": _task(row_adapter=None),
            # D-006: no pinned dataset identity.
            "no_dataset": _task(hf_id=None, pinned=False,
                                unpinned_reason="fixture: decision not yet resolved"),
        },
    }
    cfg = resolve(tree, run_mode="dummy", experiment_id="blocked-check")
    cfg.raw["data"] = {
        **cfg.raw["data"],
        "provider": "huggingface",            # set post-resolve: the dummy overlay forces synthetic
        "sampling": "shuffled_buffer",
        "shuffle_buffer_size": 64,
        "offline_snapshots": False,
        "smoke_max_examples_per_split": 4,
        "volumes": {
            task: {"train_pool": 24, "test": 8, "test_split": "validation"}
            for task in ("available", "no_adapter", "no_dataset")
        },
    }
    return cfg


def test_blocked_tasks_are_reason_coded_and_the_loader_is_never_called_for_them(raw_config, monkeypatch):
    calls: list[str] = []

    def loader(hf_id, hf_config=None, **kwargs):
        calls.append(f"{hf_id}:{kwargs.get('split')}")
        return _FakeIterable(SPLIT_ROWS[kwargs["split"]])

    data = prepare_data(_cfg(raw_config, monkeypatch, loader=loader))

    assert data.blocked == {
        "no_adapter": "unresolved_row_adapter",
        "no_dataset": "unpinned_dataset",
    }
    assert set(data.blocked.values()) == {"unresolved_row_adapter", "unpinned_dataset"}
    assert data.unpinned == ["no_dataset"]                  # retained subset, FR-7.1
    assert [task.task_id for task in data.tasks] == ["available"]
    # Exactly two loads, both for the available task: a blocked task costs no network.
    assert calls == ["fixture/dataset:train", "fixture/dataset:validation"]
    assert data.leakage_report["clean"] is True


def test_every_reason_code_names_its_decision(raw_config, monkeypatch):
    data = prepare_data(_cfg(raw_config, monkeypatch,
                             loader=lambda *a, **k: _FakeIterable(SPLIT_ROWS[k["split"]])))
    for code in data.blocked.values():
        assert code in BLOCK_REASONS
        assert BLOCK_REASONS[code]["decision"] in {"D-006", "D-015"}
        assert BLOCK_REASONS[code]["reason"]


def test_a_loader_error_for_an_available_task_propagates(raw_config, monkeypatch):
    """FR-7.2: a transient failure must never masquerade as an open decision."""
    def loader(*args, **kwargs):
        raise RuntimeError("simulated hub outage")

    with pytest.raises(RuntimeError, match="simulated hub outage"):
        prepare_data(_cfg(raw_config, monkeypatch, loader=loader))


def test_zero_available_tasks_raises_citing_both_decisions(raw_config, monkeypatch):
    """FR-7.5."""
    def loader(*args, **kwargs):
        raise AssertionError("no loader call is expected: every task is blocked")

    cfg = _cfg(raw_config, monkeypatch, loader=loader)
    cfg.raw["datasets"]["order"] = ["no_adapter", "no_dataset"]
    with pytest.raises(PreconditionError) as info:
        prepare_data(cfg)
    assert "D-006" in str(info.value)
    assert "D-015" in str(info.value)

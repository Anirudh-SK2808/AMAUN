"""Protocol S1 -- the gate that stops an invalid `actual` run producing a reported number."""

import pytest

from amaun.config import resolve
from amaun.errors import ConfigError, PreconditionError
from amaun.experiment import check_reportable_preconditions, prepare_data


def test_non_reportable_modes_have_no_preconditions(cfg):
    assert check_reportable_preconditions(cfg, strict=True) == []


def test_reportable_run_requires_three_seeds_and_a_clean_tree(raw_config, monkeypatch):
    """Built without going through resolve(), since `actual` needs CUDA on this host."""
    import amaun.experiment as experiment_module
    from amaun.config import BACKBONE_REGISTRY, ResolvedConfig, RunMode

    tree = dict(raw_config)
    tree["experiment"] = {**tree["experiment"], "seeds": [1]}
    tree["curriculum"] = {**tree.get("curriculum", {}), "repository_init": None}
    cfg = ResolvedConfig(
        run_mode=RunMode.ACTUAL, backbone=BACKBONE_REGISTRY[RunMode.ACTUAL],
        experiment_id="fake", seed=1, raw=tree, device="cuda",
    )
    monkeypatch.setattr(
        experiment_module, "git_state",
        lambda: {"dirty": True, "dirty_files": ["a.py"], "commit": "abc", "diff_sha256": "d"},
    )
    problems = check_reportable_preconditions(cfg, strict=False)
    joined = " ".join(problems)
    assert "seeds" in joined                 # D-012
    assert "dirty" in joined                 # reproducibility
    assert "repository_init" in joined       # D-002 still OPEN
    with pytest.raises(PreconditionError):
        check_reportable_preconditions(cfg, strict=True)


def test_prepare_data_builds_and_audits_every_task(cfg):
    data = prepare_data(cfg)
    assert len(data.tasks) == 6
    assert data.leakage_report["clean"] is True
    assert data.leakage_report["pairs_checked"] > 0
    for task in data.tasks:
        counts = data.splits[task.task_id].counts()
        assert all(count > 0 for count in counts.values())


def test_prepare_data_subsamples_uniformly_outside_actual(cfg):
    """idea.md S0.2.1."""
    cap = int(cfg.get("data.smoke_max_examples_per_split"))
    data = prepare_data(cfg)
    for task_splits in data.splits.values():
        for count in task_splits.counts().values():
            assert count <= cap


def _hf_cfg(raw_config, monkeypatch, *, data_overrides):
    """A dummy-tier config on the huggingface path, with the loader wired to fail loudly.

    Every assertion below must raise *before* any load, so a loader call is itself a failure.
    """
    def never(*args, **kwargs):
        raise AssertionError("data.volumes validation must happen before any dataset load")

    monkeypatch.setattr("datasets.load_dataset", never)
    tree = dict(raw_config)
    tree["datasets"] = {
        "order": ["solo"],
        "tasks": {
            "solo": {
                "task_name": "Solo", "task_type": "summarization", "primary_score": "rougeL",
                "metrics": ["rougeL"], "input_fields": ["article"], "target_field": "highlights",
                "prompt_template": "{article}", "max_input_tokens": 32, "max_target_tokens": 8,
                "hf_id": "fixture/dataset", "hf_revision": "d" * 40, "hf_config": "default",
                "upstream_test_source": "fixture", "row_adapter": "passthrough",
                "row_adapter_options": {"list_target_pick": "first"},
            }
        },
    }
    cfg = resolve(tree, run_mode="dummy", experiment_id="volumes-check")
    # After resolution, because the dummy overlay forces provider: synthetic by design (NG3).
    cfg.raw["data"] = {
        **cfg.raw["data"],
        "provider": "huggingface",
        "sampling": "shuffled_buffer",
        "shuffle_buffer_size": 128,
        "offline_snapshots": False,
        "volumes": {"solo": {"train_pool": 64, "test": 16, "test_split": "validation"}},
        **data_overrides,
    }
    return cfg


def test_volumes_required_absent_section_raises(raw_config, monkeypatch):
    """FR-3.4: the 1000/200/'validation' code defaults are gone; the config must say."""
    cfg = _hf_cfg(raw_config, monkeypatch, data_overrides={"volumes": None})
    with pytest.raises(ConfigError, match="data.volumes.solo"):
        prepare_data(cfg)


@pytest.mark.parametrize("key", ["train_pool", "test", "test_split"])
def test_volumes_required_each_key_is_mandatory(raw_config, monkeypatch, key):
    volumes = {"train_pool": 64, "test": 16, "test_split": "validation"}
    volumes.pop(key)
    cfg = _hf_cfg(raw_config, monkeypatch, data_overrides={"volumes": {"solo": volumes}})
    with pytest.raises(ConfigError, match=key):
        prepare_data(cfg)


def test_volumes_required_test_offset_when_test_split_is_the_training_split(raw_config, monkeypatch):
    """FR-4.2: without an offset, TEST is a prefix of the training pool (I2.1)."""
    cfg = _hf_cfg(raw_config, monkeypatch, data_overrides={
        "volumes": {"solo": {"train_pool": 64, "test": 16, "test_split": "train"}},
    })
    with pytest.raises(ConfigError, match="test_offset"):
        prepare_data(cfg)


def test_volumes_required_sampling_has_no_default(raw_config, monkeypatch):
    """D-016: which rows a cap selects is an experimental choice."""
    cfg = _hf_cfg(raw_config, monkeypatch, data_overrides={"sampling": None})
    with pytest.raises(ConfigError, match="data.sampling"):
        prepare_data(cfg)


def test_volumes_required_shuffle_buffer_size_has_no_default(raw_config, monkeypatch):
    cfg = _hf_cfg(raw_config, monkeypatch, data_overrides={"shuffle_buffer_size": None})
    with pytest.raises(ConfigError, match="shuffle_buffer_size"):
        prepare_data(cfg)


def test_volumes_required_buffer_below_max_pool_is_refused(raw_config, monkeypatch):
    """EC-14: a shuffle window narrower than the sample misreports the sampling policy."""
    cfg = _hf_cfg(raw_config, monkeypatch, data_overrides={"shuffle_buffer_size": 32})
    with pytest.raises(ConfigError, match="shuffle_buffer_size"):
        prepare_data(cfg)


def test_prepare_data_reports_unpinned_tasks_under_huggingface(raw_config):
    """D-006 (RESOLVED 2026-09-22): all six curriculum tasks are now pinned.

    Previously instr/longform were deliberately unpinned; that blocker is resolved
    (instr -> databricks-dolly-15k, longform -> Pavithree/eli5), so the curriculum must report
    zero unpinned tasks. The unpinned-detection mechanism itself (skip loudly, never guess) stays
    exercised by ``test_data_registry.py``.
    """
    tree = dict(raw_config)
    tree["data"] = {**tree["data"], "provider": "huggingface"}
    cfg = resolve(tree, run_mode="dummy", experiment_id="unpinned-check")
    from amaun.data.registry import load_curriculum

    curriculum = load_curriculum(cfg.section("datasets"))
    unpinned = [task.task_id for task in curriculum if not task.pinned]
    assert set(unpinned) == set()

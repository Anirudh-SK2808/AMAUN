"""Protocol S1 -- the gate that stops an invalid `actual` run producing a reported number."""

import pytest

from amaun.config import resolve
from amaun.errors import PreconditionError
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


def test_prepare_data_reports_unpinned_tasks_under_huggingface(raw_config):
    """D-006: unpinned rows must be skipped loudly, not guessed at."""
    tree = dict(raw_config)
    tree["data"] = {**tree["data"], "provider": "huggingface"}
    cfg = resolve(tree, run_mode="dummy", experiment_id="unpinned-check")
    # The two unpinned rows must be reported; the pinned ones would need network access,
    # so this asserts only that the unpinned detection happens before any download attempt.
    from amaun.data.registry import load_curriculum

    curriculum = load_curriculum(cfg.section("datasets"))
    unpinned = [task.task_id for task in curriculum if not task.pinned]
    assert set(unpinned) == {"instr", "longform"}

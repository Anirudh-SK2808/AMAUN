"""M14 -- artefact namespacing and provenance (criteria I2, I3)."""

import json
from pathlib import Path

import pytest

from amaun.config import RunMode, resolve
from amaun.errors import RunModeArtifactError
from amaun.utils.paths import assert_write_allowed, run_paths, smoke_label
from amaun.utils.provenance import collect_provenance, git_state, package_versions
from amaun.utils.tracking import ExperimentTracker


def test_paths_are_namespaced_by_run_mode(cfg):
    paths = run_paths(cfg, base="results")
    assert paths.root == Path("results/dummy/pytest")
    # Adapters are namespaced too, so checkpoints cannot cross tiers (idea.md S16.12).
    assert paths.adapters == Path("adapters/dummy/pytest")
    assert paths.candidates.is_relative_to(paths.adapters)


def test_cross_run_mode_writes_are_refused():
    """idea.md S0.1.1: smoke and actual artefacts must never mix."""
    assert_write_allowed(Path("results/dummy/x/metrics.jsonl"), RunMode.DUMMY)
    for forbidden in ("results/actual/x/metrics.jsonl", "results/local_smoke/x/m.jsonl"):
        with pytest.raises(RunModeArtifactError):
            assert_write_allowed(Path(forbidden), RunMode.DUMMY)


def test_unknown_top_level_results_directory_is_refused():
    with pytest.raises(RunModeArtifactError):
        assert_write_allowed(Path("results/scratch/x.json"), RunMode.DUMMY)


def test_smoke_label_is_present_for_non_reportable_modes():
    assert smoke_label(RunMode.ACTUAL) == ""
    for mode in (RunMode.DUMMY, RunMode.LOCAL_SMOKE):
        assert "not a reported result" in smoke_label(mode)


def test_tracker_writes_config_and_provenance(cfg, tmp_path):
    tracker = ExperimentTracker(cfg, base=tmp_path).start()
    assert tracker.paths.config_used.exists()
    assert tracker.paths.provenance.exists()
    payload = json.loads(tracker.paths.config_used.read_text())
    assert payload["run_mode"] == "dummy"
    assert payload["config_hash"]


def test_every_record_carries_run_mode_and_seed(cfg, tmp_path):
    """Invariant I14.1."""
    tracker = ExperimentTracker(cfg, base=tmp_path).start()
    tracker.log_metric(task="sum", score=0.0)
    tracker.log_decision(outcome="CREATE")
    tracker.log_repository_event(event="register")
    for stream in ("metrics", "decisions", "repository_events"):
        records = list(tracker.read(stream))
        assert records
        for record in records:
            assert record["run_mode"] == "dummy"
            assert record["seed"] == cfg.seed
            assert record["schema_version"] == 1
            assert record["config_hash"] == cfg.config_hash


def test_records_are_append_only_and_ordered(cfg, tmp_path):
    tracker = ExperimentTracker(cfg, base=tmp_path).start()
    for index in range(5):
        tracker.log_metric(index=index)
    records = list(tracker.read("metrics"))
    assert [r["seq"] for r in records] == [1, 2, 3, 4, 5]
    assert [r["index"] for r in records] == [0, 1, 2, 3, 4]


def test_tracker_refuses_cross_mode_directory(raw_config, tmp_path):
    """A dummy run must not be able to write into the actual namespace."""
    cfg = resolve(dict(raw_config), run_mode="dummy", experiment_id="x")
    tracker = ExperimentTracker(cfg, base=tmp_path)
    forbidden = tmp_path / "actual" / "x" / "sneaky.json"
    with pytest.raises(RunModeArtifactError):
        assert_write_allowed(forbidden, cfg.run_mode, base=tmp_path)


def test_provenance_captures_what_reproduction_needs():
    """Brief S18."""
    provenance = collect_provenance()
    assert provenance["git"]["commit"] or provenance["git"]["commit"] is None
    assert "dirty" in provenance["git"]
    versions = package_versions()
    for package in ("torch", "transformers", "peft"):
        assert package in versions
    hardware = provenance["hardware"]
    for key in ("platform", "machine", "python", "hostname"):
        assert hardware[key]


def test_dirty_tree_records_a_diff_hash():
    """A boolean 'dirty' cannot prove two dirty runs had the same state; a diff hash can."""
    state = git_state()
    if state["dirty"]:
        assert state["diff_sha256"]
        assert state["dirty_files"]

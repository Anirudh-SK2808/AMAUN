"""The L4 decision study, end to end on the dummy tier (wiring, leakage order, memory audit).

One module-scoped run backs every assertion: it is the expensive part, and each test checks a
different property of the same artifacts.
"""

from __future__ import annotations

import json

import torch

import pytest

from amaun.config import load_config, resolve
from amaun.continual.decision_study import DecisionStudy
from amaun.data.splits import SplitAccessGuard, SplitName
from amaun.experiment import prepare_data
from amaun.models.backbone import BackboneCache, load_backbone
from amaun.utils.tracking import ExperimentTracker

from conftest import ROOT

CONFIG = ROOT / "configs" / "experiments" / "l4_decision_study.yaml"


@pytest.fixture(scope="module")
def study_run(tmp_path_factory):
    raw = load_config(CONFIG, config_root=ROOT / "configs")
    cfg = resolve(raw, run_mode="dummy", experiment_id="pytest-decision-study")
    tracker = ExperimentTracker(cfg, base=tmp_path_factory.mktemp("results")).start()
    data = prepare_data(cfg)
    BackboneCache.release()
    backbone = load_backbone(cfg)

    reads: list[tuple[str, str, str]] = []          # (event, task, split)
    original_get = SplitAccessGuard.get

    def recording_get(self, name):
        reads.append(("read", self.task_id, SplitName(name).value))
        return original_get(self, name)

    study = DecisionStudy(cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits, tracker=tracker)
    original_log = tracker.log_decision

    def recording_log(**record):
        reads.append(("decision", record["task_id"], record["strategy"]))
        return original_log(**record)

    original_supervision = study.utility_supervision

    def marked_supervision():
        reads.append(("supervision_start", "", ""))
        out = original_supervision()
        reads.append(("supervision_end", "", ""))
        return out

    # The second gate decision (FinQA) is forced open to exercise the skip-training path; the
    # first (ELI5) is the fitted gate's own decision.  Forcing changes only `fired`, not evidence.
    from amaun.manager.reuse_gate import GateDecision, ReuseGate

    original_decide = ReuseGate.decide
    gate_calls: list[str] = []

    def decide(self, features, *, run_mode):
        out = original_decide(self, features, run_mode=run_mode)
        gate_calls.append("call")
        if len(gate_calls) == 2:
            best = max(out.predicted_gaps, key=out.predicted_gaps.get)
            return GateDecision(True, best, out.predicted_gaps, out.eligible, "forced open by test")
        return out

    tracker.log_decision = recording_log
    study.utility_supervision = marked_supervision
    SplitAccessGuard.get = recording_get
    ReuseGate.decide = decide
    try:
        summary = study.run()
    finally:
        SplitAccessGuard.get = original_get
        ReuseGate.decide = original_decide
        BackboneCache.release()
    return {"summary": summary, "root": tracker.paths.root, "reads": reads, "study": study, "data": data}


def _load(run, name):
    return json.loads((run["root"] / "decision_study" / name).read_text())


def test_every_phase_completes_without_failures_or_skips(study_run):
    summary = study_run["summary"]
    assert summary["failures"] == [] and summary["skipped"] == []
    assert summary["preflight"]["ok"] is True
    assert summary["backbone_fingerprint_unchanged"] is True
    for name in ("assignments.json", "initial_adapters.json", "utility_supervision.json",
                 "evaluation_longform.json", "evaluation_domain.json", "memory_audit.json", "summary.json"):
        assert (study_run["root"] / "decision_study" / name).exists(), name


def test_assignments_are_recorded_and_no_adapters_share_training_rows(study_run):
    adapters = _load(study_run, "initial_adapters.json")
    assert [(a["adapter_id"], a["task_id"]) for a in adapters] == [
        ("seed-sum", "sum"), ("seed-qa", "qa"), ("seed-sentiment", "sentiment")]
    assert all(a["shares_training_rows_with"] == {} for a in adapters)
    assert all(a["checkpoint"] for a in adapters)


def test_utility_supervision_is_measured_not_proxied_and_limitations_are_stated(study_run):
    report = _load(study_run, "utility_supervision.json")
    assert report["supervision"]["n_examples"] == 6          # every ordered pair of 3 adapters
    assert report["supervision"]["n_proxy"] == 0
    for item in report["measurements"]:
        assert item["fusion"]["kd"]["enabled"] is True        # retention data was found
        assert "semantic.d_task_sq" not in item["missing_features"]
        assert set(item["validation_loss"]) == {"new_standalone", "new_fused", "old_standalone", "old_fused"}
    assert any("too few" in note for note in report["limitations"])


def test_no_test_split_is_read_before_fitting_or_before_a_tasks_decisions(study_run):
    reads = study_run["reads"]
    start = reads.index(("supervision_start", "", ""))
    end = reads.index(("supervision_end", "", ""))
    assert not [r for r in reads[start:end] if r[0] == "read" and r[2] == "test"]
    for task_id in ("longform", "domain"):
        decisions = [i for i, r in enumerate(reads) if r[0] == "decision" and r[1] == task_id]
        test_reads = [i for i, r in enumerate(reads[end:], start=end)
                      if r[0] == "read" and r[1] == task_id and r[2] == "test"]
        assert decisions and test_reads
        assert max(decisions) < min(test_reads), f"{task_id}: TEST read before every decision was logged"


def test_every_policy_decides_and_every_option_is_measured(study_run):
    for task_id in ("longform", "domain"):
        evaluation = _load(study_run, f"evaluation_{task_id}.json")
        assert set(evaluation["decisions"]) == {"amaun", "amaun_train_first", "always_merge",
                                                "always_separate", "fixed_rule"}
        assert evaluation["decisions"]["always_separate"]["option"] == "CREATE"
        assert evaluation["decisions"]["always_merge"]["option"].startswith("FUSE:")
        options = evaluation["policy_outcomes"]["options"]
        assert len(options) == 7 and all(o["measured"] for o in options.values())
        assert options["CREATE"]["adapters_retained"] == 4
        assert all(o["adapters_retained"] == 3 for k, o in options.items() if k != "CREATE")
        for row in evaluation["actual_vs_predicted"].values():
            assert row["actual_measured_on_test"]["utility"] is not None


def test_repository_is_not_mutated_by_counterfactual_evaluation(study_run):
    study = study_run["study"]
    assert sorted(study.repository.ids()) == ["seed-qa", "seed-sentiment", "seed-sum"]


def test_memory_audit_reports_stable_live_tensors_across_cycles(study_run):
    audit = _load(study_run, "memory_audit.json")
    assert audit["n_cycles"] == 3
    assert audit["verdict"] == "stable", audit
    assert audit["live_tensor_growth"] is not None and audit["live_tensor_growth"] <= audit["tensor_tolerance"]


def test_per_task_split_caps_override_the_global_caps(study_run):
    counts = {task_id: splits.counts() for task_id, splits in study_run["data"].splits.items()}
    assert counts["domain"]["validation"] == 5 and counts["domain"]["support"] == 5   # task override
    assert all(counts[t]["validation"] == 6 for t in ("sum", "qa", "sentiment", "longform"))


def test_preflight_oom_is_reported_and_aborts_without_changing_settings(tmp_path, monkeypatch):
    """Simulated CUDA OOM on the first check: exact error + smallest adjustment recorded, the run
    stops before training, and training.batch_size is untouched."""
    import amaun.continual.decision_study as module

    raw = load_config(CONFIG, config_root=ROOT / "configs")
    cfg = resolve(raw, run_mode="dummy", experiment_id="pytest-preflight-oom")
    tracker = ExperimentTracker(cfg, base=tmp_path).start()
    data = prepare_data(cfg)
    BackboneCache.release()
    backbone = load_backbone(cfg)
    try:
        study = DecisionStudy(cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits, tracker=tracker)

        def oom(*args, **kwargs):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (simulated)")

        monkeypatch.setattr(module, "masked_cross_entropy", oom)
        summary = study.run()
    finally:
        BackboneCache.release()

    assert "preflight OOM" in summary["aborted"]
    assert "initial_adapters" not in summary                     # nothing was trained
    report = json.loads((tracker.paths.root / "decision_study" / "preflight.json").read_text())
    failed = [c for c in report["checks"] if not c["ok"]]
    assert failed and "Tried to allocate 2.00 GiB" in failed[0]["error"]
    assert "training.batch_size: 1" in failed[0]["smallest_adjustment"]
    assert cfg.get("training.batch_size") == 2                   # dummy value, unchanged
    # Every task was still checked after the first failure, so one OOM does not hide another.
    assert {c["task_id"] for c in report["checks"]} == {"sum", "qa", "sentiment", "longform", "domain"}


def test_reuse_gate_is_fitted_on_measured_supervision_pairs(study_run):
    gate = _load(study_run, "utility_supervision.json")["reuse_gate"]
    assert gate["trained"] and gate["n_train"] == 6
    assert len(gate["in_sample"]) == 6
    for item in _load(study_run, "utility_supervision.json")["measurements"]:
        features = item["reuse_gate_example"]["features"]
        assert all(features[k] is not None for k in ("cand_loss", "base_loss", "loss_gain", "d_task_sq"))


def test_gate_decides_before_training_and_its_reuse_is_the_amaun_decision(study_run):
    reads = study_run["reads"]
    phase3 = reads.index(("supervision_end", "", ""))
    # FinQA: gate forced open.  Its REUSE must be logged before ANY non-SUPPORT read of FinQA.
    amaun_at = next(i for i, r in enumerate(reads) if r == ("decision", "domain", "amaun"))
    before = [r for r in reads[phase3:amaun_at] if r[0] == "read" and r[1] == "domain"]
    assert before and all(r[2] == "support" for r in before), before
    evaluation = _load(study_run, "evaluation_domain.json")
    gate = evaluation["reuse_gate"]
    assert gate["decision"]["fired"] is True
    assert evaluation["decisions"]["amaun"]["option"] == f"REUSE:{gate['decision']['adapter_id']}"
    assert gate["training_seconds_saved_if_deployed"] > 0 and gate["optimizer_steps_saved_if_deployed"] > 0
    assert all(row.get("actual_gap_on_test") is not None for row in gate["predicted_vs_actual"].values())
    # The train-first policy still decided after training, on the same measured options.
    assert evaluation["policy_outcomes"]["amaun_train_first"]["measured"] is True


def test_a_closed_gate_leaves_the_train_first_decision_without_reuse(study_run):
    evaluation = _load(study_run, "evaluation_longform.json")
    gate = evaluation["reuse_gate"]["decision"]
    if gate["fired"]:
        assert evaluation["decisions"]["amaun"]["option"].startswith("REUSE:")
    else:
        assert not evaluation["decisions"]["amaun"]["option"].startswith("REUSE:")
        assert evaluation["reuse_gate"]["training_seconds_saved_if_deployed"] == 0.0

"""Stage-1 REUSE gate (D-003 O1): decides REUSE before any temporary adapter is trained."""

from __future__ import annotations

import random

import pytest

from amaun.errors import ManagerError
from amaun.manager.reuse_gate import (
    GATE_FEATURES,
    GateExample,
    ReuseGate,
    ReuseGateConfig,
    reuse_gap,
)

from test_continual_runner import _amaun_strategy, _runner

CFG = ReuseGateConfig(enabled=True, tolerance=0.01, ridge_lambda=0.1, probe_examples=4,
                      require_gain_over_backbone=True, min_train_examples=2)


def _features(rel_gain: float, d_task: float, rng: random.Random) -> dict[str, float]:
    base = 3.0
    return {
        "d_task_sq": d_task, "cand_loss": base * (1 - rel_gain), "base_loss": base,
        "loss_gain": base * rel_gain, "rel_loss_gain": rel_gain,
        "cand_confidence": 0.5 + rng.uniform(-0.05, 0.05), "cand_entropy": 2.0 + rng.uniform(-0.1, 0.1),
        "cand_ece": 0.1, "cand_token_accuracy": 0.3,
    }


def _examples(n=24, seed=0):
    """Ground truth: reuse is acceptable when the candidate already captures most of the gain."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        rel_gain = rng.uniform(-0.1, 0.6)
        d_task = rng.uniform(0.0, 2.0)
        gap = 0.2 * (rel_gain - 0.4) - 0.02 * d_task          # >= 0 roughly when rel_gain > 0.45
        out.append(GateExample("t", f"c{i}", _features(rel_gain, d_task, rng), gap))
    return out


def test_gate_learns_the_measured_gap_from_several_features():
    gate = ReuseGate.fit(_examples(), CFG, run_mode="dummy")
    assert gate.trained
    rng = random.Random(9)
    near = gate.predict(_features(0.58, 0.1, rng))     # candidate captures nearly all the gain
    far = gate.predict(_features(0.05, 1.9, rng))      # barely helps and is distant
    assert near > far
    assert sum(abs(w) > 1e-6 for w in gate.weights) >= 2   # a multi-feature rule (idea.md S16.2)


def test_gate_fires_only_for_an_eligible_candidate_within_tolerance():
    gate = ReuseGate.fit(_examples(), CFG, run_mode="dummy")
    rng = random.Random(3)
    good, poor = _features(0.58, 0.1, rng), _features(0.05, 1.9, rng)
    decision = gate.decide({"good": good, "poor": poor}, run_mode="dummy")
    assert decision.fired and decision.adapter_id == "good"
    assert not gate.decide({"poor": poor}, run_mode="dummy").fired
    # A candidate that does not beat the frozen backbone on SUPPORT is never reused.
    worse_than_backbone = {**good, "loss_gain": -0.1, "rel_loss_gain": -0.03}
    closed = gate.decide({"x": worse_than_backbone}, run_mode="dummy")
    assert not closed.fired and closed.eligible == {"x": False}


def test_gate_is_disabled_below_min_examples_and_when_turned_off():
    tiny = ReuseGate.fit(_examples(n=1), CFG, run_mode="dummy")
    assert not tiny.trained and "DISABLED" in tiny.notes[0]
    rng = random.Random(1)
    assert not tiny.decide({"c": _features(0.6, 0.0, rng)}, run_mode="dummy").fired
    off = ReuseGate.fit(_examples(), ReuseGateConfig(**{**CFG.__dict__, "enabled": False}), run_mode="dummy")
    assert not off.decide({"c": _features(0.6, 0.0, rng)}, run_mode="dummy").fired


def test_gate_refuses_another_run_mode_and_round_trips(tmp_path):
    gate = ReuseGate.fit(_examples(), CFG, run_mode="dummy")
    with pytest.raises(ManagerError, match="run_mode"):
        gate.decide({}, run_mode="local_smoke")
    path = gate.save(tmp_path / "gate.json")
    loaded = ReuseGate.load(path, expected_run_mode="dummy")
    sample = _features(0.3, 0.5, random.Random(2))
    assert loaded.predict(sample) == pytest.approx(gate.predict(sample))
    with pytest.raises(ManagerError):
        ReuseGate.load(path, expected_run_mode="actual")


def test_missing_features_are_imputed_with_training_means_not_nan():
    gate = ReuseGate.fit(_examples(), CFG, run_mode="dummy")
    sample = {name: None for name in GATE_FEATURES}
    assert gate.predict(sample) == pytest.approx(gate.bias)   # every z is 0 -> intercept only


def test_reuse_gap_uses_the_supervision_proxy():
    assert reuse_gap(1.0, 1.0, scale=10.0) == 0.0
    assert reuse_gap(2.0, 1.0, scale=10.0) < 0 < reuse_gap(0.5, 1.0, scale=10.0)


# --------------------------------------------------------------------------------------
# Runner: the gate really runs BEFORE training
# --------------------------------------------------------------------------------------


def _forced_gate(*, fire: bool) -> ReuseGate:
    """A trained gate with zero weights: the prediction is the bias, so firing is deterministic."""
    cfg = ReuseGateConfig(**{**CFG.__dict__, "require_gain_over_backbone": False})
    dim = len(GATE_FEATURES)
    return ReuseGate(cfg, [0.0] * dim, [1.0] * dim, [0.0] * dim, 1.0 if fire else -1.0,
                     n_train=6, run_mode="dummy", trained=True)


def _gated_runner(cfg, backbone, tasks, splits, tmp_path, *, fire):
    strategy = _amaun_strategy(cfg)
    strategy.decision_order = "prescreen_then_train"
    strategy.reuse_gate = _forced_gate(fire=fire)
    runner, tracker = _runner(cfg, backbone, tasks, splits, tmp_path, strategy, f"gate-{fire}")
    runner.bootstrap_repository(["sum"])
    return runner, tracker


def test_a_firing_gate_reuses_without_training(cfg, backbone, tasks, splits, tmp_path):
    runner, tracker = _gated_runner(cfg, backbone, tasks, splits, tmp_path, fire=True)
    steps_before = runner.budget.optimizer_steps
    result = runner.run(["qa"])
    event = result.events[0]
    assert event.outcome == "REUSE" and event.training is None
    assert event.reuse_gate["fired"] is True
    assert runner.budget.optimizer_steps == steps_before          # nothing was trained
    assert event.decision["temp_adapter_trained"] is False
    assert event.decision["threshold_source"] == "reuse_gate"
    assert "qa" in runner.repository.retrieve("seed-sum").metadata.tasks_served
    # No temporary adapter was ever put in this run's candidate store, and none was registered.
    assert len(runner.repository) == 1 and runner.repository.candidates.ids() == []
    assert event.evaluations                                       # the reused adapter was evaluated
    decisions = list(tracker.read("decisions"))
    assert decisions[-1]["reuse_gate"]["fired"] is True


def test_a_closed_gate_trains_and_cannot_choose_reuse_afterwards(cfg, backbone, tasks, splits, tmp_path):
    runner, _ = _gated_runner(cfg, backbone, tasks, splits, tmp_path, fire=False)
    runner.strategy.thresholds.tau_reuse = -1e9                   # would force REUSE if allowed
    steps_before = runner.budget.optimizer_steps
    event = runner.run(["qa"]).events[0]
    assert runner.budget.optimizer_steps > steps_before            # A_temp was trained
    assert event.reuse_gate["fired"] is False
    assert event.outcome != "REUSE" and event.decision["temp_adapter_trained"] is True


def test_bootstrap_fits_a_gate_from_measured_dev_integrations(cfg, backbone, tasks, splits, tmp_path):
    from amaun.manager.bootstrap import bootstrap_utility_network

    runner, _ = _runner(cfg, backbone, tasks, splits, tmp_path, _amaun_strategy(cfg), "boot")
    runner.bootstrap_repository(["sum", "qa"])
    dev = [t for t in tasks if t.task_id == "instr"]
    result = bootstrap_utility_network(
        cfg=cfg, backbone=backbone, peft_model=runner.peft_model, repository=runner.repository,
        dev_tasks=dev, splits=splits, extractor=runner.extractor, all_tasks=tasks,
    )
    gate = result.reuse_gate
    assert gate is not None and gate.trained and gate.n_train == 2   # 1 dev task x 2 candidates
    assert result.as_dict()["reuse_gate"]["n_train"] == 2

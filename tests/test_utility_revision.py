"""D-029: revised utility models (constant features dropped, small network / ridge, nested CV) and the
checkpoint surgery that lets the running study redo Phase 4 and Phase 5's decisions."""

import copy
import importlib.util
import itertools
import json
import random
from pathlib import Path

import pytest
import torch

from amaun.config import load_config, resolve
from amaun.features.base import FeatureVector
from amaun.features.normalizer import FeatureNormalizer
from amaun.manager import utility_cv
from amaun.manager.targets import SupervisionExample, SupervisionSet
from amaun.manager.utility_cv import FitGate, fit_revised, revised_specs, select_hyper, select_revised
from amaun.manager.utility_network import OUTCOME_HEADS, RidgeUtility, load_utility_model

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["sum", "qa", "sentiment", "qa-B", "instr", "dialog"]
NAMES = ["semantic.d_task_sq", "performance.L_val_gap", "gradient.cos", "resource.RepoSize", "historical.Age_i"]
SPECS = {"nn": {"hidden_dims": [8], "max_epochs": 20, "batch_size": 4}, "ridge": {"lambdas": [0.1, 1.0, 10.0, 100.0]}}


def _vector(rng):
    # The last two features are constant over the supervision pairs, as on the L4 (no history yet).
    return FeatureVector(names=list(NAMES), values=[rng.random(), rng.random() - 0.5, rng.random(), 6.0, 0.0],
                         family_slices={"semantic": (0, 1), "performance": (1, 2), "gradient": (2, 3),
                                        "resource": (3, 4), "historical": (4, 5)})


def _supervision() -> SupervisionSet:
    rng = random.Random(0)
    supervision = SupervisionSet(fit_split="utility_train")
    for i, (new, old) in enumerate(itertools.permutations(TASKS, 2)):
        vector = _vector(rng)
        beneficial = vector.values[0] > 0.6
        supervision.examples.append(SupervisionExample(
            candidate_id=f"seed-{old}", task_id=new, features=vector, proxy=False, proxy_steps=None,
            beneficial=beneficial,
            targets={"primary_score": 0.9 + 0.2 * vector.values[0], "forgetting": 0.03 * vector.values[2],
                     "confidence": rng.random(), "memory": 73.9, "communication": 73.9,
                     "long_term_loss": 1.0 + rng.random()}))
    return supervision


@pytest.fixture(scope="module")
def full_cfg():
    raw = load_config(ROOT / "configs" / "experiments" / "l4_full_study.yaml", config_root=ROOT / "configs")
    return resolve(raw, run_mode="dummy", experiment_id="pytest-utility-revision")


# -- normaliser ---------------------------------------------------------------------------


def test_constant_features_are_dropped_and_a_later_change_stays_finite():
    rng = random.Random(1)
    vectors = [_vector(rng) for _ in range(10)]
    normalizer = FeatureNormalizer().fit(vectors, split="utility_train", run_mode="dummy", drop_constant=True)
    assert normalizer.active_names == NAMES[:3] and normalizer.n_active == 3
    later = FeatureVector(names=list(NAMES), values=[0.5, 0.0, 0.5, 11.0, 4.0],      # repository grew
                          family_slices=vectors[0].family_slices)
    row = normalizer.transform(later, run_mode="dummy")
    assert len(row) == 3 and all(abs(v) < 10 for v in row)
    legacy = FeatureNormalizer().fit(vectors, split="utility_train", run_mode="dummy")
    assert abs(legacy.transform(later, run_mode="dummy")[3]) > 1e8                  # the defect D-029 removes


def test_a_saved_normaliser_keeps_its_mask(tmp_path):
    rng = random.Random(2)
    normalizer = FeatureNormalizer().fit([_vector(rng) for _ in range(6)], split="utility_train", run_mode="dummy",
                                         drop_constant=True)
    loaded = FeatureNormalizer.load(normalizer.save(tmp_path / "n.json"), expected_run_mode="dummy")
    assert loaded.active == normalizer.active


# -- ridge ----------------------------------------------------------------------------------


def test_ridge_matches_the_closed_form_and_round_trips(tmp_path):
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(30, 4, generator=g), torch.randn(30, len(OUTCOME_HEADS), generator=g)
    model = RidgeUtility(4, OUTCOME_HEADS, lam=1.0).fit(x, y)
    xd = (x - x.mean(0)).double()
    yz = ((y - y.mean(0)) / y.std(0, unbiased=False)).double()
    expected = torch.linalg.solve(xd.T @ xd + torch.eye(4, dtype=torch.float64), xd.T @ yz)
    assert torch.allclose(model.weight.double(), expected, atol=1e-5)
    model.save(tmp_path / "r.pt", run_mode="dummy", feature_names=["a", "b", "c", "d"])
    loaded, names = load_utility_model(tmp_path / "r.pt", expected_run_mode="dummy")
    assert isinstance(loaded, RidgeUtility) and names == ["a", "b", "c", "d"]
    assert torch.allclose(loaded.utility(x), model.utility(x))


# -- nested CV and selection --------------------------------------------------------------------


def test_the_outer_held_out_task_never_reaches_the_inner_loop(full_cfg, monkeypatch):
    seen = []
    real = utility_cv.select_hyper

    def spy(cfg, examples, spec, *, fit_split):
        seen.append({e.task_id for e in examples})
        return real(cfg, examples, spec, fit_split=fit_split)

    monkeypatch.setattr(utility_cv, "select_hyper", spy)
    report = utility_cv.cv_model(full_cfg, _supervision(), revised_specs(SPECS)[1])
    outer = [f["held_out_task"] for f in report["folds"]]
    assert outer == sorted(TASKS)
    for held_out, tasks in zip(outer, seen):                   # one inner selection per outer fold ...
        assert held_out not in tasks and len(tasks) == len(TASKS) - 1
    assert seen[-1] == set(TASKS)                               # ... then the final one on every task


def test_network_epochs_are_capped_and_early_stopping_takes_the_lowest_loss(full_cfg):
    spec = revised_specs(SPECS)[0]
    assert spec.grid == tuple(range(1, 21))
    epochs, curve = select_hyper(full_cfg, _supervision().examples, spec, fit_split="utility_train")
    assert 1 <= epochs <= 20 and len(curve) == 20
    assert curve[str(epochs)] == min(curve.values())


def test_selection_rule():
    def report(name, kind, acc, mse):
        return {"name": name, "kind": kind, "oof_decision_accuracy": acc, "per_head_oof_mse": {"primary_score": mse}}

    assert select_revised([report("nn_small", "nn", 0.5, 3.0), report("ridge", "ridge", 0.33, 1.0)])["name"] == "nn_small"
    assert select_revised([report("nn_small", "nn", 0.5, 1.0), report("ridge", "ridge", 0.5, 2.0)])["name"] == "nn_small"
    assert select_revised([report("nn_small", "nn", 0.5, 1.0), report("ridge", "ridge", 0.5, 1.0)])["name"] == "ridge"


def test_fit_revised_selects_a_model_and_takes_tau_from_its_nested_out_of_fold(full_cfg):
    result = fit_revised(full_cfg, _supervision(), gate=FitGate(20, 0.0, 1.0), specs=revised_specs(SPECS))
    assert result.fitted and result.selected in ("nn_small", "ridge")
    assert result.normalizer.n_active == 3 and "nested out-of-fold" in result.thresholds.notes
    names = {c["name"] for c in result.as_dict()["candidates"]}
    assert names == {"nn_small", "ridge"}


# -- checkpoint surgery -----------------------------------------------------------------------


def _surgery():
    spec = importlib.util.spec_from_file_location("surgery_d029", ROOT / "scripts" / "checkpoint_surgery_d029.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path, cfg, *, old_raw):
    old = copy.deepcopy(cfg)
    old.raw = old_raw
    run_dir = tmp_path / "results" / "dummy" / cfg.experiment_id
    (run_dir / "decision_study" / "checkpoint").mkdir(parents=True)
    (run_dir / "utility_network").mkdir()
    (run_dir / "utility_network" / "utility_network.pt").write_text("v0")
    (run_dir / "decisions.jsonl").write_text("{}\n")
    (run_dir / "decision_study" / "evaluation_domain.json").write_text("{}")
    (run_dir / "config_used.json").write_text(json.dumps({
        "run_mode": "dummy", "experiment_id": cfg.experiment_id, "config_hash": old.config_hash, "config": old_raw}))
    units = {"phase:P3": {}, "P3:a->b": {}, "phase:P4": {}, "P5:train:domain": {}, "P5:domain:option:CREATE": {},
             "P5:domain:option:REUSE:seed-sum": {}, "P5:domain": {}, "P6:amaun_no_cap:state": {}}
    state = {"config_hash": old.config_hash, "phases": {p: "done" for p in ("P0", "P1", "P2", "P3", "P4")},
             "units": units, "caches": {"evaluations": {"k": 1}}}
    (run_dir / "decision_study" / "checkpoint" / "state.json").write_text(json.dumps(state))
    return run_dir


def _old_raw(cfg):
    raw = copy.deepcopy(cfg.raw)
    del raw["full_study"]["utility_models"]
    raw["full_study"]["policies"] = [p for p in raw["full_study"]["policies"] if p != "amaun_v0"]
    return raw


def test_surgery_keeps_measurements_and_redoes_decisions(tmp_path, full_cfg):
    surgery = _surgery()
    run_dir = _run(tmp_path, full_cfg, old_raw=_old_raw(full_cfg))
    plan = surgery.plan(run_dir, tmp_path / "adapters", full_cfg)
    assert plan["changed_keys"] and all(surgery.is_allowed(k) for k in plan["changed_keys"])
    assert plan["remove_units"] == ["P5:domain", "P6:amaun_no_cap:state", "phase:P4"]
    surgery.apply(plan)
    state = json.loads((run_dir / "decision_study" / "checkpoint" / "state.json").read_text())
    assert state["config_hash"] == full_cfg.config_hash and "P4" not in state["phases"]
    for kept in ("P3:a->b", "P5:train:domain", "P5:domain:option:CREATE", "P5:domain:option:REUSE:seed-sum"):
        assert kept in state["units"]
    assert state["caches"] == {"evaluations": {"k": 1}}
    assert state["surgeries"][0]["decision"] == "D-029"
    assert (run_dir / "utility_network.pre_d029" / "utility_network.pt").exists()
    assert (run_dir / "decisions.jsonl.pre_d029").exists() and not (run_dir / "decisions.jsonl").exists()
    assert (run_dir / "decision_study" / "checkpoint" / "state.json.pre_d029").exists()
    with pytest.raises(SystemExit, match="already applied"):
        surgery.plan(run_dir, tmp_path / "adapters", full_cfg)


def test_surgery_refuses_a_config_change_outside_the_allow_list(tmp_path, full_cfg):
    surgery = _surgery()
    old = _old_raw(full_cfg)
    old["training"] = {**old["training"], "batch_size": 99}            # something phases 0-3 read
    run_dir = _run(tmp_path, full_cfg, old_raw=old)
    with pytest.raises(SystemExit, match="outside"):
        surgery.plan(run_dir, tmp_path / "adapters", full_cfg)

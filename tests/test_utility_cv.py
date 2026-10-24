"""FR-12 - FR-15: measured targets, gate, leave-one-new-task-out CV, tau from out-of-fold, refit.

The fitted path only runs when the beneficial fraction is inside the gate; the dummy study's
supervision is one-class, so it is exercised here on hand-built pairs shaped like the L4 run's
Phase 3 (6 seed tasks, 30 ordered pairs, a third beneficial).
"""

import itertools
import random

import pytest

from amaun.features.base import FeatureVector
from amaun.manager.targets import SupervisionExample, SupervisionSet
from amaun.manager.utility_cv import FitGate, fit_with_cv

TASKS = ["sum", "qa", "sentiment", "qa-B", "instr", "dialog"]
NAMES = ["semantic.d_task_sq", "performance.L_val_gap", "resource.bytes"]


def _supervision(beneficial_every: int = 3) -> SupervisionSet:
    rng = random.Random(0)
    supervision = SupervisionSet(fit_split="utility_train")
    for i, (new, old) in enumerate(itertools.permutations(TASKS, 2)):
        beneficial = i % beneficial_every == 0
        vector = FeatureVector(names=list(NAMES), values=[rng.random(), rng.random() - 0.5, 1.0 + rng.random()],
                               family_slices={"semantic": (0, 1), "performance": (1, 2), "resource": (2, 3)})
        supervision.examples.append(SupervisionExample(
            candidate_id=f"seed-{old}", task_id=new, features=vector, proxy=False, proxy_steps=None,
            beneficial=beneficial,
            targets={"primary_score": 1.0 if beneficial else 0.9, "forgetting": 0.0 if beneficial else 0.03,
                     "confidence": rng.random(), "memory": 73.9, "communication": 73.9,
                     "long_term_loss": 1.0 + rng.random()}))
    return supervision


@pytest.fixture
def full_study_cfg():
    from pathlib import Path

    from amaun.config import load_config, resolve

    root = Path(__file__).resolve().parents[1]
    raw = load_config(root / "configs" / "experiments" / "l4_full_study.yaml", config_root=root / "configs")
    return resolve(raw, run_mode="dummy", experiment_id="pytest-utility-cv")


def test_fit_with_cv_runs_leave_one_task_out_and_takes_tau_from_out_of_fold(full_study_cfg):
    supervision = _supervision()
    gate = FitGate.from_config(full_study_cfg.get("full_study.utility_fit_gate"))
    result = fit_with_cv(full_study_cfg, supervision, gate=gate)

    assert result.fitted, result.reason
    assert [f["held_out_task"] for f in result.folds] == sorted(TASKS)
    for fold in result.folds:                          # no task overlap between a fold's fit and its test
        assert fold["held_out_task"] not in fold["train_tasks"]
    assert len(result.out_of_fold) == len(supervision.examples)
    assert result.thresholds is not None and "out-of-fold" in result.thresholds.notes
    assert set(result.per_head_oof_mse) and result.oof_decision_accuracy is not None
    assert result.network is not None and result.normalizer.fitted_on_split == "utility_train"


def test_fit_gate_refuses_one_class_supervision(full_study_cfg):
    gate = FitGate.from_config(full_study_cfg.get("full_study.utility_fit_gate"))
    result = fit_with_cv(full_study_cfg, _supervision(beneficial_every=1), gate=gate)
    assert not result.fitted and "outside" in result.reason

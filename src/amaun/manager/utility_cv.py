"""Fitting the utility network on measured supervision, with an honest out-of-fold check.

docs/l4_next_run_prompt.md items 12-14 (D-019):

1. **Gate.**  Fit only with >= ``min_pairs`` measured pairs and a beneficial fraction in
   ``[min_fraction, max_fraction]``.  Otherwise there is nothing for a ranking model to learn
   (run 1: 6 pairs, all harmful) and the study falls back to ``probe_rule``.
2. **Leave-one-new-task-out CV.**  For every new task, the normaliser and the network are fitted
   on the pairs of the *other* new tasks and predict this task's pairs.  Reported: out-of-fold MSE
   per head (in the fit fold's z units), and out-of-fold decision accuracy -- whether the highest
   predicted U picks the best measured candidate for that task.
3. **tau_u from out-of-fold utilities** with the configured threshold procedure, then a refit on
   every pair.  In-sample utilities would make tau look better separated than it is.

Nothing here reads TEST: the examples are VALIDATION measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from amaun.config import ResolvedConfig
from amaun.features.normalizer import FeatureNormalizer
from amaun.manager.decision import Thresholds
from amaun.manager.targets import SupervisionExample, SupervisionSet
from amaun.manager.threshold import select_thresholds
from amaun.manager.utility_network import (
    OUTCOME_HEADS,
    UtilityNetwork,
    UtilityNetworkConfig,
    train_utility_network,
)
from amaun.utils.logging import get_logger

log = get_logger("manager.utility_cv")


@dataclass
class FitGate:
    min_pairs: int
    min_fraction: float
    max_fraction: float

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "FitGate":
        return cls(int(section["min_pairs"]), float(section["min_beneficial_fraction"]),
                   float(section["max_beneficial_fraction"]))

    def check(self, flags: Sequence[bool]) -> tuple[bool, str]:
        n = len(flags)
        if n < self.min_pairs:
            return False, f"{n} measured pairs < {self.min_pairs}"
        fraction = sum(flags) / n
        if not self.min_fraction <= fraction <= self.max_fraction:
            return False, (f"beneficial fraction {fraction:.3f} outside "
                           f"[{self.min_fraction}, {self.max_fraction}]")
        return True, f"{n} pairs, beneficial fraction {fraction:.3f}"


@dataclass
class CVResult:
    fitted: bool
    reason: str
    network: UtilityNetwork | None = None
    normalizer: FeatureNormalizer | None = None
    thresholds: Thresholds | None = None
    folds: list[dict[str, Any]] = field(default_factory=list)
    out_of_fold: list[dict[str, Any]] = field(default_factory=list)
    per_head_oof_mse: dict[str, float] = field(default_factory=dict)
    oof_decision_accuracy: float | None = None
    refit_record: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted, "reason": self.reason,
            "thresholds": self.thresholds.as_dict() if self.thresholds else None,
            "folds": self.folds, "out_of_fold": self.out_of_fold,
            "per_head_oof_mse": self.per_head_oof_mse,
            "oof_decision_accuracy": self.oof_decision_accuracy,
            "refit": self.refit_record,
        }


def _fit(cfg: ResolvedConfig, examples: Sequence[SupervisionExample], fit_split: str
         ) -> tuple[UtilityNetwork, FeatureNormalizer, Any]:
    normalizer = FeatureNormalizer(epsilon=float(cfg.get("utility.normalisation.epsilon", 1e-8)))
    normalizer.fit([e.features for e in examples], split=fit_split, run_mode=cfg.run_mode.value)
    subset = SupervisionSet(examples=list(examples), fit_split=fit_split)
    features, targets = subset.tensors(normalizer, cfg.run_mode.value, OUTCOME_HEADS)
    network = UtilityNetwork(features.shape[1], UtilityNetworkConfig.from_config(cfg.section("utility")["network"]))
    record = train_utility_network(network, features, targets, config=cfg.section("utility")["training"],
                                   fit_split=fit_split, run_mode=cfg.run_mode.value, seed=cfg.seed)
    network.eval()
    return network, normalizer, record


def best_measured_candidate(rows: Sequence[Mapping[str, Any]]) -> str:
    """The acceptable candidate with the highest relative new score, else the highest score."""
    acceptable = [r for r in rows if r["beneficial"]]
    pool = acceptable or rows
    return max(pool, key=lambda r: (r["relative_new"], r["candidate_id"]))["candidate_id"]


def fit_with_cv(cfg: ResolvedConfig, supervision: SupervisionSet, *, gate: FitGate) -> CVResult:
    """Gate -> leave-one-new-task-out CV -> tau_u from OOF -> refit on all pairs."""
    examples = list(supervision.examples)
    ok, reason = gate.check([e.beneficial for e in examples])
    if not ok:
        log.warning("utility network NOT fitted: %s; amaun falls back to probe_rule", reason)
        return CVResult(fitted=False, reason=reason)

    heads = OUTCOME_HEADS
    new_tasks = sorted({e.task_id for e in examples})
    oof: list[dict[str, Any]] = []
    squared = {h: 0.0 for h in heads}
    folds = []
    for held_out in new_tasks:
        train = [e for e in examples if e.task_id != held_out]
        test = [e for e in examples if e.task_id == held_out]
        if not train:
            continue
        # The fold's rows are a subset of the supervision split, so its statistics are fitted under
        # that split's name (the normaliser's leakage guard admits only known splits); the held-out
        # task is recorded in ``folds``.
        network, normalizer, _ = _fit(cfg, train, fit_split=supervision.fit_split)
        subset = SupervisionSet(examples=test)
        features, targets = subset.tensors(normalizer, cfg.run_mode.value, heads)
        with torch.no_grad():
            utility = network.utility(features).tolist()
            predicted_z = network.predict_outcomes(features)
            target_z = network.standardise_targets(targets)
        for i, example in enumerate(test):
            for j, head in enumerate(heads):
                squared[head] += float((predicted_z[i, j] - target_z[i, j]) ** 2)
            oof.append({"task_id": held_out, "candidate_id": example.candidate_id,
                        "utility_oof": utility[i], "beneficial": example.beneficial,
                        "relative_new": float(example.targets["primary_score"]),
                        "forgetting": float(example.targets["forgetting"])})
        folds.append({"held_out_task": held_out, "n_train": len(train), "n_test": len(test),
                      "train_tasks": sorted({e.task_id for e in train})})

    accuracy_hits = []
    for task in new_tasks:
        rows = [r for r in oof if r["task_id"] == task]
        if not rows:
            continue
        picked = max(rows, key=lambda r: (r["utility_oof"], r["candidate_id"]))["candidate_id"]
        best = best_measured_candidate(rows)
        accuracy_hits.append(picked == best)
        for r in rows:
            r["picked_by_oof_utility"] = r["candidate_id"] == picked
            r["best_measured"] = r["candidate_id"] == best

    thresholds = select_thresholds(
        cfg.section("manager")["threshold"],
        observed_utilities=[r["utility_oof"] for r in oof],
        beneficial_flags=[r["beneficial"] for r in oof],
    )
    thresholds = Thresholds(tau_u=thresholds.tau_u, tau_reuse=thresholds.tau_reuse, source=thresholds.source,
                            selected_on=thresholds.selected_on,
                            notes=f"selected on out-of-fold utilities (leave-one-new-task-out): {thresholds.notes}")
    network, normalizer, record = _fit(cfg, examples, fit_split=supervision.fit_split)
    return CVResult(
        fitted=True, reason=reason, network=network, normalizer=normalizer, thresholds=thresholds,
        folds=folds, out_of_fold=oof,
        per_head_oof_mse={h: squared[h] / len(oof) for h in heads} if oof else {},
        oof_decision_accuracy=sum(accuracy_hits) / len(accuracy_hits) if accuracy_hits else None,
        refit_record=record.as_dict(),
    )

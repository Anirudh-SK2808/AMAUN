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
    RidgeUtility,
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


# --------------------------------------------------------------------------------------
# D-029: revised models -- constant features dropped, small network or ridge, nested CV
# --------------------------------------------------------------------------------------
#
# The first fit (200 epochs, [64, 64], all 54 features) memorised the 30 pairs: training MSE
# 0.03, out-of-fold primary-head MSE 4.8 (z units; predicting the mean scores 1.0).  Two smaller
# models are fitted on the features that vary, each with its one hyper-parameter (network epochs
# <= max_epochs, i.e. early stopping; ridge lambda) chosen by an INNER leave-one-task-out loop on
# the outer fold's training tasks only, so every out-of-fold number is honest.  One of them is
# then selected by a rule fixed before either was evaluated (:func:`select_revised`).


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                         # "nn" | "ridge"
    grid: tuple[float, ...]           # nn: epochs 1..max_epochs; ridge: lambdas
    hidden_dims: tuple[int, ...] = ()
    batch_size: int = 4


def revised_specs(section: Mapping[str, Any]) -> list[ModelSpec]:
    """``full_study.utility_models`` -> the candidate models."""
    nn_cfg, ridge_cfg = section["nn"], section["ridge"]
    return [
        ModelSpec("nn_small", "nn", tuple(range(1, int(nn_cfg["max_epochs"]) + 1)),
                  hidden_dims=tuple(int(d) for d in nn_cfg["hidden_dims"]), batch_size=int(nn_cfg["batch_size"])),
        ModelSpec("ridge", "ridge", tuple(float(x) for x in ridge_cfg["lambdas"])),
    ]


def _normalizer(cfg: ResolvedConfig, examples: Sequence[SupervisionExample], fit_split: str) -> FeatureNormalizer:
    return FeatureNormalizer(epsilon=float(cfg.get("utility.normalisation.epsilon", 1e-8))).fit(
        [e.features for e in examples], split=fit_split, run_mode=cfg.run_mode.value, drop_constant=True)


def _fit_revised(cfg: ResolvedConfig, examples: Sequence[SupervisionExample], spec: ModelSpec, hyper: float,
                 *, fit_split: str, epoch_callback: Any = None,
                 normalizer: FeatureNormalizer | None = None) -> tuple[torch.nn.Module, FeatureNormalizer]:
    normalizer = normalizer or _normalizer(cfg, examples, fit_split)
    features, targets = SupervisionSet(examples=list(examples), fit_split=fit_split).tensors(
        normalizer, cfg.run_mode.value, OUTCOME_HEADS)
    if spec.kind == "ridge":
        model: torch.nn.Module = RidgeUtility(features.shape[1], OUTCOME_HEADS, lam=hyper).fit(features, targets)
    else:
        net_cfg = UtilityNetworkConfig.from_config({**cfg.section("utility")["network"],
                                                    "hidden_dims": list(spec.hidden_dims)})
        model = UtilityNetwork(features.shape[1], net_cfg)
        train_cfg = {**cfg.section("utility")["training"], "epochs": int(hyper), "batch_size": spec.batch_size,
                     "normalize_targets": True}
        train_utility_network(model, features, targets, config=train_cfg, fit_split=fit_split,
                              run_mode=cfg.run_mode.value, seed=cfg.seed, epoch_callback=epoch_callback)
    model.eval()
    return model, normalizer


@torch.no_grad()
def _heldout_mse(model: torch.nn.Module, normalizer: FeatureNormalizer, examples: Sequence[SupervisionExample],
                 run_mode: str) -> float:
    """Mean z-space MSE over every head on held-out pairs (the early-stopping / lambda criterion)."""
    features, targets = SupervisionSet(examples=list(examples)).tensors(normalizer, run_mode, OUTCOME_HEADS)
    return float(((model.predict_outcomes(features) - model.standardise_targets(targets)) ** 2).mean())


def select_hyper(cfg: ResolvedConfig, examples: Sequence[SupervisionExample], spec: ModelSpec, *,
                 fit_split: str) -> tuple[float, dict[str, float]]:
    """Inner leave-one-task-out over ``examples`` only: the grid value with the lowest mean held-out
    MSE.  Ties go to the simpler model (fewer epochs, larger lambda)."""
    tasks = sorted({e.task_id for e in examples})
    totals = {value: 0.0 for value in spec.grid}
    for held_out in tasks:
        train = [e for e in examples if e.task_id != held_out]
        test = [e for e in examples if e.task_id == held_out]
        if spec.kind == "ridge":
            for lam in spec.grid:
                model, normalizer = _fit_revised(cfg, train, spec, lam, fit_split=fit_split)
                totals[lam] += _heldout_mse(model, normalizer, test, cfg.run_mode.value) / len(tasks)
        else:
            # One training run to max_epochs, recording the held-out loss after every epoch.
            curve: dict[float, float] = {}
            normalizer = _normalizer(cfg, train, fit_split)

            def track(epoch: int, model: torch.nn.Module) -> None:
                curve[float(epoch)] = _heldout_mse(model, normalizer, test, cfg.run_mode.value)

            _fit_revised(cfg, train, spec, max(spec.grid), fit_split=fit_split, epoch_callback=track,
                         normalizer=normalizer)
            for epoch in spec.grid:
                totals[epoch] += curve[float(epoch)] / len(tasks)
    simpler = (lambda v: v) if spec.kind == "nn" else (lambda v: -v)
    best = min(spec.grid, key=lambda v: (totals[v], simpler(v)))
    return best, {str(k): v for k, v in totals.items()}


def _decision_accuracy(oof: list[dict[str, Any]]) -> float | None:
    hits = []
    for task in sorted({r["task_id"] for r in oof}):
        rows = [r for r in oof if r["task_id"] == task]
        picked = max(rows, key=lambda r: (r["utility_oof"], r["candidate_id"]))["candidate_id"]
        best = best_measured_candidate(rows)
        hits.append(picked == best)
        for r in rows:
            r["picked_by_oof_utility"] = r["candidate_id"] == picked
            r["best_measured"] = r["candidate_id"] == best
    return sum(hits) / len(hits) if hits else None


def cv_model(cfg: ResolvedConfig, supervision: SupervisionSet, spec: ModelSpec) -> dict[str, Any]:
    """Nested leave-one-new-task-out CV of one candidate, then its refit on every pair."""
    examples, split = list(supervision.examples), supervision.fit_split
    run_mode = cfg.run_mode.value
    oof: list[dict[str, Any]] = []
    squared = {h: 0.0 for h in OUTCOME_HEADS}
    folds = []
    for held_out in sorted({e.task_id for e in examples}):
        train = [e for e in examples if e.task_id != held_out]
        test = [e for e in examples if e.task_id == held_out]
        hyper, _ = select_hyper(cfg, train, spec, fit_split=split)
        model, normalizer = _fit_revised(cfg, train, spec, hyper, fit_split=split)
        features, targets = SupervisionSet(examples=test).tensors(normalizer, run_mode, OUTCOME_HEADS)
        with torch.no_grad():
            utility = model.utility(features).tolist()
            error = model.predict_outcomes(features) - model.standardise_targets(targets)
        for i, example in enumerate(test):
            for j, head in enumerate(OUTCOME_HEADS):
                squared[head] += float(error[i, j] ** 2)
            oof.append({"task_id": held_out, "candidate_id": example.candidate_id, "utility_oof": utility[i],
                        "beneficial": example.beneficial, "relative_new": float(example.targets["primary_score"]),
                        "forgetting": float(example.targets["forgetting"])})
        folds.append({"held_out_task": held_out, "train_tasks": sorted({e.task_id for e in train}),
                      "hyper": hyper, "n_active_features": normalizer.n_active})
    accuracy = _decision_accuracy(oof)
    hyper, curve = select_hyper(cfg, examples, spec, fit_split=split)
    model, normalizer = _fit_revised(cfg, examples, spec, hyper, fit_split=split)
    return {"name": spec.name, "kind": spec.kind, "hyper_name": "epochs" if spec.kind == "nn" else "lambda",
            "final_hyper": hyper, "final_selection_curve": curve, "folds": folds, "out_of_fold": oof,
            "per_head_oof_mse": {h: v / len(oof) for h, v in squared.items()} if oof else {},
            "oof_decision_accuracy": accuracy, "n_active_features": normalizer.n_active,
            "active_features": normalizer.active_names, "model": model, "normalizer": normalizer}


#: D-029, fixed before either revised model was evaluated.
SELECTION_RULE = ("higher out-of-fold decision accuracy; tie -> lower out-of-fold primary_score MSE; "
                  "tie -> ridge (the simpler model)")


def select_revised(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(reports, key=lambda r: (-(r["oof_decision_accuracy"] or 0.0),
                                       r["per_head_oof_mse"].get("primary_score", float("inf")),
                                       0 if r["kind"] == "ridge" else 1))


@dataclass
class RevisedResult:
    fitted: bool
    reason: str
    selected: str | None = None
    model: torch.nn.Module | None = None
    normalizer: FeatureNormalizer | None = None
    thresholds: Thresholds | None = None
    reports: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"fitted": self.fitted, "reason": self.reason, "selected": self.selected,
                "selection_rule": SELECTION_RULE,
                "thresholds": self.thresholds.as_dict() if self.thresholds else None,
                "candidates": [{k: v for k, v in r.items() if k not in ("model", "normalizer")}
                               for r in self.reports]}


def fit_revised(cfg: ResolvedConfig, supervision: SupervisionSet, *, gate: FitGate,
                specs: Sequence[ModelSpec]) -> RevisedResult:
    """Gate -> nested CV of every candidate -> rule-based selection -> tau_u from its out-of-fold."""
    ok, reason = gate.check([e.beneficial for e in supervision.examples])
    if not ok:
        return RevisedResult(fitted=False, reason=reason)
    reports = [cv_model(cfg, supervision, spec) for spec in specs]
    chosen = select_revised(reports)
    oof = chosen["out_of_fold"]
    picked = select_thresholds(cfg.section("manager")["threshold"],
                               observed_utilities=[r["utility_oof"] for r in oof],
                               beneficial_flags=[r["beneficial"] for r in oof])
    thresholds = Thresholds(tau_u=picked.tau_u, tau_reuse=picked.tau_reuse, source=picked.source,
                            selected_on=picked.selected_on,
                            notes=f"{chosen['name']}: selected on nested out-of-fold utilities: {picked.notes}")
    log.info("D-029 utility models: %s; selected %s", {r["name"]: (r["oof_decision_accuracy"],
             round(r["per_head_oof_mse"].get("primary_score", float("nan")), 3)) for r in reports}, chosen["name"])
    return RevisedResult(fitted=True, reason=reason, selected=chosen["name"], model=chosen["model"],
                         normalizer=chosen["normalizer"], thresholds=thresholds, reports=list(reports))

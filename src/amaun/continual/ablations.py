"""D-030: ablations of the finished l4_full_study, scored against the run's own measurements.

The run itself is never modified: its outputs are read, and everything new is written elsewhere.

**Feature ablations (A1, A6, A7, A12) are replayed offline.** Every input of a utility policy's
Phase 5 decision was logged at decision time -- the reuse gate's verdict, the SUPPORT probes, the
prescreen and every candidate's feature vector -- so a variant's decision is recomputed exactly
by changing only the features (and therefore the fitted model and its utilities). The variant's
utility model is fitted by the run's own D-029 protocol on the run's 30 Phase 3 pairs, with the
same transformation applied to their features. Decisions are scored by
:mod:`amaun.continual.study_scoring` against the measured TEST options. Two identity checks gate
everything: replaying with the run's frozen models must reproduce its logged utilities and
decisions, and re-scoring its logged decisions must reproduce its stored tables.

**A12** swaps the task representation for ``token_stats`` (support-set word statistics): only
``semantic.d_task_sq``, ``semantic.d_task_l2`` and ``semantic.cos_task`` depend on it, and they are
recomputed from the SUPPORT rows of the new task and of the candidate's own task.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from amaun.continual import study_criteria as criteria
from amaun.continual import study_scoring as scoring
from amaun.data.splits import manager_guard
from amaun.features.base import FeatureVector
from amaun.features.normalizer import FeatureNormalizer
from amaun.manager.decision import Thresholds
from amaun.manager.retrieval import score_candidates
from amaun.manager.targets import SupervisionExample, SupervisionSet
from amaun.manager.task_representation import build_task_representation
from amaun.manager.utility_network import load_utility_model

UTILITY_POLICIES = ("amaun", "amaun_train_first")
#: The semantic features that depend on the task representation (A12).
TASK_REPRESENTATION_FEATURES = ("semantic.d_task_sq", "semantic.d_task_l2", "semantic.cos_task")


# --------------------------------------------------------------------------------------
# Feature vectors
# --------------------------------------------------------------------------------------


def family_of(name: str) -> str:
    return name.split(".", 1)[0]


def vector_from_values(values: Mapping[str, float], missing: Sequence[str] = ()) -> FeatureVector:
    """Rebuild a logged ``{name: value}`` vector (families are contiguous in name order)."""
    names = list(values)
    slices: dict[str, tuple[int, int]] = {}
    for index, name in enumerate(names):
        start, _ = slices.get(family_of(name), (index, index))
        slices[family_of(name)] = (start, index + 1)
    return FeatureVector(names=names, values=[float(values[n]) for n in names], family_slices=slices,
                         missing=list(missing))


def vector_from_logged(logged: Mapping[str, Any]) -> FeatureVector:
    """A Phase 3 ``features`` record (names / values / family_slices / missing)."""
    return FeatureVector(names=list(logged["names"]), values=list(logged["values"]),
                         family_slices={k: tuple(v) for k, v in logged["family_slices"].items()},
                         missing=list(logged.get("missing", [])))


def subset(vector: FeatureVector, *, keep: Sequence[str] | None = None,
           drop: Sequence[str] | None = None) -> FeatureVector:
    """The vector restricted to the kept families (or without the dropped ones)."""
    def wanted(name: str) -> bool:
        family = family_of(name)
        return (keep is None or family in keep) and (drop is None or family not in drop)

    kept = {n: v for n, v in zip(vector.names, vector.values) if wanted(n)}
    if not kept:
        raise ValueError(f"feature subset keep={keep} drop={drop} leaves no features")
    return vector_from_values(kept, [m for m in vector.missing if wanted(m)])


def token_stats_embedding(task_id: str, splits: Mapping[str, Any], task_repr: Mapping[str, Any]) -> list[float]:
    """A12's task representation: support-set word statistics (no model)."""
    representation = build_task_representation(task_id, manager_guard(splits[task_id]),
                                               config={**task_repr, "encoder": "token_stats"}, backbone=None)
    return list(representation.embedding)


def with_task_representation(vector: FeatureVector, task_embedding: Sequence[float],
                             candidate_embedding: Sequence[float]) -> FeatureVector:
    """Recompute the task-representation features as :class:`SemanticFeatures` does."""
    left, right = list(task_embedding), list(candidate_embedding)
    if len(left) != len(right) or not left:
        raise ValueError("task and candidate embeddings differ in size")
    sq = sum((a - b) ** 2 for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    new = {"semantic.d_task_sq": sq, "semantic.d_task_l2": math.sqrt(sq),
           "semantic.cos_task": sum(a * b for a, b in zip(left, right)) / norms if norms > 0 else 0.0}
    values = [new.get(name, value) for name, value in zip(vector.names, vector.values)]
    return FeatureVector(names=list(vector.names), values=values, family_slices=dict(vector.family_slices),
                         missing=list(vector.missing))


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    keep: tuple[str, ...] | None = None
    drop: tuple[str, ...] | None = None
    task_representation: str | None = None          # "token_stats" for A12

    @classmethod
    def from_config(cls, name: str, section: Mapping[str, Any]) -> "FeatureVariant":
        keep, drop = section.get("keep_families"), section.get("drop_families")
        return cls(name, tuple(keep) if keep else None, tuple(drop) if drop else None,
                   section.get("task_representation"))

    def transform(self, embeddings: Mapping[str, Sequence[float]] | None = None
                  ) -> Callable[[FeatureVector, str, str], FeatureVector]:
        """``(vector, new_task, candidate_task) -> vector`` for this variant."""
        if self.task_representation not in (None, "token_stats"):
            raise ValueError(f"unknown task representation '{self.task_representation}'")
        if self.task_representation and embeddings is None:
            raise ValueError(f"{self.name} needs token_stats embeddings for every task")

        def apply(vector: FeatureVector, new_task: str, candidate_task: str) -> FeatureVector:
            if self.task_representation:
                vector = with_task_representation(vector, embeddings[new_task], embeddings[candidate_task])
            if self.keep or self.drop:
                vector = subset(vector, keep=self.keep, drop=self.drop)
            return vector

        return apply


# --------------------------------------------------------------------------------------
# The frozen run
# --------------------------------------------------------------------------------------


@dataclass
class FrozenModel:
    model: Any
    normalizer: FeatureNormalizer
    thresholds: Thresholds


class FrozenRun:
    """Read-only view of a finished l4_full_study run."""

    TASKS_ORDER_FILE = "assignments.json"

    def __init__(self, root: Path | str, *, run_mode: str) -> None:
        self.root = Path(root)
        self.d = self.root / "decision_study"
        self.run_mode = run_mode
        load = lambda name: json.loads((self.d / name).read_text())
        self.assignments = load(self.TASKS_ORDER_FILE)
        self.evaluation_tasks = [t["task_id"] for t in self.assignments["evaluation_tasks"]]
        self.evaluations = {t: load(f"evaluation_{t}.json") for t in self.evaluation_tasks}
        self.supervision = load("utility_supervision.json")
        self.policy_comparison = load("policy_comparison.json")
        self.fusion_selection = load("fusion_selection.json")
        self.standalone_test = load("standalone_test_scores.json")
        self.seed_task = {a["adapter_id"]: a["task_id"] for a in self.assignments["initial_adapters"]}

    def standalone(self, task_id: str) -> dict[str, float]:
        """Seed task -> standalone TEST score, in Phase 5's candidate order (the run's sum order)."""
        order = list(self.evaluations[task_id]["probes"]["candidates"])
        return {self.standalone_test[a]["task_id"]: self.standalone_test[a]["primary_score"] for a in order}

    def model(self, family: str = "amaun") -> FrozenModel | None:
        directory = self.root / "utility_network" / ("v0" if family == "amaun_v0" else "")
        if not (directory / "utility_network.pt").exists():
            return None
        model, _ = load_utility_model(directory / "utility_network.pt", expected_run_mode=self.run_mode)
        return FrozenModel(model, FeatureNormalizer.load(directory / "normalizer.json", expected_run_mode=self.run_mode),
                           Thresholds(**json.loads((directory / "thresholds.json").read_text())))

    def vectors(self, task_id: str) -> dict[str, FeatureVector]:
        logged = self.evaluations[task_id]["features"]
        return {c: vector_from_values(v["values"], v["missing"]) for c, v in logged.items()}


# --------------------------------------------------------------------------------------
# Replay and scoring
# --------------------------------------------------------------------------------------


def replay_decisions(evaluation: Mapping[str, Any], vectors: Mapping[str, FeatureVector], frozen: FrozenModel, *,
                     run_mode: str, equivalence: float,
                     policies: Sequence[str] = UTILITY_POLICIES) -> tuple[dict[str, Any], dict[str, float]]:
    """A utility policy's Phase 5 decision from the logged gate / probes / prescreen (FullStudy.decide)."""
    scored = score_candidates(network=frozen.model, normalizer=frozen.normalizer, vectors=dict(vectors),
                              run_mode=run_mode)
    utilities = {s.adapter_id: s.utility for s in scored}
    probes, gate, prescreen = evaluation["probes"], evaluation.get("reuse_gate") or {}, evaluation["prescreen"]
    decisions = {}
    for policy in policies:
        if policy in ("amaun", "amaun_v0"):
            reuse = gate.get("adapter_id") if gate.get("fired") else None
        else:   # amaun_train_first (D-020)
            best = max(utilities, key=lambda c: (utilities[c], c))
            reuse = best if utilities[best] >= frozen.thresholds.tau_reuse and prescreen.get(best) else None
        decisions[policy] = criteria.amaun_order(
            reuse_choice=reuse, backbone=probes["backbone"], candidates=probes["candidates"], utilities=utilities,
            tau_u=frozen.thresholds.tau_u, equivalence=equivalence)
    return decisions, utilities


def supervision_from_frozen(frozen_run: FrozenRun, transform: Callable[[FeatureVector, str, str], FeatureVector]
                            ) -> SupervisionSet:
    """The run's 30 Phase 3 pairs with ``transform`` applied to each pair's features."""
    supervision = SupervisionSet(fit_split="utility_train")
    for m in frozen_run.supervision["measurements"]:
        vector = transform(vector_from_logged(m["features"]), m["new_task"], m["old_task"])
        supervision.examples.append(SupervisionExample(
            candidate_id=m["old_adapter"], task_id=m["new_task"], features=vector, targets=m["targets"],
            proxy=False, proxy_steps=None, beneficial=m["beneficial"]))
    supervision.n_pairs_considered = supervision.n_pairs_evaluated = len(supervision.examples)
    return supervision


def score_with_decisions(frozen_run: FrozenRun, decisions_by_task: Mapping[str, Mapping[str, Any]], *,
                         caps: Sequence[int | None], policies: Sequence[str], equivalence: float,
                         forgetting: float, resamples: int,
                         extra_options: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Per-task outcomes and pooled policy tables for the given decisions, on the run's measurements."""
    payloads, per_task = {}, {}
    eligible = list(frozen_run.policy_comparison["eligible_tasks"])
    for task_id, evaluation in frozen_run.evaluations.items():
        options = {**evaluation["options"], **((extra_options or {}).get(task_id) or {})}
        interval = evaluation["policy_outcomes"]["no_cap"].get("criteria_scale", "").startswith("interval")
        outcomes = {scoring.cap_label(cap): scoring.score_task(
            options, decisions_by_task[task_id], cap=cap, n_candidates=len(evaluation["probes"]["candidates"]),
            interval=interval, standalone=frozen_run.standalone(task_id), equivalence=equivalence,
            forgetting=forgetting, resamples=resamples) for cap in caps}
        payloads[task_id] = {"options": options, "policy_outcomes": outcomes}
        per_task[task_id] = outcomes
    pooled = scoring.compare_policies(
        payloads, eligible=eligible, non_discriminating=frozen_run.policy_comparison["non_discriminating_tasks"],
        policies=policies, caps=caps, resamples=resamples)
    return {"per_task": per_task, "pooled": pooled}


def unmeasured_fuse(per_task: Mapping[str, Mapping[str, Any]], policies: Sequence[str]) -> list[tuple[str, str]]:
    """(task, candidate) for every FUSE option a policy executed that has not been measured."""
    out = set()
    for task_id, outcomes in per_task.items():
        for table in outcomes.values():
            for policy in policies:
                row = table["policies"].get(policy) or {}
                executed = row.get("executed") or ""
                if not row.get("measured", True) and executed.startswith("FUSE:"):
                    out.add((task_id, executed.partition(":")[2]))
    return sorted(out)


# --------------------------------------------------------------------------------------
# Identity checks (run before any ablation is scored)
# --------------------------------------------------------------------------------------


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """Every place two JSON-like values differ (floats compared exactly)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ {sorted(set(a) ^ set(b))}"]
        return [d for k in a for d in differences(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def identity_replay(run: FrozenRun, *, equivalence: float, utility_tolerance: float = 1e-6) -> dict[str, Any]:
    """Check 1: replaying with the run's frozen models reproduces its logged utilities and decisions."""
    worst, mismatches = 0.0, []
    for family, policies in (("amaun", ("amaun", "amaun_train_first")), ("amaun_v0", ("amaun_v0",))):
        frozen = run.model(family)
        if frozen is None:
            continue
        for task_id, evaluation in run.evaluations.items():
            if not any(p in evaluation["decisions"] for p in policies):
                continue
            decisions, utilities = replay_decisions(evaluation, run.vectors(task_id), frozen, run_mode=run.run_mode,
                                                    equivalence=equivalence, policies=policies)
            logged = (evaluation.get("utilities") or {}).get(family) or {}
            for candidate, value in utilities.items():
                worst = max(worst, abs(value - logged.get(candidate, float("inf"))))
            for policy in policies:
                want = evaluation["decisions"][policy]
                if decisions[policy] != {k: want.get(k) for k in decisions[policy]}:
                    mismatches.append({"task": task_id, "policy": policy, "replayed": decisions[policy], "logged": want})
    return {"ok": not mismatches and worst <= utility_tolerance, "max_utility_difference": worst,
            "utility_tolerance": utility_tolerance, "decision_mismatches": mismatches}


def identity_scoring(run: FrozenRun, *, caps: Sequence[int | None], policies: Sequence[str], equivalence: float,
                     forgetting: float, resamples: int) -> dict[str, Any]:
    """Check 2: re-scoring the run's logged decisions reproduces its stored tables exactly."""
    scored = score_with_decisions(run, {t: e["decisions"] for t, e in run.evaluations.items()}, caps=caps,
                                  policies=policies, equivalence=equivalence, forgetting=forgetting,
                                  resamples=resamples)
    found = []
    for task_id, outcomes in scored["per_task"].items():
        found += differences(json.loads(json.dumps(outcomes, default=str)),
                             {k: run.evaluations[task_id]["policy_outcomes"][k] for k in outcomes}, task_id)
    found += differences(json.loads(json.dumps(scored["pooled"], default=str)), run.policy_comparison, "pooled")
    return {"ok": not found, "n_differences": len(found), "first_differences": found[:20]}

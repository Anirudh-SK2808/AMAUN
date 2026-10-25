"""Phase 5 scoring of the l4_full_study as pure functions of measured options and decisions.

Factored out of :class:`amaun.continual.full_study.FullStudy` (unchanged in content) so that the
same code scores the run itself and, offline, the D-030 ablations against the run's measurements.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from amaun.continual import study_criteria as criteria


def cap_label(cap: int | None) -> str:
    return "no_cap" if cap is None else f"cap_{cap}"


def score_task(options: Mapping[str, Mapping[str, Any]], decisions: Mapping[str, Mapping[str, Any]], *,
               cap: int | None, n_candidates: int, interval: bool, standalone: Mapping[str, float],
               equivalence: float, forgetting: float, resamples: int) -> dict[str, Any]:
    """Items 6-7 for one task under one cap setting, from the measured options.

    ``standalone`` maps every seed task to its standalone TEST score (the secondary, run-1
    average-performance columns); ``interval`` selects D-027's gain-retention criteria.
    """
    at_cap = cap is not None and n_candidates >= cap
    measures = {k: criteria.OptionMeasure(
        key=k, new_score=o["new_task_score"], old_drops=o["old_drops"], adapters_after=o["adapters_after"],
        training_seconds=o["training_seconds"], stored_bytes=o["stored_bytes"], per_example=o["per_example"])
        for k, o in options.items()}
    if criteria.CREATE not in measures:
        return {"error": "CREATE was not measured; nothing can be judged", "oracle": {"option": None},
                "policies": {}}
    create = measures[criteria.CREATE]
    # D-027: an interval-scale metric (rescaled BERTScore) is judged by gain retention over the
    # zero-shot score, which the BACKBONE option measures on the same TEST examples.
    zero_shot = measures.get(criteria.BACKBONE) if interval else None
    if interval and zero_shot is None:
        return {"error": "BACKBONE was not measured; an interval-scale task cannot be judged",
                "oracle": {"option": None}, "policies": {}}
    zs = None if zero_shot is None else zero_shot.new_score
    oracle = criteria.choose_oracle(measures, at_cap=at_cap, equivalence=equivalence, forgetting=forgetting,
                                    zero_shot=zs)
    acceptability = {k: {**criteria.acceptability(m, create, equivalence=equivalence, forgetting=forgetting,
                                                  resamples=resamples, zero_shot=zero_shot),
                         "feasible": criteria.feasible(m, at_cap=at_cap)}
                     for k, m in measures.items()}
    # Secondary, run-1 definitions: unweighted mean of the new and every seed task's score.
    average = {k: (o["new_task_score"] + sum(o["old_task_scores"].get(t, s) for t, s in standalone.items()))
               / (1 + len(standalone)) for k, o in options.items()}
    feasible_keys = [k for k in options if acceptability[k]["feasible"]]
    oracle_by_average = max(feasible_keys, key=lambda k: (average[k], -options[k]["adapters_after"]))
    separate = decisions.get("always_separate", {}).get("option")
    separate_key = criteria.executed_option(separate, at_cap=at_cap) if separate else None
    policies = {}
    for name, decision in decisions.items():
        chosen = decision.get("option")
        if chosen is None:
            policies[name] = {"chosen": None, "measured": False, "reason": decision.get("reason")}
            continue
        executed = criteria.executed_option(chosen, at_cap=at_cap)
        if executed not in measures:
            policies[name] = {"chosen": chosen, "executed": executed, "measured": False,
                              "reason": "option not measured"}
            continue
        quality = criteria.decision_quality(executed, measures, oracle["option"], equivalence=equivalence,
                                            forgetting=forgetting, zero_shot=zs)
        ci = None
        if separate_key in measures and executed != separate_key:
            ci = criteria.paired_bootstrap_ci(measures[executed].per_example, measures[separate_key].per_example,
                                              resamples=resamples)
        flags = []
        if not quality["acceptable"]:
            flags.append("unacceptable: below the equivalence bar or > 1% forgetting")
        if quality["cost_excess"]["adapters"] > 0 or quality["cost_excess"]["training_seconds"] > 0:
            flags.append("costlier than the oracle")
        policies[name] = {
            "chosen": chosen, "executed": executed, "infeasible_create_replaced": chosen != executed,
            "measured": True, **quality, "flags": flags, "ci_vs_always_separate": ci,
            "new_task_score": measures[executed].new_score,
            "average_performance": average[executed],
            "regret_vs_oracle": average[oracle_by_average] - average[executed],
            "fallback": decision.get("fallback"),
        }
    return {"at_cap": at_cap, "cap": cap, "oracle": oracle, "acceptability": acceptability,
            "criteria_scale": "interval: gain retention over zero-shot (D-027)" if zs is not None else "ratio",
            "average_performance": average, "oracle_option": oracle_by_average,
            "oracle_option_note": "run-1 definition (best average performance), secondary",
            "policies": policies}


def compare_policies(payloads: Mapping[str, Mapping[str, Any]], *, eligible: Sequence[str],
                     non_discriminating: Sequence[str], policies: Sequence[str], caps: Sequence[int | None],
                     resamples: int) -> dict[str, Any]:
    """Policy tables over the eligible tasks, per cap setting (items 7, 10).

    ``payloads[task]`` holds that task's ``options`` and ``policy_outcomes`` (an
    ``evaluation_<task>.json``).
    """
    out: dict[str, Any] = {"eligible_tasks": list(eligible), "non_discriminating_tasks": list(non_discriminating)}
    for cap in caps:
        label = cap_label(cap)
        table = {}
        for policy in policies:
            rows, diffs_a, diffs_b = [], [], []
            for task_id in eligible:
                payload = payloads[task_id]
                row = payload["policy_outcomes"][label]["policies"].get(policy) or {}
                rows.append((task_id, row))
                sep = payload["policy_outcomes"][label]["policies"].get("always_separate") or {}
                if row.get("measured") and sep.get("measured"):
                    diffs_a += payload["options"][row["executed"]]["per_example"]
                    diffs_b += payload["options"][sep["executed"]]["per_example"]
            measured = [r for _, r in rows if r.get("measured")]

            def mean(values):
                values = list(values)
                return sum(values) / len(values) if values else None
            table[policy] = {
                "decisions": {t: r.get("executed", r.get("chosen")) for t, r in rows},
                "n_measured": len(measured),
                "decision_correct": mean(1.0 if r["decision_correct"] else 0.0 for r in measured),
                "acceptable": mean(1.0 if r["acceptable"] else 0.0 for r in measured),
                "mean_quality_shortfall": mean(r["quality_shortfall"] for r in measured),
                "mean_cost_excess_adapters": mean(r["cost_excess"]["adapters"] for r in measured),
                "mean_cost_excess_training_seconds": mean(r["cost_excess"]["training_seconds"] for r in measured),
                "mean_forgetting": mean(r["forgetting"] for r in measured),
                "max_forgetting": max((r["forgetting"] for r in measured), default=None),
                "mean_average_performance": mean(r["average_performance"] for r in measured),
                "mean_regret_vs_oracle": mean(r["regret_vs_oracle"] for r in measured),
                "pooled_ci_vs_always_separate": (criteria.paired_bootstrap_ci(
                    diffs_a, diffs_b, resamples=resamples)
                    if diffs_a and policy != "always_separate" else None),
            }
        out[label] = table
    out["caveat"] = ("one seed: CIs resample TEST examples only; scores pooled across tasks mix metrics "
                     "and are descriptive")
    return out

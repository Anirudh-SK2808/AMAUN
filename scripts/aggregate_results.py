#!/usr/bin/env python
"""Aggregate arm summaries into comparison tables, with budget-parity and seed gating.

Reads ``results/<run_mode>/*/summary.json``.  Defaults to ``run_mode=actual`` per idea.md
S0.1.1, and visibly labels anything else as smoke-test output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.config import RunMode
from amaun.errors import BudgetParityError, InsufficientSeedsError
from amaun.evaluation.stats import aggregate_seeds, compare_arms
from amaun.training.budget import ArmBudget, BudgetLedger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # idea.md S0.1.1: reporting hooks MUST default to `actual`.
    parser.add_argument("--run-mode", default=RunMode.ACTUAL.value,
                        choices=[m.value for m in RunMode])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("ARM_A", "ARM_B"), default=None)
    parser.add_argument("--budget-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.results_root) / args.run_mode
    if not root.exists():
        print(f"no results under {root}")
        return 1

    summaries = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        with summary_path.open() as handle:
            summaries[summary_path.parent.name] = json.load(handle)
    if not summaries:
        print(f"no summary.json files under {root}")
        return 1

    mode = RunMode(args.run_mode)
    if not mode.is_reportable:
        print(f"\n[SMOKE TEST ({mode.value}) -- not a reported result]")
        print("These figures are diagnostic only and must never appear in a results table.\n")

    header = f"{'arm':26s} {'init':18s} {'avg':>8s} {'forget':>8s} {'BWT':>8s} {'FWT':>8s} {'adapters':>9s} {'steps':>7s}"
    print(header)
    print("-" * len(header))
    rows = []
    for arm, summary in summaries.items():
        metrics = summary.get("continual_metrics") or {}
        repository = summary.get("final_repository") or {}
        budget = summary.get("budget") or {}

        def fmt(value):
            return f"{value:8.4f}" if isinstance(value, (int, float)) else f"{'n/a':>8s}"

        print(f"{arm:26s} {str(summary.get('repository_init')):18s} "
              f"{fmt(metrics.get('average_performance'))} {fmt(metrics.get('forgetting'))} "
              f"{fmt(metrics.get('backward_transfer'))} {fmt(metrics.get('forward_transfer'))} "
              f"{repository.get('n_adapters', 0):9d} {budget.get('optimizer_steps', 0):7d}")
        rows.append({
            "arm": arm,
            "repository_init": summary.get("repository_init"),
            "average_performance": metrics.get("average_performance"),
            "forgetting": metrics.get("forgetting"),
            "backward_transfer": metrics.get("backward_transfer"),
            "forward_transfer": metrics.get("forward_transfer"),
            "n_adapters": repository.get("n_adapters"),
            "tasks_served_per_adapter": (
                sum(len(v) for v in (repository.get("tasks_served") or {}).values())
                / max(1, repository.get("n_adapters") or 1)
            ),
            "stored_parameters": repository.get("total_parameters"),
            "storage_bytes": repository.get("storage_bytes"),
            "optimizer_steps": budget.get("optimizer_steps"),
            "tokens_seen": budget.get("tokens_seen"),
            "outcome_counts": json.dumps(summary.get("outcome_counts") or {}),
        })

    out = Path(args.out) if args.out else root / "comparison.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ntable: {out}")

    if args.compare:
        arm_a, arm_b = args.compare
        missing = [arm for arm in (arm_a, arm_b) if arm not in summaries]
        if missing:
            print(f"cannot compare: missing arm(s) {missing}")
            return 1

        # Budget parity first: a difference under unequal budgets is not attributable.
        ledger = BudgetLedger(tolerance=args.budget_tolerance)
        for arm in (arm_a, arm_b):
            budget = summaries[arm].get("budget") or {}
            known = {k: v for k, v in budget.items() if k in ArmBudget.__dataclass_fields__}
            ledger.arms[arm] = ArmBudget(**{**known, "arm_id": arm})
        try:
            verdict = ledger.check_parity(arm_a, arm_b)
            print(f"\nbudget parity: fair={verdict['fair']}")
        except BudgetParityError as exc:
            print(f"\nBUDGET PARITY REFUSED:\n{exc}")
            return 2

        scores_a = _per_task_scores(summaries[arm_a])
        scores_b = _per_task_scores(summaries[arm_b])
        try:
            result = compare_arms(arm_a, arm_b, scores_a, scores_b)
            print(f"\n{result.interpretation}")
            print(f"  p={result.p_value:.4f}  d={result.effect_size_d:.3f}  "
                  f"pairs={result.n_pairs}  seeds={result.n_seeds}")
        except InsufficientSeedsError as exc:
            print(f"\nCOMPARISON REFUSED: {exc}")
            return 2
    return 0


def _per_task_scores(summary: dict) -> dict[tuple[str, int], float]:
    """Per-task final scores keyed by ``(task_id, seed)`` for the paired test."""
    per_task = (summary.get("continual_metrics") or {}).get("per_task_final") or {}
    seed = int((summary.get("seed") or 0))
    return {(task_id, seed): float(score) for task_id, score in per_task.items()}


if __name__ == "__main__":
    raise SystemExit(main())

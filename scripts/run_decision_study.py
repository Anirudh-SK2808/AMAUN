#!/usr/bin/env python
"""Run the L4 decision study: initial adapters -> measured utility supervision -> similar-task
evaluation of the utility network against simple policies -> memory audit.

See docs/l4_decision_study.md for the design and configs/experiments/l4_decision_study.yaml for
every value.  ``--prepare-only`` builds and audits the data splits and exits, which needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.continual.decision_study import DecisionStudy
from amaun.data.splits import SplitName
from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.models.backbone import load_backbone
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--prepare-only", action="store_true",
                        help="build, cap and audit the splits, print their sizes, and exit")
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    data = prepare_data(cfg)
    split_sizes = {task_id: splits.counts() for task_id, splits in data.splits.items()}
    tracker.write_json("decision_study/data.json", {
        "split_sizes": split_sizes,
        "blocked": data.blocked,
        "leakage_audit": data.leakage_report,
        "provenance": {task_id: splits.provenance for task_id, splits in data.splits.items()},
    })
    print(f"{smoke_label(cfg.run_mode)}")
    for task_id, counts in split_sizes.items():
        print(f"  {task_id:<10} " + "  ".join(f"{name.value}={counts[name.value]}" for name in SplitName))
    if data.blocked:
        print(f"ERROR: blocked tasks {data.blocked}; the study needs every configured task.")
        return 2
    if args.prepare_only:
        print(f"\nsplits written to {tracker.paths.root / 'decision_study' / 'data.json'}")
        return 0

    backbone = load_backbone(cfg)
    study = DecisionStudy(cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits,
                          tracker=tracker)
    summary = study.run()

    if summary.get("aborted"):
        print(f"\nABORTED: {summary['aborted']}")
        return 3
    print("\npolicy comparison (descriptive; one seed):")
    print(json.dumps(summary["policy_comparison"], indent=2, default=str))
    print(f"\nmemory audit: {summary['memory_audit'].get('verdict')}")
    print(f"total wall time: {summary['total_wall_seconds'] / 3600:.2f} h "
          f"(budget {summary['time_budget_hours']} h, fits: {summary['fits_time_budget']})")
    if summary["skipped"] or summary["failures"]:
        print(f"skipped: {len(summary['skipped'])}  failures: {len(summary['failures'])} "
              f"(see decision_study/summary.json)")
    print(f"results: {tracker.paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

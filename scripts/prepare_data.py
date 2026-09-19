#!/usr/bin/env python
"""Build, validate, and report the four-way splits for every task.

Runs the leakage audit and writes a dataset report.  This is PHASE 5 of the execution order
(dataset validation) and must pass before any training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.utils.logging import get_logger
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker

log = get_logger("scripts.prepare_data")


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()

    data = prepare_data(cfg)

    report = {
        "run_mode": cfg.run_mode.value,
        "provider": cfg.get("data.provider"),
        "split_ratios": cfg.get("data.split_ratios"),
        "data_seed": cfg.get("data.seed"),
        "tasks": [
            {
                **task.as_dict(),
                "split_counts": data.splits[task.task_id].counts(),
                "split_provenance": data.splits[task.task_id].provenance,
            }
            for task in data.tasks
        ],
        "unpinned_tasks": data.unpinned,
        "leakage_audit": data.leakage_report,
    }
    path = tracker.write_json("dataset_report.json", report)

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"dataset report: {path}")
    for task in data.tasks:
        counts = data.splits[task.task_id].counts()
        print(f"  {task.task_id:10s} {task.task_type.value:22s} {counts}  license={task.license}")
    if data.unpinned:
        print(f"\n  UNPINNED (D-006, not loadable): {data.unpinned}")
    audit = data.leakage_report
    print(f"\nleakage audit: {audit['pairs_checked']} split pairs checked, clean={audit['clean']}")
    if audit["overlaps"]:
        print(f"  non-critical overlaps: {audit['overlaps']}")
    return 0 if audit["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

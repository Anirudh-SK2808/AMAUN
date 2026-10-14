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

from amaun.experiment import BLOCK_REASONS, add_common_arguments, prepare_data, resolve_from_args
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

    snapshots_summary = [
        {
            "task": task_id,
            "upstream_split": report.upstream_split,
            "key": report.snapshot_key,
            "source": report.source,
            "rows_consumed": report.rows_consumed,
            "n_examples": report.n_examples,
            "exhausted": report.exhausted,
            "skip": report.skip,
            "rows_yielding_no_examples": report.rows_yielding_no_examples,
            "over_length_fraction": report.over_length_fraction,
            "over_length_metric": report.over_length_metric,
        }
        for task_id, reports in data.load_reports.items()
        for report in reports
    ]
    blocked_table = {
        task_id: {
            "code": code,
            "decision": BLOCK_REASONS[code]["decision"],
            "reason": BLOCK_REASONS[code]["reason"],
        }
        for task_id, code in data.blocked.items()
    }

    report = {
        "run_mode": cfg.run_mode.value,
        "label": smoke_label(cfg.run_mode),
        "provider": cfg.get("data.provider"),
        "split_ratios": cfg.get("data.split_ratios"),
        "data_seed": cfg.get("data.seed"),
        "volumes": cfg.get("data.volumes", None),
        "sampling": cfg.get("data.sampling", None),
        "shuffle_buffer_size": cfg.get("data.shuffle_buffer_size", None),
        "offline_snapshots": cfg.get("data.offline_snapshots", None),
        "cache_dir": cfg.get("data.cache_dir", None),
        "available_tasks": [task.task_id for task in data.tasks],
        "tasks": [
            {
                **task.as_dict(),
                "split_counts": data.splits[task.task_id].counts(),
                "split_provenance": data.splits[task.task_id].provenance,
            }
            for task in data.tasks
        ],
        "blocked_tasks": blocked_table,
        "unpinned_tasks": data.unpinned,
        "snapshots": snapshots_summary,
        "leakage_audit": data.leakage_report,
    }
    path = tracker.write_json("dataset_report.json", report)

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"dataset report: {path}")
    for task in data.tasks:
        counts = data.splits[task.task_id].counts()
        print(f"  {task.task_id:10s} {task.task_type.value:22s} {counts}  license={task.license}")
    if blocked_table:
        print("\n  BLOCKED (reported, not guessed at):")
        for task_id, entry in blocked_table.items():
            print(f"    {task_id:10s} {entry['code']:24s} {entry['decision']}  {entry['reason']}")
    else:
        print("\n  blocked tasks: none")
    if snapshots_summary:
        print("\n  snapshots:")
        for entry in snapshots_summary:
            print(
                f"    {entry['task']:10s} {entry['upstream_split']:12s} source={entry['source']:8s} "
                f"rows={entry['rows_consumed']:6d} examples={entry['n_examples']:5d} "
                f"exhausted={entry['exhausted']} over_len={entry['over_length_fraction']:.3f} "
                f"key={entry['key']}"
            )
        print(f"    over-length metric: {snapshots_summary[0]['over_length_metric']}")
    audit = data.leakage_report
    print(f"\nleakage audit: {audit['pairs_checked']} split pairs checked, clean={audit['clean']}")
    if audit["overlaps"]:
        print(f"  non-critical overlaps: {audit['overlaps']}")
    return 0 if audit["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

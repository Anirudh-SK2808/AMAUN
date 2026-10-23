#!/usr/bin/env python
"""Run the full AMAUN decision study (docs/l4_next_run_prompt.md): one resumable run, phases 0-7.

    .venv/bin/python scripts/run_full_study.py --config configs/experiments/l4_full_study.yaml \
        --run-mode local_smoke [--resume]

``--resume`` continues from decision_study/checkpoint/state.json after a restart (the config hash
must match).  ``--prepare-only`` builds and audits every split (including the cross-task content
audit and the duplicate tasks' provenance) and exits; it needs no GPU, and is how the dataset
revisions are verified on the run host before launch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.continual.full_study import FullStudy
from amaun.data.splits import SplitName
from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.models.backbone import load_backbone
from amaun.utils.paths import smoke_label
from amaun.utils.seed import configure_determinism
from amaun.utils.tracking import ExperimentTracker


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--prepare-only", action="store_true",
                        help="build, cap and audit the splits, print their sizes, and exit")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run recorded in decision_study/checkpoint/state.json")
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    # Before any CUDA context exists: cuBLAS reads its workspace setting at initialisation.
    determinism = configure_determinism(bool(cfg.get("experiment.deterministic", False)))
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    data = prepare_data(cfg)
    split_sizes = {task_id: splits.counts() for task_id, splits in data.splits.items()}
    data_payload = {
        "split_sizes": split_sizes,
        "blocked": data.blocked,
        "leakage_audit": data.leakage_report,
        "provenance": {task_id: splits.provenance for task_id, splits in data.splits.items()},
        "determinism": determinism,
    }
    tracker.write_json("decision_study/data.json", data_payload)
    print(smoke_label(cfg.run_mode))
    for task_id, counts in split_sizes.items():
        print(f"  {task_id:<12} " + "  ".join(f"{name.value}={counts[name.value]}" for name in SplitName))
    if data.blocked:
        print(f"ERROR: blocked tasks {data.blocked}; the study needs every configured task.")
        return 2
    if args.prepare_only:
        print(f"\nsplits and audits written to {tracker.paths.root / 'decision_study' / 'data.json'}")
        return 0

    backbone = load_backbone(cfg)
    study = FullStudy(cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits, tracker=tracker,
                      resume=args.resume, data_payload=data_payload)
    summary = study.run()
    if summary.get("aborted"):
        print(f"\nABORTED: {summary['aborted']}")
        return 3
    print("\npolicy comparison (descriptive; one seed):")
    print(json.dumps((summary.get("P5") or {}).get("policy_comparison"), indent=2, default=str))
    print(f"total wall time: {summary['total_wall_seconds'] / 3600:.2f} h (budget {summary['time_budget_hours']} h)")
    if summary["skipped"] or summary["failures"]:
        print(f"skipped: {len(summary['skipped'])}  failures: {len(summary['failures'])} "
              f"(see decision_study/summary.json)")
    print(f"results: {tracker.paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

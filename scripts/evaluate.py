#!/usr/bin/env python
"""Evaluate stored adapters on the TEST splits and emit machine-readable results.

PHASE 18.  This script is the only place outside the runner that reads TEST, and it does so
through an evaluator guard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.adapters.lora import LoRASpec, ensure_lora, load_delta_into, zero_delta_
from amaun.adapters.repository import AdapterRepository
from amaun.data.splits import evaluator_guard
from amaun.evaluation.evaluator import Evaluator
from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.models.backbone import load_backbone
from amaun.models.generation import GenerationConfig
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--repository", default=None)
    parser.add_argument("--adapters", nargs="*", default=None, help="adapter ids (default: all)")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--include-zero-shot", action="store_true",
                        help="also evaluate the frozen backbone with no adapter (the b_j reference)")
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    data = prepare_data(cfg, task_ids=args.tasks)
    backbone = load_backbone(cfg)
    peft_model = ensure_lora(backbone.model, LoRASpec.from_config(cfg.section("lora")))

    repository = AdapterRepository.load(
        Path(args.repository) if args.repository else tracker.paths.adapters,
        backbone_fingerprint=backbone.fingerprint,
        journal_path=tracker.paths.repository_events,
    )
    evaluator = Evaluator(
        backbone, GenerationConfig.from_config(cfg.section("generation")),
        max_examples=int(cfg.get("evaluation.max_test_examples", 8)),
        batch_size=int(cfg.get("training.batch_size")),
    )

    adapter_ids = args.adapters or repository.ids()
    rows = []
    if args.include_zero_shot:
        zero_delta_(peft_model)
        for task in data.tasks:
            result = evaluator.evaluate(
                task, evaluator_guard(data.splits[task.task_id]), model=peft_model, adapter_id=None
            )
            rows.append({"adapter_id": "<zero-shot>", **result.as_dict()})

    for adapter_id in adapter_ids:
        load_delta_into(peft_model, repository.retrieve(adapter_id).delta)
        for task in data.tasks:
            result = evaluator.evaluate(
                task, evaluator_guard(data.splits[task.task_id]),
                model=peft_model, adapter_id=adapter_id,
            )
            rows.append({"adapter_id": adapter_id, **result.as_dict()})
            tracker.log_metric(phase="standalone_evaluation", **{
                "adapter_id": adapter_id, "task_id": task.task_id,
                "primary_score": result.primary_score, **result.metrics,
            })

    tracker.write_json("evaluation_matrix.json", {
        "generation_config": GenerationConfig.from_config(cfg.section("generation")).as_dict(),
        "rows": rows,
    })

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"{'adapter':28s} {'task':10s} {'primary':>9s}  metric")
    for row in rows:
        print(f"{row['adapter_id']:28s} {row['task_id']:10s} {row['primary_score']:9.4f}  "
              f"{row['primary_score_name']}")
    print(f"\nresults: {tracker.paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Train the initial adapters INDEPENDENTLY from the same frozen theta_0.

PHASE 8 of the execution order.  Implements the fan-out of brief S1:

    Base --> Adapter1
         --> Adapter2
         ...

explicitly NOT the chain ``Base -> A1 -> A2 -> ...``.  Independence is guaranteed by a fresh
PEFT injection, a fresh optimiser, and a per-adapter derived seed, and is *verified* here by
recording the backbone fingerprint before and after every adapter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.adapters.lora import LoRASpec
from amaun.adapters.metadata import AdapterMetadata, AdapterOrigin
from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.data.splits import trainer_guard
from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.models.backbone import backbone_fingerprint, load_backbone
from amaun.training.budget import ArmBudget
from amaun.training.lora_trainer import TrainingConfig, train_adapter
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--tasks", nargs="*", default=None, help="subset of task ids (default: all)")
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    data = prepare_data(cfg, task_ids=args.tasks)

    backbone = load_backbone(cfg)
    reference_fingerprint = backbone_fingerprint(backbone.model)
    lora_spec = LoRASpec.from_config(cfg.section("lora"))
    train_cfg = TrainingConfig.from_config(cfg.section("training"))
    repository = AdapterRepository(
        tracker.paths.adapters, backbone_fingerprint=backbone.fingerprint,
        journal_path=tracker.paths.repository_events,
    )
    budget = ArmBudget(arm_id="bootstrap_all")

    rows = []
    for task in data.tasks:
        adapter_id = f"seed-{task.task_id}"
        before = backbone_fingerprint(backbone.model)
        delta, record = train_adapter(
            backbone=backbone, task=task, guard=trainer_guard(data.splits[task.task_id]),
            lora_spec=lora_spec, train_cfg=train_cfg, resolved=cfg,
            adapter_id=adapter_id, budget=budget,
        )
        after = backbone_fingerprint(backbone.model)
        entry = AdapterEntry(
            delta=delta,
            metadata=AdapterMetadata(
                adapter_id=adapter_id, task_name=task.task_name, task_type=task.task_type.value,
                dataset=task.hf_id or f"synthetic:{task.task_id}", dataset_revision=task.hf_revision,
                model_version=str(cfg.backbone.model_id), backbone_fingerprint=backbone.fingerprint,
                run_mode=cfg.run_mode.value, lora_rank=delta.rank,
                target_modules=list(lora_spec.target_modules), training_configuration=record.as_dict(),
                origin=AdapterOrigin.TRAINED.value, tasks_served=[task.task_id],
            ),
        )
        repository.register(entry, bootstrap=True, task_index=0)
        tracker.write_json(f"training/{adapter_id}.json", record.as_dict())
        rows.append({
            "adapter_id": adapter_id, "task_id": task.task_id, "steps": record.steps,
            "final_train_loss": record.final_train_loss, "wall_seconds": record.wall_seconds,
            "delta_hash": delta.content_hash()[:16],
            "backbone_unchanged": before == after == reference_fingerprint,
            "seed": record.seed,
        })

    tracker.write_json("bootstrap_summary.json", {
        "reference_backbone_fingerprint": reference_fingerprint,
        "independence_protocol": (
            "fresh PEFT injection + fresh optimiser + seed derived from adapter_id, applied "
            "immediately before each adapter's training; no state crosses adapters"
        ),
        "adapters": rows,
        "budget": budget.as_dict(),
    })
    repository.snapshot(tracker.paths.repository_snapshots / "bootstrap.json", label="bootstrap")

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"{'adapter':22s} {'task':10s} {'steps':>6s} {'loss':>9s} {'sec':>7s}  {'hash':16s} frozen")
    for row in rows:
        loss = f"{row['final_train_loss']:.4f}" if row["final_train_loss"] is not None else "n/a"
        print(f"{row['adapter_id']:22s} {row['task_id']:10s} {row['steps']:6d} {loss:>9s} "
              f"{row['wall_seconds']:7.2f}  {row['delta_hash']:16s} {row['backbone_unchanged']}")
    print(f"\nrepository: {len(repository)} adapters, {repository.memory_bytes()} bytes")
    print(f"all backbones unchanged: {all(r['backbone_unchanged'] for r in rows)}")
    return 0 if all(row["backbone_unchanged"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

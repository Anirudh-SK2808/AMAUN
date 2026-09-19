#!/usr/bin/env python
"""Train ONE LoRA adapter on ONE task, from the frozen theta_0.

PHASE 7 of the execution order (single LoRA smoke test).  Verifies the frozen-backbone
invariant before and after training and reports the fingerprint both times.
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
    parser.add_argument("--task", required=True, help="task id from the curriculum")
    parser.add_argument("--adapter-id", default=None)
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    data = prepare_data(cfg, task_ids=[args.task])
    task = data.tasks[0]

    backbone = load_backbone(cfg)
    fingerprint_before = backbone_fingerprint(backbone.model)

    adapter_id = args.adapter_id or f"{task.task_id}-s{cfg.seed}"
    budget = ArmBudget(arm_id=f"single:{adapter_id}")
    delta, record = train_adapter(
        backbone=backbone, task=task, guard=trainer_guard(data.splits[task.task_id]),
        lora_spec=LoRASpec.from_config(cfg.section("lora")),
        train_cfg=TrainingConfig.from_config(cfg.section("training")),
        resolved=cfg, adapter_id=adapter_id, budget=budget,
    )
    fingerprint_after = backbone_fingerprint(backbone.model)

    repository = AdapterRepository(
        tracker.paths.adapters, backbone_fingerprint=backbone.fingerprint,
        journal_path=tracker.paths.repository_events,
    )
    entry = AdapterEntry(
        delta=delta,
        metadata=AdapterMetadata(
            adapter_id=adapter_id, task_name=task.task_name, task_type=task.task_type.value,
            dataset=task.hf_id or f"synthetic:{task.task_id}", dataset_revision=task.hf_revision,
            model_version=str(cfg.backbone.model_id), backbone_fingerprint=backbone.fingerprint,
            run_mode=cfg.run_mode.value, lora_rank=delta.rank,
            target_modules=list(cfg.get("lora.target_modules")),
            training_configuration=record.as_dict(), origin=AdapterOrigin.TRAINED.value,
            tasks_served=[task.task_id],
        ),
    )
    repository.register(entry, bootstrap=True, task_index=0)
    tracker.write_json(f"training/{adapter_id}.json", record.as_dict())

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"adapter            {adapter_id}")
    print(f"steps              {record.steps}")
    print(f"final train loss   {record.final_train_loss}")
    print(f"trainable params   {record.trainable_parameters}")
    print(f"wall seconds       {record.wall_seconds:.2f}")
    print(f"delta layers/rank  {len(delta)} / {delta.rank}")
    print(f"backbone frozen    {fingerprint_before == fingerprint_after} "
          f"({fingerprint_before[:12]} -> {fingerprint_after[:12]})")
    print(f"saved to           {entry.path}")
    return 0 if fingerprint_before == fingerprint_after else 1


if __name__ == "__main__":
    raise SystemExit(main())

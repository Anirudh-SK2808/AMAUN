#!/usr/bin/env python
"""Run the federated component: multiple clients, local adapters, compatibility-gated aggregation.

PHASE 16 of the execution order.  TASK, CLIENT, ADAPTER and ROUND are independent axes; the
partition scheme is required config and has no default (docs/decision_register.md D-011).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.adapters.lora import LoRASpec, ensure_lora
from amaun.adapters.metadata import AdapterMetadata, AdapterOrigin
from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.data.splits import SplitName, trainer_guard
from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.federated.client import partition_clients, select_participants
from amaun.federated.compatibility import ClientUpdate
from amaun.federated.server import FederatedServer
from amaun.models.backbone import backbone_fingerprint, load_backbone
from amaun.training.budget import ArmBudget
from amaun.training.lora_trainer import TrainingConfig, train_adapter
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--naive-aggregation", action="store_true",
                        help="F2 control: aggregate ALL adapters regardless of compatibility")
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    data = prepare_data(cfg)
    backbone = load_backbone(cfg)
    reference_fingerprint = backbone_fingerprint(backbone.model)

    federated_cfg = dict(cfg.section("federated"))
    if args.naive_aggregation:
        # F2: a single compatibility bucket, i.e. no gate at all.
        federated_cfg["compatibility"] = {
            **(federated_cfg.get("compatibility") or {}),
            "require_same_rank": False,
            "require_same_target_modules": False,
            "require_same_backbone": False,
        }

    clients = partition_clients(data.splits, config=federated_cfg, seed=cfg.seed)
    lora_spec = LoRASpec.from_config(cfg.section("lora"))
    train_cfg = TrainingConfig.from_config(cfg.section("training"))
    ensure_lora(backbone.model, lora_spec)
    server = FederatedServer(federated_cfg)
    budget = ArmBudget(arm_id="federated")

    # Conceptual full-model payload -- computed for the ratio, never transmitted.
    server.ledger.conceptual_full_model_bytes = sum(
        p.numel() * p.element_size() for p in backbone.model.parameters()
    )

    round_reports = []
    for round_index in range(args.rounds):
        participants = select_participants(
            clients, participation_rate=float(federated_cfg["participation_rate"]),
            round_index=round_index, seed=cfg.seed,
        )
        updates: list[ClientUpdate] = []
        for client in participants:
            for task_id in client.task_ids:
                task = next(t for t in data.tasks if t.task_id == task_id)
                adapter_id = f"{client.client_id}-{task_id}-r{round_index}"
                delta, record = train_adapter(
                    backbone=backbone, task=task, guard=trainer_guard(data.splits[task_id]),
                    lora_spec=lora_spec, train_cfg=train_cfg, resolved=cfg,
                    adapter_id=adapter_id, budget=budget,
                )
                entry = AdapterEntry(
                    delta=delta,
                    metadata=AdapterMetadata(
                        adapter_id=adapter_id, task_name=task.task_name,
                        task_type=task.task_type.value,
                        dataset=task.hf_id or f"synthetic:{task_id}",
                        dataset_revision=task.hf_revision,
                        model_version=str(cfg.backbone.model_id),
                        backbone_fingerprint=backbone.fingerprint,
                        run_mode=cfg.run_mode.value, lora_rank=delta.rank,
                        target_modules=list(lora_spec.target_modules),
                        origin=AdapterOrigin.TRAINED.value, tasks_served=[task_id],
                        training_configuration={"steps": record.steps, "client": client.client_id},
                    ),
                )
                updates.append(
                    ClientUpdate(
                        client_id=client.client_id, entry=entry,
                        n_examples=client.n_examples(task_id),   # n_c, measured
                        task_ids=[task_id], round_index=round_index,
                    )
                )

        results, report = server.run_round(round_index, updates, target_rank=lora_spec.r)
        tracker.write_json(f"federated_rounds/round_{round_index:03d}.json", report.as_dict())
        round_reports.append(report.as_dict())

    summary = {
        "partition": federated_cfg["partition"],
        "num_clients": federated_cfg["num_clients"],
        "aggregation": federated_cfg["aggregation"],
        "compatibility": federated_cfg["compatibility"],
        "naive_aggregation_control": args.naive_aggregation,
        "clients": [client.as_dict() for client in clients],
        "rounds": round_reports,
        "communication": server.ledger.as_dict(),
        "budget": budget.as_dict(),
        "backbone_unchanged": backbone_fingerprint(backbone.model) == reference_fingerprint,
        "axes_note": (
            "TASK, CLIENT, ADAPTER and ROUND are independent: a client may hold shards of "
            "several tasks and several adapters, and one client != one task (D-011)"
        ),
    }
    tracker.write_json("federated_summary.json", summary)

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"partition            {federated_cfg['partition']} over {len(clients)} clients")
    print(f"aggregation          {federated_cfg['aggregation']['method']}")
    print(f"compatibility gate   {'DISABLED (F2 control)' if args.naive_aggregation else 'enabled'}")
    for report in round_reports:
        print(f"  round {report['round_index']}: {len(report['participating_clients'])} clients, "
              f"{len(report['groups'])} groups, {report['payload_bytes_total']} bytes, "
              f"{report['n_incompatible_singletons']} singleton(s)")
    ledger = server.ledger.as_dict()
    print(f"total LoRA traffic   {ledger['total_bytes']} bytes over "
          f"{ledger['n_transmissions']} transmissions")
    print(f"full-model counterfactual {ledger['counterfactual_full_model_bytes']} bytes "
          f"(ratio {ledger['lora_to_full_ratio']:.6f}) -- the backbone is never transmitted")
    if cfg.run_mode.value == "dummy":
        print("  NOTE: on the dummy tier LoRA is ~19% of a deliberately tiny backbone, so this")
        print("        ratio is a wiring check, not an efficiency result.")
    print(f"backbone unchanged   {summary['backbone_unchanged']}")
    return 0 if summary["backbone_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

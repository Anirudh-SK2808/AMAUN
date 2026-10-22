#!/usr/bin/env python
"""Bootstrap the AMAUN utility network on a decision-development task stream.

PHASES 10-12 of the execution order.  Generates supervision by actually performing candidate
integrations (idea.md S9), fits the normaliser on that split only, trains ``g_omega``, and
selects ``tau`` -- all without touching any TEST split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.adapters.lora import LoRASpec, ensure_lora
from amaun.adapters.repository import AdapterRepository
from amaun.experiment import add_common_arguments, prepare_data, resolve_from_args
from amaun.features.extractor import FeatureExtractor
from amaun.manager.bootstrap import bootstrap_utility_network
from amaun.models.backbone import load_backbone
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--repository", default=None, help="path to an existing adapter repository")
    parser.add_argument("--dev-tasks", nargs="+", required=True,
                        help="decision-development task ids (must be disjoint from the evaluated stream)")
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    data = prepare_data(cfg)
    backbone = load_backbone(cfg)

    repository_path = Path(args.repository) if args.repository else tracker.paths.adapters
    repository = AdapterRepository.load(
        repository_path, backbone_fingerprint=backbone.fingerprint,
        journal_path=tracker.paths.repository_events,
    )
    if repository.is_empty:
        print(f"ERROR: repository at {repository_path} is empty. Run train_all_adapters.py first.")
        return 2

    peft_model = ensure_lora(backbone.model, LoRASpec.from_config(cfg.section("lora")))
    dev_tasks = [task for task in data.tasks if task.task_id in set(args.dev_tasks)]
    if len(dev_tasks) != len(set(args.dev_tasks)):
        print(f"ERROR: unknown dev task(s); known: {[t.task_id for t in data.tasks]}")
        return 2

    result = bootstrap_utility_network(
        cfg=cfg, backbone=backbone, peft_model=peft_model, repository=repository,
        dev_tasks=dev_tasks, splits=data.splits,
        extractor=FeatureExtractor(cfg.section("features")), all_tasks=data.tasks,
    )

    network_path = tracker.paths.utility_network / "utility_network.pt"
    result.network.save(network_path, run_mode=cfg.run_mode.value, feature_names=result.feature_names)
    result.normalizer.save(tracker.paths.utility_network / "normalizer.json")
    tracker.write_json("utility_network/bootstrap_report.json", {
        **result.as_dict(),
        "dev_tasks": [task.task_id for task in dev_tasks],
        "leakage_note": "supervision read TRAIN/VALIDATION/SUPPORT only; no TEST access exists in this path",
    })

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"supervision examples  {len(result.supervision)} "
          f"({result.supervision.as_dict()['n_proxy']} proxy, "
          f"{result.supervision.as_dict()['n_full']} full)")
    print(f"pairs considered      {result.supervision.n_pairs_considered}")
    print(f"pairs evaluated       {result.supervision.n_pairs_evaluated}")
    print(f"beneficial fraction   {result.supervision.as_dict()['beneficial_fraction']}")
    print(f"feature dimension     {len(result.feature_names)}")
    print(f"final L_U             {result.training_record['final_loss']}")
    print(f"tau_u / tau_reuse     {result.thresholds.tau_u:.4f} / {result.thresholds.tau_reuse:.4f}")
    print(f"threshold source      {result.thresholds.source} ({result.thresholds.notes})")
    if result.supervision.proxy_bias:
        print(f"measured proxy bias   { {k: round(v, 5) for k, v in result.supervision.proxy_bias.items()} }")
    print(f"saved                 {network_path}")
    for note in result.notes:
        print(f"  NOTE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Run one continual-learning arm end to end.

PHASE 15 of the execution order.  Handles both repository-initialisation protocols of
docs/decision_register.md D-002 and every ablation toggle, and refuses to produce a
reportable result whose preconditions are unmet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.adapters.repository import AdapterRepository
from amaun.baselines.registry import build_strategy, unimplemented_report
from amaun.baselines.strategies import AmaunStrategy
from amaun.continual.runner import ContinualRunner
from amaun.evaluation.cl_metrics import compute as compute_cl_metrics
from amaun.evaluation.efficiency import format_growth_row
from amaun.evaluation.evaluator import Evaluator, zero_shot_baselines
from amaun.experiment import (
    add_common_arguments,
    check_reportable_preconditions,
    prepare_data,
    resolve_from_args,
)
from amaun.features.extractor import FeatureExtractor
from amaun.manager.bootstrap import bootstrap_utility_network
from amaun.models.backbone import load_backbone
from amaun.models.generation import GenerationConfig
from amaun.utils.paths import smoke_label
from amaun.utils.tracking import ExperimentTracker


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--strategy", default="amaun",
                        help="amaun | B2_separate_lora | B3_naive_average | B4_similarity_selection | ...")
    parser.add_argument("--dev-tasks", nargs="*", default=None,
                        help="decision-development tasks for utility/threshold fitting (D-005)")
    parser.add_argument("--skip-precondition-check", action="store_true",
                        help="report unmet preconditions without aborting (never for a reported run)")
    args = parser.parse_args()

    cfg = resolve_from_args(args)
    problems = check_reportable_preconditions(cfg, strict=not args.skip_precondition_check)
    tracker = ExperimentTracker(cfg, base=args.results_root).start()
    if problems:
        tracker.write_json("precondition_warnings.json", {"unmet": problems})

    init = cfg.get("curriculum.repository_init", None)
    if init is None:
        print("ERROR: curriculum.repository_init is unset and has NO DEFAULT.")
        print("       docs/decision_register.md D-002 is OPEN: bootstrap_six and empty_sequential")
        print("       answer different research questions and produce different headline tables.")
        return 2

    data = prepare_data(cfg)
    backbone = load_backbone(cfg)
    extractor = FeatureExtractor(cfg.section("features"))
    evaluator = Evaluator(
        backbone, GenerationConfig.from_config(cfg.section("generation")),
        max_examples=int(cfg.get("evaluation.max_test_examples", 8)),
        batch_size=int(cfg.get("training.batch_size")),
    )
    repository = AdapterRepository(
        tracker.paths.adapters, backbone_fingerprint=backbone.fingerprint,
        journal_path=tracker.paths.repository_events,
    )

    stream = cfg.get("curriculum.evaluated_stream")
    seed_tasks = cfg.get("curriculum.seed_tasks", []) or []
    if init == "empty_sequential" and seed_tasks:
        print("ERROR: empty_sequential requires curriculum.seed_tasks to be empty (idea.md S3: R_c^0 = {})")
        return 2

    # A placeholder stream that merely re-arrives the seed tasks cannot test H2/H3.
    overlap_warning = None
    if init == "bootstrap_six" and set(stream) & set(seed_tasks):
        overlap_warning = (
            f"the evaluated stream {stream} overlaps the seed tasks {seed_tasks}. Under "
            f"bootstrap_six the stream must contain NEW tasks, or REUSE/FUSE can never be the "
            f"CORRECT decision and H2/H3 are untestable (docs/experiment_protocol.md S3, "
            f"assumption A-14). This run is a WIRING CHECK, not a valid scientific configuration."
        )
        tracker.write_json("configuration_warnings.json", {"curriculum": overlap_warning})

    # Strategy construction, including the utility bootstrap where needed.
    bootstrap_payload = None
    if args.strategy == "amaun":
        # Seed the repository first: the utility bootstrap needs candidates to integrate with.
        temp_runner = ContinualRunner(
            cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits,
            repository=repository, strategy=_PlaceholderStrategy(), extractor=extractor,
            evaluator=evaluator, tracker=tracker, arm_id=f"{args.strategy}:seed",
        )
        if init == "bootstrap_six":
            temp_runner.bootstrap_repository(seed_tasks)

        dev_tasks = [t for t in data.tasks if t.task_id in set(args.dev_tasks or [])]
        if not dev_tasks:
            print("ERROR: --dev-tasks is required for the amaun strategy.")
            print("       Selecting tau on the evaluated stream would be selection on test data")
            print("       (docs/decision_register.md D-005).")
            return 2
        if repository.is_empty:
            print("ERROR: the utility network needs a non-empty repository to observe integrations.")
            print("       Use repository_init: bootstrap_six, or run train_all_adapters.py first.")
            return 2

        result = bootstrap_utility_network(
            cfg=cfg, backbone=backbone, peft_model=temp_runner.peft_model,
            repository=repository, dev_tasks=dev_tasks, splits=data.splits, extractor=extractor,
            all_tasks=data.tasks,
        )
        result.network.save(
            tracker.paths.utility_network / "utility_network.pt",
            run_mode=cfg.run_mode.value, feature_names=result.feature_names,
        )
        result.normalizer.save(tracker.paths.utility_network / "normalizer.json")
        if result.reuse_gate is not None:
            result.reuse_gate.save(tracker.paths.utility_network / "reuse_gate.json")
        bootstrap_payload = result.as_dict()
        tracker.write_json("utility_network/bootstrap_report.json", bootstrap_payload)

        strategy = AmaunStrategy(
            network=result.network, normalizer=result.normalizer, thresholds=result.thresholds,
            decision_order=str(cfg.get("manager.decision_order")),
            allowed_outcomes=cfg.get("manager.allowed_outcomes"),
            top_k=int(cfg.get("manager.retrieval.k")),
            reuse_gate=result.reuse_gate,
        )
    else:
        strategy = build_strategy(args.strategy, cfg.section("manager"))

    runner = ContinualRunner(
        cfg=cfg, backbone=backbone, tasks=data.tasks, splits=data.splits,
        repository=repository, strategy=strategy, extractor=extractor,
        evaluator=evaluator, tracker=tracker, arm_id=args.strategy,
    )
    if init == "bootstrap_six" and repository.is_empty:
        runner.bootstrap_repository(seed_tasks)

    # Zero-shot references b_j for forward transfer (assumption A-06), measured adapter-free.
    from amaun.adapters.lora import zero_delta_
    from amaun.data.splits import evaluator_guard

    zero_delta_(runner.peft_model)
    runner.matrix.zero_shot_baseline = zero_shot_baselines(
        evaluator, data.tasks, {t.task_id: evaluator_guard(data.splits[t.task_id]) for t in data.tasks},
        model=runner.peft_model,
    )

    outcome = runner.run(stream)
    metrics = compute_cl_metrics(outcome.matrix)

    # D-014: the growth row for the final task must carry performance alongside the count.
    growth = [
        format_growth_row(_snapshot_obj(outcome.events[-1].efficiency), metrics.average_performance)
    ] if outcome.events else []

    summary = {
        "arm": args.strategy,
        "repository_init": init,
        "run_mode": cfg.run_mode.value,
        "is_reportable": cfg.run_mode.is_reportable,
        "outcome_counts": outcome.outcome_counts(),
        "continual_metrics": metrics.as_dict(),
        "final_repository": {
            "n_adapters": len(outcome.repository),
            "archived": outcome.repository.archived_ids(),
            "tasks_served": outcome.repository.tasks_served(),
            "storage_bytes": outcome.repository.memory_bytes(),
            "total_parameters": outcome.repository.total_parameters(),
        },
        "budget": outcome.budget.as_dict(),
        "efficiency_trajectory": [event.efficiency for event in outcome.events],
        "growth_rows": growth,
        "utility_bootstrap": bootstrap_payload,
        "unimplemented_baselines": unimplemented_report(),
        "precondition_warnings": problems,
        "configuration_warning": overlap_warning,
        "leakage_audit": data.leakage_report,
    }
    tracker.write_json("summary.json", summary)
    tracker.write_json("budget_ledger.json", outcome.budget.as_dict())

    print(f"\n{smoke_label(cfg.run_mode)}")
    print(f"arm                  {args.strategy}  (repository_init={init})")
    print(f"decisions            {outcome.outcome_counts()}")
    print(f"average performance  {metrics.average_performance:.4f} ({outcome.matrix.score_name})")
    print(f"forgetting           {metrics.forgetting}")
    print(f"backward transfer    {metrics.backward_transfer}")
    print(f"forward transfer     {metrics.forward_transfer}")
    print(f"final adapters       {len(outcome.repository)} (archived {len(outcome.repository.archived_ids())})")
    print(f"tasks per adapter    "
          f"{outcome.events[-1].efficiency['tasks_served_per_adapter']:.3f}")
    print(f"stored parameters    {outcome.repository.total_parameters()}")
    print(f"optimizer steps      {outcome.budget.optimizer_steps}")
    for note in metrics.notes:
        print(f"  NOTE: {note}")
    if overlap_warning:
        print(f"\n  CONFIGURATION WARNING: {overlap_warning}")
    print(f"\nresults: {tracker.paths.root}")
    return 0


def _snapshot_obj(payload: dict):
    """Rebuild an EfficiencySnapshot-like object from its serialised form."""
    from amaun.evaluation.efficiency import EfficiencySnapshot

    known = {k: v for k, v in payload.items() if k in EfficiencySnapshot.__dataclass_fields__}
    return EfficiencySnapshot(**known)


class _PlaceholderStrategy:
    """Used only to construct a runner for the pre-bootstrap seeding phase."""

    strategy_id = "placeholder"
    requires_utility_network = False

    def score(self, vectors, *, run_mode):
        return []

    def decide(self, **kwargs):
        raise RuntimeError("placeholder strategy must not make decisions")


if __name__ == "__main__":
    raise SystemExit(main())

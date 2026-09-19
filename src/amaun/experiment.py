"""Shared experiment assembly: the one place a run is wired together.

Scripts stay thin wrappers around these functions so every entry point performs the *same*
preconditions, split construction, and provenance capture.  A script that assembled its own
pipeline could quietly differ from the others -- for instance by building splits with a
different seed -- and that difference would be invisible in the results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from amaun.config import ResolvedConfig, RunMode, load_config, resolve
from amaun.data.leakage import audit
from amaun.data.providers import huggingface_pool, synthetic_pool
from amaun.data.registry import TaskSpec, load_curriculum
from amaun.data.splits import SplitRatios, TaskSplits, build_splits
from amaun.errors import PreconditionError, UnpinnedDatasetError
from amaun.utils.logging import get_logger, setup_logging
from amaun.utils.provenance import git_state
from amaun.utils.seed import set_all_seeds

log = get_logger("experiment")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="path to an experiment config YAML")
    parser.add_argument(
        "--run-mode", required=True, choices=[m.value for m in RunMode],
        help="REQUIRED, no default (idea.md S0.1): dummy | local_smoke | actual",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the first configured seed")
    parser.add_argument("--experiment-id", default=None, help="override the results directory name")
    parser.add_argument("--device", default=None, help="cpu | mps | cuda | auto")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--config-root", default="configs")
    return parser


def resolve_from_args(args: argparse.Namespace) -> ResolvedConfig:
    raw = load_config(args.config, config_root=args.config_root)
    cfg = resolve(
        raw, run_mode=args.run_mode, experiment_id=args.experiment_id,
        seed=args.seed, device=args.device,
    )
    setup_logging(args.log_level, cfg.run_mode)
    set_all_seeds(cfg.seed)
    return cfg


def check_reportable_preconditions(cfg: ResolvedConfig, *, strict: bool = True) -> list[str]:
    """Gate for `actual` runs (docs/experiment_protocol.md S1).

    Returns the list of unmet preconditions.  A reportable run must have none.
    """
    problems: list[str] = []
    if not cfg.run_mode.is_reportable:
        return problems

    seeds = cfg.get("experiment.seeds", [])
    if len(seeds) < 3:
        problems.append(
            f"experiment.seeds has {len(seeds)} entry/entries; a reported arm needs >=3 "
            f"(docs/decision_register.md D-012)"
        )
    state = git_state()
    if state["dirty"] and not cfg.get("experiment.allow_dirty_git", False):
        problems.append(
            f"git tree is dirty ({len(state['dirty_files'])} file(s)) and "
            f"experiment.allow_dirty_git is false; a reported run must be reproducible from a commit"
        )
    if cfg.get("curriculum.repository_init", None) is None:
        problems.append(
            "curriculum.repository_init is unset (docs/decision_register.md D-002 is OPEN); it has "
            "no default because the choice changes what the headline table means"
        )
    if strict and problems:
        raise PreconditionError(
            "this run is marked reportable (run_mode=actual) but its preconditions are unmet:\n  - "
            + "\n  - ".join(problems)
        )
    return problems


@dataclass
class PreparedData:
    tasks: list[TaskSpec]
    splits: dict[str, TaskSplits]
    leakage_report: dict[str, Any]
    unpinned: list[str]


def prepare_data(cfg: ResolvedConfig, *, task_ids: Sequence[str] | None = None) -> PreparedData:
    """Build four disjoint splits per task and audit the whole corpus for leakage."""
    tasks = load_curriculum(cfg.section("datasets"))
    if task_ids is not None:
        wanted = set(task_ids)
        tasks = [task for task in tasks if task.task_id in wanted]
    data_cfg = cfg.section("data")
    provider = str(data_cfg["provider"])
    ratios = SplitRatios.from_config(data_cfg["split_ratios"])
    data_seed = int(data_cfg["seed"])

    splits: dict[str, TaskSplits] = {}
    unpinned: list[str] = []
    for task in tasks:
        if provider == "synthetic":
            pool = synthetic_pool(task, int(data_cfg["synthetic_pool_size"]), seed=data_seed)
            test = synthetic_pool(
                task, int(data_cfg["synthetic_test_size"]), seed=data_seed, offset=1_000_000
            )
        elif provider == "huggingface":
            try:
                task.require_pinned()
            except UnpinnedDatasetError as exc:
                unpinned.append(task.task_id)
                log.error("skipping task '%s': %s", task.task_id, exc)
                continue
            volumes = (data_cfg.get("volumes") or {}).get(task.task_id) or {}
            pool = huggingface_pool(
                task, int(volumes.get("train_pool", 1000)), split="train",
                cache_dir=data_cfg.get("cache_dir"),
            )
            test = huggingface_pool(
                task, int(volumes.get("test", 200)), split=volumes.get("test_split", "validation"),
                cache_dir=data_cfg.get("cache_dir"),
            )
        else:
            raise PreconditionError(f"unknown data.provider '{provider}'")

        task_splits = build_splits(
            task.task_id, training_pool=pool, test_pool=test, seed=data_seed, ratios=ratios,
            provenance={"provider": provider, "hf_id": task.hf_id, "hf_revision": task.hf_revision},
        )
        cap = data_cfg.get("smoke_max_examples_per_split")
        if cap and cfg.run_mode is not RunMode.ACTUAL:
            # idea.md S0.2.1: one config key, applied uniformly to all four splits.
            task_splits = task_splits.subsample(int(cap), seed=data_seed)
            task_splits.assert_disjoint()
        splits[task.task_id] = task_splits
        log.info("task %-10s splits %s", task.task_id, task_splits.counts())

    if not splits:
        raise PreconditionError(
            f"no task produced usable splits (unpinned: {unpinned}). Resolve "
            f"docs/decision_register.md D-006, or set data.provider: synthetic."
        )

    report = audit(splits)
    return PreparedData(
        tasks=[task for task in tasks if task.task_id in splits],
        splits=splits,
        leakage_report=report.as_dict(),
        unpinned=unpinned,
    )

"""Shared experiment assembly: the one place a run is wired together.

Scripts stay thin wrappers around these functions so every entry point performs the *same*
preconditions, split construction, and provenance capture.  A script that assembled its own
pipeline could quietly differ from the others -- for instance by building splits with a
different seed -- and that difference would be invisible in the results.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Mapping, Sequence

from amaun.config import ResolvedConfig, RunMode, load_config, resolve
from amaun.data.leakage import audit
from amaun.data.providers import huggingface_pool, synthetic_pool
from amaun.data.registry import TaskSpec, load_curriculum
from amaun.data.row_adapters import resolve as resolve_row_adapter
from amaun.data.splits import SplitRatios, TaskSplits, build_splits
from amaun.errors import (
    ConfigError,
    DataError,
    PreconditionError,
    UnpinnedDatasetError,
    UnresolvedRowAdapterError,
)
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
    # FR-7.1: task_id -> reason code, one of "unpinned_dataset" (D-006) or
    # "unresolved_row_adapter" (D-015).  Nothing else is skippable: any other failure propagates,
    # because silent degradation on a transient error would be indistinguishable from a real block.
    blocked: dict[str, str] = dataclass_field(default_factory=dict)
    # Per-task provider load reports (training pool first, TEST pool second).
    load_reports: dict[str, list[Any]] = dataclass_field(default_factory=dict)


BLOCK_REASONS: dict[str, dict[str, str]] = {
    "unpinned_dataset": {
        "decision": "D-006",
        "reason": "the dataset has no pinned hf_id/hf_revision; guessing one would silently "
                  "change what this adapter represents",
    },
    "unresolved_row_adapter": {
        "decision": "D-015",
        "reason": "no row->example rule is configured for this task's upstream schema; guessing "
                  "one would silently change what this adapter learns",
    },
}

# The upstream split the training pool is drawn from.  Pre-existing behaviour, kept: TEST comes
# from a separately named split (or an offset window of this one).
_TRAINING_POOL_SPLIT = "train"


def _task_volumes(cfg_volumes: Mapping[str, Any] | None, task_id: str) -> dict[str, Any]:
    """Read ``data.volumes.<task>`` with every required key mandatory (FR-3.4).

    The pre-change code substituted ``1000`` / ``200`` / ``"validation"`` here.  Those were three
    experiment-affecting defaults living in code, which docs/system_specification.md M0 forbids:
    how many examples a task is trained on, and which upstream split becomes TEST, must be visible
    in ``configs/`` and in ``config_used.json``.
    """
    volumes = (cfg_volumes or {}).get(task_id)
    if not volumes:
        raise ConfigError(
            f"data.volumes.{task_id} is required under data.provider: huggingface and has no "
            f"default (docs/system_specification.md M0). Declare train_pool, test, and test_split "
            f"for every task in configs/."
        )
    missing = [key for key in ("train_pool", "test", "test_split") if volumes.get(key) is None]
    if missing:
        raise ConfigError(
            f"data.volumes.{task_id} is missing required key(s) {missing}; AMAUN provides no "
            f"default for an experiment-affecting volume or TEST source"
        )
    train_pool = int(volumes["train_pool"])
    test = int(volumes["test"])
    test_split = str(volumes["test_split"])
    test_offset = volumes.get("test_offset")
    if test_split == _TRAINING_POOL_SPLIT and test_offset is None:
        raise ConfigError(
            f"data.volumes.{task_id}.test_offset is required because test_split == "
            f"'{_TRAINING_POOL_SPLIT}', i.e. the TEST pool is carved from the *same* upstream split "
            f"as the training pool. Without an offset both loads stream from row 0 and TEST "
            f"becomes a prefix of TRAIN, violating invariant I2.1 "
            f"(specs/local-smoke-data-bootstrap/design.md S4)."
        )
    return {
        "train_pool": train_pool,
        "test": test,
        "test_split": test_split,
        "test_offset": int(test_offset) if test_offset is not None else 0,
    }


def _sampling_settings(data_cfg: Mapping[str, Any], volumes: Mapping[str, Any], task_id: str
                       ) -> tuple[str, int | None]:
    """Read ``data.sampling`` / ``data.shuffle_buffer_size`` (both mandatory, D-016)."""
    sampling = data_cfg.get("sampling")
    if sampling is None:
        raise ConfigError(
            "data.sampling is required under data.provider: huggingface (head_of_stream | "
            "shuffled_buffer) and has no default: which rows a volume cap selects is an "
            "experimental choice (docs/decision_register.md D-016)"
        )
    sampling = str(sampling)
    buffer_size = data_cfg.get("shuffle_buffer_size")
    if sampling == "shuffled_buffer":
        if buffer_size is None:
            raise ConfigError(
                "data.shuffle_buffer_size is required when data.sampling == shuffled_buffer and "
                "has no default; the buffer size determines how near-random the sample is "
                "(D-016 condition 1)"
            )
        buffer_size = int(buffer_size)
        needed = max(int(volumes["train_pool"]), int(volumes["test"]))
        if buffer_size < needed:
            raise ConfigError(
                f"data.shuffle_buffer_size={buffer_size} is smaller than max(train_pool, test)="
                f"{needed} for task '{task_id}'. A shuffle window narrower than the sample is "
                f"indistinguishable from head_of_stream and would misreport the sampling policy "
                f"(FR-3.7 / EC-14)."
            )
        return sampling, buffer_size
    return sampling, None


def _load_pool(task: TaskSpec, size: int, *, split: str, skip: int, adapter: Any,
               data_cfg: Mapping[str, Any], data_seed: int, sampling: str,
               buffer_size: int | None) -> tuple[list[Any], Any]:
    """One provider load, with HF's own failures surfaced rather than masked as 'blocked'."""
    try:
        return huggingface_pool(
            task, size, split=split, row_adapter=adapter, data_seed=data_seed,
            sampling=sampling, shuffle_buffer_size=buffer_size,
            offline_snapshots=bool(data_cfg.get("offline_snapshots", False)),
            cache_dir=data_cfg.get("cache_dir"), skip=skip,
        )
    except ValueError as exc:                     # datasets raises ValueError for an unknown split
        raise DataError(
            f"task '{task.task_id}': upstream split '{split}' could not be loaded from "
            f"{task.hf_id} (revision {task.hf_revision}). Underlying error: {exc}"
        ) from exc
    except (DataError, ConfigError):
        raise
    except Exception as exc:
        # FR-7.2: a load failure always propagates -- it is never converted into a "blocked"
        # task, because that would make a transient network error indistinguishable from an open
        # decision.  When the run is offline we add EC-4's hint; otherwise the original error is
        # re-raised untouched rather than being re-typed.
        if os.environ.get("HF_HUB_OFFLINE"):
            raise DataError(
                f"task '{task.task_id}': loading {task.hf_id}:{split} failed with "
                f"{type(exc).__name__}: {exc}. HF_HUB_OFFLINE is set and no matching local "
                f"snapshot exists for these coordinates; run scripts/prepare_data.py online once "
                f"to populate data.cache_dir."
            ) from exc
        raise


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
    blocked: dict[str, str] = {}
    load_reports: dict[str, list[Any]] = {}
    for task in tasks:
        provenance: dict[str, Any] = {
            "provider": provider, "hf_id": task.hf_id, "hf_revision": task.hf_revision,
        }
        if provider == "synthetic":
            pool = synthetic_pool(task, int(data_cfg["synthetic_pool_size"]), seed=data_seed)
            test = synthetic_pool(
                task, int(data_cfg["synthetic_test_size"]), seed=data_seed, offset=1_000_000
            )
        elif provider == "huggingface":
            # Only these two conditions are skippable, and both are checked before any network
            # call so a blocked task costs nothing and cannot be confused with a load failure.
            try:
                task.require_pinned()
                adapter = resolve_row_adapter(task)
            except UnpinnedDatasetError as exc:
                blocked[task.task_id] = "unpinned_dataset"
                log.error("skipping task '%s' (unpinned_dataset): %s", task.task_id, exc)
                continue
            except UnresolvedRowAdapterError as exc:
                blocked[task.task_id] = "unresolved_row_adapter"
                log.error("skipping task '%s' (unresolved_row_adapter): %s", task.task_id, exc)
                continue

            volumes = _task_volumes(data_cfg.get("volumes"), task.task_id)
            sampling, buffer_size = _sampling_settings(data_cfg, volumes, task.task_id)
            pool, pool_report = _load_pool(
                task, volumes["train_pool"], split=_TRAINING_POOL_SPLIT, skip=0, adapter=adapter,
                data_cfg=data_cfg, data_seed=data_seed, sampling=sampling, buffer_size=buffer_size,
            )
            if (volumes["test_split"] == _TRAINING_POOL_SPLIT
                    and volumes["test_offset"] < pool_report.rows_consumed):
                # FR-4.4: checked against the MEASURED row count, because a row adapter may emit
                # several examples per row and the configured example cap is therefore not a bound
                # on rows read.
                raise DataError(
                    f"task '{task.task_id}': data.volumes.{task.task_id}.test_offset="
                    f"{volumes['test_offset']} is below the {pool_report.rows_consumed} row(s) the "
                    f"training pool actually consumed from upstream split "
                    f"'{_TRAINING_POOL_SPLIT}'. The two windows would overlap and TEST would leak "
                    f"into TRAIN (invariant I2.1)."
                )
            test, test_report = _load_pool(
                task, volumes["test"], split=volumes["test_split"],
                skip=volumes["test_offset"] if volumes["test_split"] == _TRAINING_POOL_SPLIT else 0,
                adapter=adapter, data_cfg=data_cfg, data_seed=data_seed, sampling=sampling,
                buffer_size=buffer_size,
            )
            load_reports[task.task_id] = [pool_report, test_report]
            provenance.update({
                "hf_config": task.hf_config,
                "hf_revision_resolved": task.hf_revision_resolved,
                "row_adapter": task.row_adapter,
                "row_adapter_options": dict(task.row_adapter_options),
                "volumes": volumes,
                "sampling": sampling,
                "shuffle_buffer_size": buffer_size,
                "loads": [pool_report.as_dict(), test_report.as_dict()],
            })
        else:
            raise PreconditionError(f"unknown data.provider '{provider}'")

        task_splits = build_splits(
            task.task_id, training_pool=pool, test_pool=test, seed=data_seed, ratios=ratios,
            provenance=provenance,
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
            f"no task produced usable splits (blocked: {blocked}). Resolve "
            f"docs/decision_register.md D-006 (dataset pinning) and D-015 (row->example "
            f"adaptation), or set data.provider: synthetic."
        )

    report = audit(splits)
    return PreparedData(
        tasks=[task for task in tasks if task.task_id in splits],
        splits=splits,
        leakage_report=report.as_dict(),
        unpinned=[t for t, code in blocked.items() if code == "unpinned_dataset"],
        blocked=blocked,
        load_reports=load_reports,
    )

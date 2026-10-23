"""The l4_full_study: one resumable run that tests the whole AMAUN decision method.

Specification: docs/l4_next_run_prompt.md.  Requirements, design and task list:
specs/l4_full_study/.  Decisions the prompt leaves open: docs/decision_register.md D-017 - D-024.

Phases (each unit of work is checkpointed, see :mod:`amaun.continual.study_state`):

0. **Preflight** -- run-1 split integrity (D-022), content leakage audit, worst-case memory with
   the most memory-hungry fusion setting (22.5 GB limit), a timing probe and the full-run time
   projection (fails above ``time_budget_hours``), BERTScore encoder loadable.
1. **Seeds** -- load seed-sum / seed-qa / seed-sentiment from run 1, train the others; zero-shot and
   standalone VALIDATION and TEST scores, computed once (R1).
2. **Fusion selection** -- the fusion grid on the run-1 pairs, on VALIDATION; one config chosen by
   :func:`amaun.continual.study_criteria.select_fusion_config`.
3. **Supervision** -- every ordered pair of seeds with the selected config; measured labels.
4. **Utility** -- gate, leave-one-new-task-out CV, tau from out-of-fold, refit; reuse gate and the
   fixed-rule threshold on the same measurements.  Everything is frozen here.
5. **Evaluation tasks** -- temp adapter, eligibility (VALIDATION), every policy's decision (SUPPORT
   only, logged), then TEST: CREATE, BACKBONE, REUSE x n, the measured FUSE set; policy tables
   without and with the adapter cap.
6. **Sequential stream** -- see :mod:`amaun.continual.sequential_stream`.
7. **Memory audit and repeatability** -- one identical fusion repeated; memory and score spread.

Leakage: phases 1-4 and every decision read TRAIN / VALIDATION / SUPPORT only.  TEST is read for
reporting in phase 1 and, per evaluation task, only after that task's decisions are logged.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.lora import MergedWeights, extract_delta, load_delta_into, zero_delta_
from amaun.adapters.metadata import AdapterMetadata, AdapterOrigin
from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.baselines.strategies import (
    AlwaysMergeStrategy,
    SimilarityConfig,
    SimilaritySelectionStrategy,
)
from amaun.config import ResolvedConfig, _deep_merge
from amaun.continual import study_criteria as criteria
from amaun.continual.data_integrity import load_reference, verify_reference_splits
from amaun.continual.decision_study import DecisionStudy, _Clock, worst_case_examples
from amaun.continual.study_state import StudyCheckpoint
from amaun.continual.time_projection import WorkItem, merge_costs, project, run1_unit_costs, summarise_items
from amaun.data.loaders import BatchPlan, GuardedLoader
from amaun.data.preprocess import IGNORE_INDEX, collate, encode_prompt, tokenize_example
from amaun.data.registry import TaskSpec
from amaun.data.splits import (
    SplitName,
    TaskSplits,
    evaluator_guard,
    manager_guard,
    trainer_guard,
    utility_guard,
)
from amaun.errors import PreconditionError
from amaun.evaluation.bertscore import BERTScoreConfig, BERTScorer
from amaun.features.base import FeatureVector
from amaun.features.extractor import probe_quality
from amaun.features.normalizer import FeatureNormalizer
from amaun.fusion.layerwise import FusionConfig
from amaun.fusion.optimize import run_fusion_kd
from amaun.manager.bootstrap import _loss_to_score, _validation_loss
from amaun.manager.decision import Thresholds
from amaun.manager.retrieval import score_candidates
from amaun.manager.reuse_gate import GateExample, ReuseGate, ReuseGateConfig
from amaun.manager.targets import SupervisionExample, SupervisionSet, measure_outcome_targets
from amaun.manager.utility_cv import FitGate, fit_with_cv
from amaun.manager.utility_network import UtilityNetwork
from amaun.models.backbone import assert_fingerprint_unchanged
from amaun.training.kd import KDConfig
from amaun.training.lora_trainer import TrainingConfig, train_adapter
from amaun.utils.logging import get_logger
from amaun.utils.paths import assert_write_allowed
from amaun.utils.resources import cleanup, memory_snapshot, stability_report

log = get_logger("continual.full_study")

FULL_POLICIES = ("amaun", "amaun_train_first", "probe_rule", "fixed_rule", "always_separate",
                 "always_merge", "reuse_only", "fuse_only")
PHASES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class FusionVariant:
    name: str
    reference: bool
    fusion: FusionConfig
    kd: KDConfig
    raw: dict[str, Any]


@dataclass
class FullStudyConfig:
    reference_run: str | None
    reuse_adapters_dir: str | None
    reuse_adapter_ids: list[str]
    reference_tasks: list[str]
    fusion_grid: list[FusionVariant]
    fusion_pair_adapters: list[str]
    label_examples: dict[str, int]
    eligibility_examples: dict[str, int]
    test_examples: dict[str, int]
    probe_examples: int
    fuse_top_k: int
    policies: list[str]
    caps: list[int | None]
    fit_gate: FitGate
    related_groups: list[list[str]]
    stream: dict[str, Any]
    memory_limit_bytes: int | None
    projection: dict[str, Any]
    equivalence: float
    forgetting: float
    eligibility_threshold: float
    bootstrap_resamples: int

    @classmethod
    def from_config(cls, cfg: ResolvedConfig, *, lora_learning_rate: float) -> "FullStudyConfig":
        section = cfg.section("full_study")
        grid = []
        for item in section["fusion_grid"]:
            fusion_raw = _deep_merge(cfg.section("fusion"), item.get("fusion") or {})
            kd_raw = _deep_merge(cfg.section("kd"), item.get("kd") or {})
            grid.append(FusionVariant(
                name=str(item["name"]), reference=bool(item.get("reference", False)),
                fusion=FusionConfig.from_config(fusion_raw, lora_learning_rate=lora_learning_rate),
                kd=KDConfig.from_config(kd_raw), raw={"fusion": fusion_raw, "kd": kd_raw},
            ))
        if len({v.name for v in grid}) != len(grid):
            raise PreconditionError("full_study.fusion_grid has duplicate names")
        policies = list(section["policies"])
        unknown = [p for p in policies if p not in FULL_POLICIES]
        if unknown:
            raise PreconditionError(f"unknown full_study policies {unknown}; known: {list(FULL_POLICIES)}")
        reuse = section.get("reuse_adapters") or {}
        limit = section.get("memory_limit_gb")
        rules = section.get("criteria") or {}
        return cls(
            reference_run=section.get("reference_run"),
            reuse_adapters_dir=reuse.get("dir"),
            reuse_adapter_ids=list(reuse.get("ids") or []),
            reference_tasks=list(section.get("reference_tasks") or []),
            fusion_grid=grid,
            fusion_pair_adapters=list(section["fusion_pair_adapters"]),
            label_examples={str(k): int(v) for k, v in section["label_examples"].items()},
            eligibility_examples={str(k): int(v) for k, v in section["eligibility_examples"].items()},
            test_examples={str(k): int(v) for k, v in section["test_examples"].items()},
            probe_examples=int(section["probe_examples"]),
            fuse_top_k=int(section["fuse_top_k"]),
            policies=policies,
            caps=[None if c is None else int(c) for c in section["caps"]],
            fit_gate=FitGate.from_config(section["utility_fit_gate"]),
            related_groups=[list(g) for g in section.get("related_groups") or []],
            stream=dict(section["stream"]),
            memory_limit_bytes=None if limit is None else int(float(limit) * 1e9),
            projection=dict(section.get("projection") or {}),
            equivalence=float(rules.get("equivalence", criteria.EQUIVALENCE)),
            forgetting=float(rules.get("forgetting", criteria.FORGETTING)),
            eligibility_threshold=float(rules.get("eligibility", criteria.ELIGIBILITY)),
            bootstrap_resamples=int(rules.get("bootstrap_resamples", 1000)),
        )

    def variant(self, name: str) -> FusionVariant:
        return next(v for v in self.fusion_grid if v.name == name)


class _ResumableClock(_Clock):
    """The run-1 budget clock, counting the wall time spent before a resume."""

    def __init__(self, hours: float, *, elapsed_before: float) -> None:
        super().__init__(hours)
        self.elapsed_before = elapsed_before

    @property
    def elapsed(self) -> float:
        return self.elapsed_before + (time.perf_counter() - self.start)


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", label)[:180]


# --------------------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------------------


class FullStudy(DecisionStudy):
    def __init__(self, *, cfg: ResolvedConfig, backbone: Any, tasks: Sequence[TaskSpec],
                 splits: Mapping[str, TaskSplits], tracker: Any, resume: bool = False,
                 data_payload: Mapping[str, Any] | None = None) -> None:
        super().__init__(cfg=cfg, backbone=backbone, tasks=tasks, splits=splits, tracker=tracker)
        self.full = FullStudyConfig.from_config(cfg, lora_learning_rate=self.runner.train_cfg.learning_rate)
        self.data_payload = dict(data_payload or {})
        self.ckpt = StudyCheckpoint(tracker.paths.root / "decision_study" / "checkpoint" / "state.json",
                                    config_hash=cfg.config_hash, experiment_id=cfg.experiment_id, resume=resume)
        self.clock = _ResumableClock(self.study.time_budget_hours, elapsed_before=self.ckpt.elapsed_before)
        self.predictions_dir = tracker.paths.root / "decision_study" / "predictions"
        self._hash_memo: dict[int, tuple[AdapterDelta, str]] = {}
        self.temp: dict[str, dict[str, Any]] = {}            # task -> {"delta", "entry", "record"}
        self.seed_by_task: dict[str, str] = {a.task_id: a.adapter_id for a in self.study.initial_adapters}
        self.network = self.normalizer = self.thresholds = None
        self.reuse_gate = None
        self.selected: FusionVariant | None = None

        #: D-026: every generating evaluation runs with the adapter merged into the base weights.
        self.merger = MergedWeights(self.peft_model) if cfg.get("evaluation.merged_weights", False) else None

        scorer_cfg = cfg.get("evaluation.bertscore", None)
        self.bertscorer = (BERTScorer(BERTScoreConfig.from_config(scorer_cfg), device=backbone.device)
                           if scorer_cfg else None)
        if self.bertscorer is not None:
            for evaluator in (self.final_evaluator, self.check_evaluator):
                evaluator.scorers["bertscore_f1"] = self.bertscorer

        if self.ckpt.resumed:
            self.repository = AdapterRepository.load(
                tracker.paths.adapters, backbone_fingerprint=backbone.fingerprint,
                journal_path=tracker.paths.repository_events)
            self.runner.repository = self.repository

    # -- writing ------------------------------------------------------------------
    def _elapsed(self) -> float:
        return self.clock.elapsed - self.ckpt.elapsed_before

    def _unit(self, key: str, compute) -> Any:
        """Return a finished unit's stored result, or compute, store and return it."""
        if self.ckpt.has_unit(key):
            return self.ckpt.unit(key)
        result = compute()
        self.ckpt.save_unit(key, result, elapsed=self._elapsed())
        return result

    def _write_predictions(self, label: str, rows: Sequence[Mapping[str, Any]]) -> str:
        path = self.predictions_dir / f"{_safe(label)}.jsonl"
        assert_write_allowed(path, self.cfg.run_mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return str(path.relative_to(self.tracker.paths.root))

    # -- measured primitives ------------------------------------------------------
    def _delta_key(self, delta: AdapterDelta | None) -> str:
        if delta is None:
            return "backbone"
        memo = self._hash_memo.get(id(delta))
        if memo is None or memo[0] is not delta:
            memo = (delta, delta.content_hash())
            self._hash_memo[id(delta)] = memo
        return memo[1]

    def _examples(self, kind: str, task_id: str) -> int:
        table = {"label": self.full.label_examples, "eligibility": self.full.eligibility_examples,
                 "test": self.full.test_examples}[kind]
        if task_id in table:
            return table[task_id]
        task = self.tasks[task_id]
        if task.duplicate_of and task.duplicate_of in table:
            return table[task.duplicate_of]
        raise PreconditionError(f"full_study.{kind}_examples has no entry for task '{task_id}'")

    def _measure(self, delta: AdapterDelta | None, task_id: str, *, split: SplitName, n: int,
                 label: str) -> dict[str, Any]:
        """Generate and score ``n`` examples of ``split`` with ``delta`` installed (None = backbone).

        Cached by (adapter content, task, split, n) in the checkpoint: a frozen adapter's greedy
        output cannot change, so every reuse of a measurement is exact (R1, R3).
        """
        key = f"{self._delta_key(delta)}|{task_id}|{split.value}|{n}"
        cache = self.ckpt.cache("evaluations")
        if key in cache:
            return cache[key]
        task = self.tasks[task_id]
        guard = {SplitName.TEST: evaluator_guard, SplitName.SUPPORT: manager_guard}.get(
            split, utility_guard)(self.splits[task_id])
        self.monitor.record(f"eval:{label}:before")
        result = self._evaluate(delta, task, guard, split=split, n=n)
        self.monitor.record(f"eval:{label}:after")
        primary = task.primary_score
        record = {
            "task_id": task_id, "adapter_id": result.adapter_id, "split": split.value, "label": label,
            "primary_score": result.primary_score, "primary_score_name": result.primary_score_name,
            "metrics": result.metrics, "mean_token_loss": result.mean_token_loss,
            "n_examples": result.n_examples, "wall_seconds": result.wall_seconds,
            "sample_predictions": result.sample_predictions,
            "per_example_primary": [row["scores"].get(primary) for row in result.per_example],
            "predictions_file": self._write_predictions(label, result.per_example),
        }
        self.tracker.log_metric(phase=f"decision_study:{label}", **{
            k: v for k, v in record.items() if k not in ("sample_predictions", "per_example_primary")})
        cache[key] = record
        self.ckpt.save(elapsed=self._elapsed())
        return record

    def _evaluate(self, delta: AdapterDelta | None, task: TaskSpec, guard: Any, *, split: SplitName,
                  n: int) -> Any:
        """Generate and score with ``delta`` (None = backbone): merged (D-026) or installed as LoRA."""
        with self._eval_model(delta) as model:
            return self.final_evaluator.evaluate(task, guard, model=model,
                                                 adapter_id=None if delta is None else delta.adapter_id,
                                                 split=split, max_examples=n)

    def _eval_model(self, delta: AdapterDelta | None):
        """Context yielding the model that generates with ``delta``: merged weights (D-026) or the
        delta installed as LoRA (run 1's path)."""
        if self.merger is not None:
            return self.merger.apply(delta)
        if delta is None:
            zero_delta_(self.peft_model)
        else:
            load_delta_into(self.peft_model, delta)
        return contextlib.nullcontext(self.peft_model)

    def _loss(self, delta: AdapterDelta, task_id: str) -> float:
        key = f"{self._delta_key(delta)}|{task_id}"
        cache = self.ckpt.cache("val_loss")
        if key not in cache:
            task = self.tasks[task_id]
            cache[key] = _validation_loss(
                self.peft_model, delta, task, self.splits[task_id], self.backbone.tokenizer,
                self.backbone.device, max_examples=self.study.supervision_loss_examples,
                batch_size=int(self.cfg.get("training.batch_size")))
        return cache[key]

    def _retention(self, old: AdapterEntry, kd: KDConfig) -> list:
        """KD ``D_rep`` -- none at all when the KD term is off (item 2)."""
        if not kd.enabled or kd.lam == 0.0 or kd.retention_examples <= 0:
            return []
        limit = int(kd.retention_examples)
        tasks = [t for t in old.metadata.tasks_served if t in self.tasks]
        if not tasks:
            return []
        per_task = max(1, -(-limit // len(tasks)))
        batches = []
        for task_id in tasks:
            task = self.tasks[task_id]
            examples = list(trainer_guard(self.splits[task_id]).get(self.study.retention_split))[:per_task]
            for start in range(0, len(examples), 2):
                batches.append(collate(self.backbone.tokenizer, examples[start : start + 2],
                                       max_input_tokens=task.max_input_tokens,
                                       max_target_tokens=task.max_target_tokens,
                                       prompt_truncation=task.prompt_truncation))
        return batches

    def _fuse_with(self, variant: FusionVariant, old: AdapterEntry, new_delta: AdapterDelta,
                   new_task_id: str, fused_id: str) -> tuple[AdapterDelta, dict[str, Any]]:
        """Fusion + KD of ``(old, new_delta)`` under one fusion-grid variant."""
        new_task = self.tasks[new_task_id]
        loader = GuardedLoader(
            trainer_guard(self.splits[new_task_id]), SplitName.TRAIN, new_task, self.backbone.tokenizer,
            BatchPlan(batch_size=self.runner.train_cfg.batch_size, shuffle=True, seed=self.cfg.seed),
        )
        task_batches = list(itertools.islice(iter(loader), max(1, variant.fusion.steps)))
        retention = self._retention(old, variant.kd)
        if self.backbone.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.monitor.record(f"fuse:{fused_id}:before")
        fused, record = run_fusion_kd(
            backbone=self.backbone, peft_model=self.peft_model, components=[old.delta, new_delta],
            task_batches=task_batches, retention_batches=retention, fusion_cfg=variant.fusion,
            kd_cfg=variant.kd, fused_adapter_id=fused_id, default_rank=self.runner.lora_spec.r,
        )
        peak = memory_snapshot(self.backbone.device, label=f"fuse:{fused_id}").get("gpu_peak_allocated_bytes")
        del task_batches, retention
        cleanup(self.backbone.device)
        self.monitor.record(f"fuse:{fused_id}:after")
        payload = record.as_dict()
        self.tracker.write_json(f"fusion_logs/{fused_id}.json", {
            **payload, "variant": variant.name, "alpha_summary": record.alpha_summary(),
            "loss_trend": record.loss_trend()})
        refactor = record.refactorization
        return fused, {
            "fused_adapter_id": fused_id, "variant": variant.name, "steps": record.steps,
            "wall_seconds": record.wall_seconds, "kd": record.kd, "alpha_summary": record.alpha_summary(),
            "alpha_moved_from_init": record.alpha_moved_from_init, "loss_trend": record.loss_trend(),
            "storage": refactor.get("storage"), "stored_bytes": fused.storage_bytes(),
            "would_be_svd_mean_relative_error": refactor.get("would_be_svd_mean_relative_error"),
            "would_be_svd_max_relative_error": refactor.get("would_be_svd_max_relative_error"),
            "svd_mean_relative_error": refactor.get("mean_relative_error"),
            "peak_allocated_bytes": peak,
        }

    # -- phase 0: preflight -------------------------------------------------------
    def phase0_preflight(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        if self.full.reference_run:
            reference = load_reference(self.full.reference_run)
            report["data_integrity"] = verify_reference_splits(
                raw=self.cfg.raw, tasks=self.tasks, splits=self.splits, reference=reference,
                task_ids=self.full.reference_tasks, tokenizer=self.backbone.tokenizer)
        else:
            report["data_integrity"] = {"ok": True, "skipped": "full_study.reference_run is not set"}
        self._write("data_integrity.json", report["data_integrity"])
        leakage = self.data_payload.get("leakage_audit") or {}
        report["leakage_audit"] = {"ok": leakage.get("clean", True) and (leakage.get("content") or {}).get("clean", True),
                                   "content_audit_ran": "content" in leakage}

        try:
            memory = self.preflight()        # run 1's worst-case train / fusion / generation checks
        except Exception as exc:              # e.g. an op with no deterministic kernel
            memory = {"ok": False, "checks": [], "error": f"{type(exc).__name__}: {exc}"}
        limit = self.full.memory_limit_bytes
        over = [c for c in memory.get("checks", []) if limit and (c.get("peak_allocated_bytes") or 0) > limit]
        memory["limit_bytes"] = limit
        memory["over_limit"] = over
        memory["ok"] = bool(memory.get("ok")) and not over
        report["memory"] = memory

        try:
            probe = self._timing_probe()
        except Exception as exc:
            probe = {"error": f"{type(exc).__name__}: {exc}", "costs": {}}
        report["time_projection"] = self._time_projection(probe)
        self._write("time_projection.json", report["time_projection"])

        if self.bertscorer is not None:
            try:
                self.bertscorer.load()
                report["bertscore"] = {"ok": True, "model_id": self.bertscorer.config.model_id,
                                       "revision": self.bertscorer.config.revision}
            except Exception as exc:
                report["bertscore"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        report["ok"] = all([report["data_integrity"]["ok"], report["leakage_audit"]["ok"], memory["ok"],
                            report["time_projection"]["fits"], (report.get("bertscore") or {"ok": True})["ok"]])
        self._write("preflight_full.json", report)
        return report

    def _training_steps(self, task_id: str) -> int:
        cfg = self.runner.train_cfg
        if cfg.max_steps:
            return int(cfg.max_steps)
        batches = max(1, math.ceil(len(self.splits[task_id][SplitName.TRAIN]) / cfg.batch_size))
        return max(1, int(max(1, batches // cfg.gradient_accumulation_steps) * cfg.epochs))

    def _generation_cap(self, task_id: str) -> int:
        return min(int(self.final_evaluator.generation.max_new_tokens), self.tasks[task_id].max_target_tokens)

    @torch.no_grad()
    def _timed_generate(self, model: Any, rows: list[list[int]], tokens: int) -> float:
        """Wall time of one generate call over ``rows`` (left-padded, as the evaluator batches)."""
        tokenizer = self.backbone.tokenizer
        pad = getattr(tokenizer, "pad_token_id", None)
        pad = getattr(tokenizer, "eos_token_id", 0) if pad is None else pad
        width = max(len(r) for r in rows)
        input_ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows], dtype=torch.long,
                                 device=self.backbone.device)
        attention = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], dtype=torch.long,
                                 device=self.backbone.device)
        kwargs = self.final_evaluator.generation.to_kwargs(
            pad_token_id=pad, eos_token_id=getattr(tokenizer, "eos_token_id", None))
        kwargs.update(max_new_tokens=tokens, min_new_tokens=tokens)
        if self.backbone.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate(input_ids=input_ids, attention_mask=attention, **kwargs)
        if self.backbone.device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    def _timing_probe(self) -> dict[str, Any]:
        """Measure this host's unit costs per task on TRAIN / VALIDATION data (never TEST)."""
        factor = float(self.full.projection.get("generation_length_factor", 1.25))
        probe_tokens = int(self.full.projection.get("probe_decode_tokens", 16))
        costs: dict[str, dict[str, float]] = {}
        expected: dict[str, float] = {}
        variant = self.full.fusion_grid[0]
        heaviest = next((v for v in self.full.fusion_grid if v.fusion.storage == "keep_components"
                         and v.kd.lam > 0 and v.fusion.steps > 0), variant)
        fresh = extract_delta(self.peft_model, adapter_id="probe-old")
        for task_id in self.tasks:
            task = self.tasks[task_id]
            val = list(utility_guard(self.splits[task_id]).get(SplitName.VALIDATION))
            batch = val[: self.runner.train_cfg.batch_size]
            zero_delta_(self.peft_model)
            start = time.perf_counter()
            probe_quality(self.peft_model, None, [collate(self.backbone.tokenizer, batch,
                                                          max_input_tokens=task.max_input_tokens,
                                                          max_target_tokens=task.max_target_tokens,
                                                          prompt_truncation=task.prompt_truncation)],
                          self.backbone.device, ece_bins=10)
            loss_example = (time.perf_counter() - start) / max(1, len(batch))
            # Generation is timed on the evaluator's own path: the same model context (merged
            # weights under D-026) and the same batch width, left-padded.
            width = max(1, int(self.final_evaluator.generation.batch_size))
            prompts = [encode_prompt(self.backbone.tokenizer, e.prompt, max_input_tokens=task.max_input_tokens,
                                     truncation=task.prompt_truncation) for e in val[:64]]
            cap = self._generation_cap(task_id)
            k = max(2, min(probe_tokens, cap))
            with self._eval_model(fresh) as model:
                self._timed_generate(model, prompts[:width], 1)                      # warm-up
                prefill = self._timed_generate(model, prompts[:width], 1)
                per_token = max(0.0, (self._timed_generate(model, prompts[:width], k) - prefill) / (k - 1))
            # Reference output length in tokens (the label positions), as trained adapters stop near it.
            # A batch decodes until its longest member stops, so the expected decode length of a batch
            # is the longest target among prompts the evaluator would group (sorted by prompt length).
            targets = [sum(1 for label in tokenize_example(
                self.backbone.tokenizer, e, max_input_tokens=task.max_input_tokens,
                max_target_tokens=task.max_target_tokens, prompt_truncation=task.prompt_truncation)[1]
                if label != IGNORE_INDEX) for e in val[:64]]
            order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))
            groups = [order[i : i + width] for i in range(0, len(order), width)]
            expected[task_id] = sum(min(float(cap), factor * max(targets[i] for i in g)) for g in groups) / max(1, len(groups))
            gen_example = loss_example + (prefill + expected[task_id] * per_token) / width

            probe_cfg = TrainingConfig(**{**asdict(self.runner.train_cfg), "max_steps": 2, "eval_every_steps": None})
            _, record = train_adapter(backbone=self.backbone, task=task, guard=trainer_guard(self.splits[task_id]),
                                      lora_spec=self.runner.lora_spec, train_cfg=probe_cfg, resolved=self.cfg,
                                      adapter_id=f"timing-probe-{task_id}")
            train_step = record.wall_seconds / max(1, record.steps)

            def fusion_variant(steps: int) -> FusionVariant:
                return FusionVariant(heaviest.name, False,
                                     FusionConfig(**{**asdict(heaviest.fusion), "steps": steps}), heaviest.kd, {})
            other = AdapterDelta(layers=fresh.layers, adapter_id="probe-new")
            entry = AdapterEntry(delta=fresh, metadata=AdapterMetadata(
                adapter_id="probe-old", task_name=task.task_name, task_type=task.task_type.value,
                dataset=str(task.hf_id), dataset_revision=task.hf_revision, model_version="probe",
                backbone_fingerprint=self.backbone.fingerprint, run_mode=self.cfg.run_mode.value,
                lora_rank=self.runner.lora_spec.r, target_modules=list(self.runner.lora_spec.target_modules),
                tasks_served=[task_id]))
            # (3 steps - 1 step) / 2: the first step's one-off set-up is not a per-step cost.
            _, one = self._fuse_with(fusion_variant(1), entry, other, task_id, f"timing-probe-fuse1-{task_id}")
            _, three = self._fuse_with(fusion_variant(3), entry, other, task_id, f"timing-probe-fuse3-{task_id}")
            fusion_step = max(0.0, (three["wall_seconds"] - one["wall_seconds"]) / 2)

            features_cfg = self.cfg.section("features")
            probe_n = int(features_cfg.get("semantic", {}).get("hidden_probe_examples", 8)) + \
                int(features_cfg.get("performance", {}).get("max_eval_examples", 8))
            costs[task_id] = {
                "loss_example": loss_example, "gen_example": gen_example, "train_step": train_step,
                "fusion_step": fusion_step,
                # Feature extraction per (task, candidate): quality + divergence probes on
                # ``probe_n`` examples (x2: candidate and A_temp) and one gradient probe.
                "probe_candidate": 2 * probe_n * loss_example + train_step / self.runner.train_cfg.gradient_accumulation_steps,
            }
        del fresh
        cleanup(self.backbone.device)
        return {"costs": costs, "expected_generation_tokens_per_batch": expected,
                "generation": {"merged_weights": self.merger is not None,
                               "batch_size": int(self.final_evaluator.generation.batch_size)},
                "note": "probe costs on this host; probe_candidate is derived from measured loss and train costs"}

    def _time_projection(self, probe: Mapping[str, Any]) -> dict[str, Any]:
        sources = {"probe": probe.get("costs") or {}}
        if self.full.reference_run and Path(self.full.reference_run).exists():
            ref_cfg = (load_reference(self.full.reference_run).get("config_used") or {}).get("config") or {}
            ref_cap = int((ref_cfg.get("generation") or {}).get("max_new_tokens", 128))
            ref_tasks = (ref_cfg.get("datasets") or {}).get("tasks") or {}
            caps_then = {t: min(ref_cap, int(ref_tasks[t]["max_target_tokens"])) for t in ref_tasks
                         if "max_target_tokens" in ref_tasks[t]}
            caps_now = {t: self._generation_cap(t) for t in self.tasks}
            sources["run1_scaled"] = run1_unit_costs(self.full.reference_run, generation_caps_run1=caps_then,
                                                     generation_caps_now=caps_now)
        dropped = []
        if "run1_scaled" in sources and self._generation_differs_from_run1():
            # Run 1 generated unmerged, one example at a time; its per-example generation cost is not
            # a cost of this run's path (D-026), so only the probe prices generation.
            for task_costs in sources["run1_scaled"].values():
                if task_costs.pop("gen_example", None) is not None:
                    dropped.append("gen_example")
        costs = merge_costs(*sources.values())
        items = self._workload()
        projection = project(items, costs, budget_hours=self.study.time_budget_hours)
        projection.update({"sources": sources, "probe": {k: v for k, v in probe.items() if k != "costs"},
                           "workload": summarise_items(items),
                           "rule": "per (task, kind) the larger of the probe and run-1-scaled costs",
                           "run1_generation_costs_dropped": bool(dropped)})
        return projection

    def _generation_differs_from_run1(self) -> bool:
        return self.merger is not None or int(self.final_evaluator.generation.batch_size) > 1

    def _workload(self) -> list[WorkItem]:
        """The work every phase will do, counted from this configuration (time projection)."""
        f, items = self.full, []
        seeds = [a for a in self.study.initial_adapters]
        seed_tasks = [a.task_id for a in seeds]
        loss_n = self.study.supervision_loss_examples

        def add(phase: str, kind: str, task: str, count: float, note: str = "") -> None:
            if count:
                items.append(WorkItem(phase, kind, task, float(count), note))

        for a in seeds:
            if a.adapter_id not in f.reuse_adapter_ids:
                add("P1", "train_step", a.task_id, self._training_steps(a.task_id), f"train {a.adapter_id}")
            add("P1", "gen_example", a.task_id, 2 * self._examples("label", a.task_id)
                + 2 * self._examples("test", a.task_id), "zero-shot + standalone VALIDATION and TEST")
            add("P1", "loss_example", a.task_id, loss_n, "standalone validation loss")
        pair_ids = list(itertools.permutations(f.fusion_pair_adapters, 2))
        by_id = {a.adapter_id: a.task_id for a in seeds}
        for variant in f.fusion_grid:
            for new_id, old_id in pair_ids:
                t_new, t_old = by_id[new_id], by_id[old_id]
                add("P2", "fusion_step", t_new, variant.fusion.steps, variant.name)
                add("P2", "gen_example", t_new, self._examples("label", t_new))
                add("P2", "gen_example", t_old, self._examples("label", t_old))
                add("P2", "loss_example", t_new, loss_n)
                add("P2", "loss_example", t_old, loss_n)
        steps = max((v.fusion.steps for v in f.fusion_grid if not v.reference), default=0)
        for new, old in itertools.permutations(seeds, 2):
            add("P3", "fusion_step", new.task_id, steps, "selected config (max over the grid)")
            add("P3", "gen_example", new.task_id, 2 * self._examples("label", new.task_id), "fused + reuse on new task")
            add("P3", "gen_example", old.task_id, self._examples("label", old.task_id), "fused on old task")
            add("P3", "loss_example", new.task_id, 2 * loss_n + 2 * f.probe_examples)
            add("P3", "loss_example", old.task_id, loss_n)
            add("P3", "probe_candidate", new.task_id, 1)
        fuse_options = int(f.projection.get("fuse_options_per_task", f.fuse_top_k + 2))
        for spec in self.study.evaluation_tasks:
            t = spec.task_id
            add("P5", "train_step", t, self._training_steps(t), "temporary adapter")
            add("P5", "gen_example", t, 2 * self._examples("eligibility", t), "eligibility")
            add("P5", "gen_example", t, (len(seeds) + 1) * f.probe_examples, "SUPPORT probes")
            add("P5", "loss_example", t, (len(seeds) + 1) * f.probe_examples, "gate features")
            add("P5", "probe_candidate", t, len(seeds), "utility features")
            add("P5", "gen_example", t, (2 + len(seeds) + fuse_options) * self._examples("test", t),
                "TEST: CREATE, BACKBONE, REUSE, FUSE")
            add("P5", "fusion_step", t, fuse_options * steps, "measured FUSE set")
            # The fused candidates are chosen in the run, so their old tasks are not known here:
            # each seed task is equally likely, priced at its own TEST size and cost.
            for old in seed_tasks:
                add("P5", "gen_example", old, fuse_options * self._examples("test", old) / len(seed_tasks),
                    "old-task TEST of FUSE options (candidates spread evenly over the seeds)")
        stream = f.stream
        fusions = int(f.projection.get("stream_fusions_per_run", 4))
        order = list(stream["order"])
        for policy in stream["policies"]:
            for _cap in stream["caps"]:
                for i, t in enumerate(order):
                    # always_separate decides nothing and serves every task by its own adapter (R2)
                    # or, at the cap, the backbone -- both already measured on TEST in phases 1 and
                    # 5 and cached -- so its stream runs add no generation.
                    if policy != "always_separate":
                        k = min(i, len(seeds))
                        add("P6", "gen_example", t, (k + 1) * f.probe_examples, "SUPPORT probes")
                        add("P6", "loss_example", t, (k + 1) * f.probe_examples, "gate features")
                        add("P6", "probe_candidate", t, k, "utility features")
                        add("P6", "gen_example", t, self._examples("test", t), "arrival TEST (upper bound)")
                if policy != "always_separate":
                    add("P6", "fusion_step", order[-1], fusions * steps, "stream fusions (assumed count)")
                    add("P6", "gen_example", order[-1], fusions * 2 * self._examples("test", order[-1]),
                        "re-evaluation after fusions")
        cycles = self.study.memory_cycles
        add("P7", "fusion_step", seed_tasks[0], cycles * steps)
        add("P7", "gen_example", seed_tasks[0], cycles * 2 * self.study.memory_cycle_eval_examples)
        return items

    # -- phase 1: seeds -----------------------------------------------------------
    def _load_reused(self, adapter_id: str, task: TaskSpec) -> AdapterEntry:
        root = Path(str(self.full.reuse_adapters_dir)) / adapter_id
        delta = AdapterDelta.load(root / "delta.pt")
        metadata = AdapterMetadata.load(root / "metadata.json")
        if metadata.content_hash and delta.content_hash() != metadata.content_hash:
            raise PreconditionError(f"reused adapter {adapter_id}: weights do not match the recorded hash")
        if metadata.backbone_fingerprint != self.backbone.fingerprint:
            raise PreconditionError(f"reused adapter {adapter_id} was fitted to backbone "
                                    f"{metadata.backbone_fingerprint[:12]}, not this one")
        if metadata.tasks_served != [task.task_id]:
            raise PreconditionError(f"reused adapter {adapter_id} serves {metadata.tasks_served}, "
                                    f"not ['{task.task_id}']")
        return AdapterEntry(delta=delta, metadata=metadata)

    def phase1_seeds(self) -> dict[str, Any]:
        records = []
        for assignment in self.study.initial_adapters:
            task = self.tasks[assignment.task_id]
            self.runner.ensure_representation(task.task_id)

            def build(assignment=assignment, task=task) -> dict[str, Any]:
                reused = assignment.adapter_id in self.full.reuse_adapter_ids
                if assignment.adapter_id in self.repository:       # registered before a crash
                    entry = self.repository.retrieve(assignment.adapter_id)
                    record = dict(entry.metadata.training_configuration)
                elif reused:
                    entry = self._load_reused(assignment.adapter_id, task)
                    self.repository.register(entry, bootstrap=True, task_index=0)
                    record = dict(entry.metadata.training_configuration)
                else:
                    delta, trained = train_adapter(
                        backbone=self.backbone, task=task, guard=trainer_guard(self.splits[task.task_id]),
                        lora_spec=self.runner.lora_spec, train_cfg=self.runner.train_cfg, resolved=self.cfg,
                        adapter_id=assignment.adapter_id, budget=self.runner.budget)
                    record = trained.as_dict()
                    entry = AdapterEntry(delta=delta, metadata=self.runner._metadata_for(
                        assignment.adapter_id, task, delta, record, AdapterOrigin.TRAINED, [task.task_id]))
                    self.repository.register(entry, bootstrap=True, task_index=0)
                return {
                    "adapter_id": assignment.adapter_id, "task_id": task.task_id, "role": assignment.role,
                    "source": "loaded_from_run1" if reused else "trained",
                    "dataset": task.hf_id, "dataset_revision": task.hf_revision,
                    "train_rows": len(self.splits[task.task_id][SplitName.TRAIN]),
                    **{k: record.get(k) for k in ("steps", "epochs_completed", "final_train_loss", "loss_trajectory",
                                                   "validation_trajectory", "wall_seconds", "tokens_seen",
                                                   "peak_memory_bytes")},
                    "checkpoint": str(self.repository.retrieve(assignment.adapter_id).path),
                    "content_hash": self.repository.retrieve(assignment.adapter_id).delta.content_hash(),
                }
            records.append(self._unit(f"P1:adapter:{assignment.adapter_id}", build))

        content = {a.adapter_id: self.splits[a.task_id].content_hashes(SplitName.TRAIN) for a in self.study.initial_adapters}
        for item in records:
            mine = content[item["adapter_id"]]
            item["shares_training_rows_with"] = {other: len(mine & rows) for other, rows in content.items()
                                                 if other != item["adapter_id"] and mine & rows}
        self._write("initial_adapters.json", records)

        zero_shot, standalone = {}, {}
        for assignment in self.study.initial_adapters:
            t, delta = assignment.task_id, self.repository.retrieve(assignment.adapter_id).delta
            zero_shot[t] = {
                "validation": self._measure(None, t, split=SplitName.VALIDATION, n=self._examples("label", t),
                                            label=f"zero_shot:val:{t}")["primary_score"],
                "test": self._measure(None, t, split=SplitName.TEST, n=self._examples("test", t),
                                      label=f"zero_shot:test:{t}")["primary_score"]}
            standalone[assignment.adapter_id] = {
                "task_id": t,
                "validation": self._measure(delta, t, split=SplitName.VALIDATION, n=self._examples("label", t),
                                            label=f"standalone:val:{assignment.adapter_id}"),
                "test": self._measure(delta, t, split=SplitName.TEST, n=self._examples("test", t),
                                      label=f"standalone:test:{assignment.adapter_id}")}
        self._write("standalone_test_scores.json", {k: v["test"] for k, v in standalone.items()})
        self._write("standalone_scores.json", standalone)
        self._write("zero_shot.json", zero_shot)
        return {"initial_adapters": [{k: r[k] for k in ("adapter_id", "task_id", "source", "steps",
                                                       "final_train_loss", "wall_seconds",
                                                       "shares_training_rows_with")} for r in records],
                "zero_shot": zero_shot,
                "standalone": {k: {"validation": v["validation"]["primary_score"],
                                   "test": v["test"]["primary_score"]} for k, v in standalone.items()}}

    # -- shared pair measurement (phases 2 and 3) ----------------------------------
    def _root(self, task_id: str) -> str:
        return self.tasks[task_id].duplicate_of or task_id

    def pair_type(self, new_task: str, old_task: str) -> str:
        a, b = self._root(new_task), self._root(old_task)
        if a == b:
            return "same_task"
        if any(a in group and b in group for group in self.full.related_groups):
            return "related"
        return "unrelated"

    def _pair(self, variant: FusionVariant, new_id: str, old_id: str, prefix: str, *,
              supervision: bool) -> dict[str, Any]:
        """Fuse ``new`` into ``old`` under ``variant`` and measure it on VALIDATION (labels: D-019)."""
        new, old = self.repository.retrieve(new_id), self.repository.retrieve(old_id)
        t_new, t_old = new.metadata.tasks_served[0], old.metadata.tasks_served[0]
        n_new, n_old = self._examples("label", t_new), self._examples("label", t_old)
        start = time.perf_counter()
        out: dict[str, Any] = {"new_adapter": new_id, "new_task": t_new, "old_adapter": old_id, "old_task": t_old,
                               "pair_type": self.pair_type(t_new, t_old), "same_task": t_new == t_old,
                               "variant": variant.name}
        if supervision:
            contexts = self.runner._build_contexts(t_new, 0, new.delta, candidates=[old])
            vector = self.runner.extractor.extract(contexts[old_id])
            del contexts
            out["features"] = asdict(vector)
            out["gate_features"] = self.runner.reuse_gate_features(
                t_new, [old], probe_examples=self.full.probe_examples)[old_id]
        fused, fusion = self._fuse_with(variant, old, new.delta, t_new, f"{prefix}-{new_id}-into-{old_id}")
        s_new = self._measure(new.delta, t_new, split=SplitName.VALIDATION, n=n_new, label=f"standalone:val:{new_id}")
        s_old = self._measure(old.delta, t_old, split=SplitName.VALIDATION, n=n_old, label=f"standalone:val:{old_id}")
        f_new = self._measure(fused, t_new, split=SplitName.VALIDATION, n=n_new,
                              label=f"{prefix}:{new_id}-into-{old_id}:new:{t_new}")
        f_old = self._measure(fused, t_old, split=SplitName.VALIDATION, n=n_old,
                              label=f"{prefix}:{new_id}-into-{old_id}:old:{t_old}")
        scale = float(self.cfg.get("utility.supervision.loss_to_score_scale"))
        pre_new, pre_old = self._loss(new.delta, t_new), self._loss(old.delta, t_old)
        post_new, post_old = self._loss(fused, t_new), self._loss(fused, t_old)
        relative_new = criteria.relative_score(f_new["primary_score"], s_new["primary_score"])
        old_drop = criteria.relative_drop(s_old["primary_score"], f_old["primary_score"])
        forgetting_proxy = max(0.0, _loss_to_score(pre_old, scale=scale) - _loss_to_score(post_old, scale=scale))
        beneficial_proxy = (forgetting_proxy <= 1e-9
                            and _loss_to_score(post_new, scale=scale) >= _loss_to_score(pre_new, scale=scale) - 1e-9)
        out.update({
            "measured_validation": {
                "examples": {"new": n_new, "old": n_old},
                "new_standalone": s_new["primary_score"], "new_fused": f_new["primary_score"],
                "old_standalone": s_old["primary_score"], "old_fused": f_old["primary_score"],
                "metric": {"new": s_new["primary_score_name"], "old": s_old["primary_score_name"]},
            },
            "relative_new": relative_new, "old_drop": old_drop,
            "beneficial": criteria.is_beneficial(relative_new, old_drop, equivalence=self.full.equivalence,
                                                 forgetting=self.full.forgetting),
            "beneficial_loss_proxy": beneficial_proxy, "forgetting_proxy": forgetting_proxy,
            "validation_loss": {"new_standalone": pre_new, "new_fused": post_new,
                                "old_standalone": pre_old, "old_fused": post_old},
            "fusion": fusion,
        })
        if supervision:
            reuse = self._measure(old.delta, t_new, split=SplitName.VALIDATION, n=n_new,
                                  label=f"reuse:val:{old_id}:{t_new}")
            out["reuse_on_new_task"] = reuse["primary_score"]
            out["reuse_gap_measured"] = criteria.relative_score(reuse["primary_score"], s_new["primary_score"]) - 1.0
            out["reuse_gap_loss_proxy"] = (_loss_to_score(self._loss(old.delta, t_new), scale=scale)
                                           - _loss_to_score(pre_new, scale=scale))
            out["targets"] = {
                "primary_score": relative_new, "forgetting": old_drop,
                "confidence": _loss_to_score(post_new, scale=scale),
                "memory": fusion["stored_bytes"] / 1e6, "communication": fusion["stored_bytes"] / 1e6,
                "long_term_loss": post_new if post_new == post_new else 0.0,
            }
        del fused
        cleanup(self.backbone.device)
        out["wall_seconds"] = time.perf_counter() - start
        return out

    # -- phase 2: fusion selection ------------------------------------------------
    def phase2_fusion_selection(self) -> dict[str, Any]:
        pairs = list(itertools.permutations(self.full.fusion_pair_adapters, 2))
        rows: dict[str, list[dict[str, Any]]] = {}
        for variant in self.full.fusion_grid:
            for new_id, old_id in pairs:
                key = f"P2:{variant.name}:{new_id}->{old_id}"
                try:
                    rows.setdefault(variant.name, []).append(self._unit(
                        key, lambda v=variant, a=new_id, b=old_id: self._pair(v, a, b, f"sel-{v.name}", supervision=False)))
                except Exception as exc:
                    self.failures.append({"phase": "P2", "unit": key, "error": f"{type(exc).__name__}: {exc}"})
                    log.exception("fusion grid unit %s failed", key)
                    cleanup(self.backbone.device)
        selection = criteria.select_fusion_config(
            {name: [{"relative_new": r["relative_new"], "old_drop": r["old_drop"]} for r in items]
             for name, items in rows.items()},
            ineligible=[v.name for v in self.full.fusion_grid if v.reference],
            forgetting=self.full.forgetting, equivalence=self.full.equivalence)
        self.selected = self.full.variant(selection["chosen"])
        payload = {**selection, "rows": rows, "configs": {v.name: v.raw for v in self.full.fusion_grid},
                   "selected_on": "VALIDATION", "run1_reference": self._run1_fusion_reference()}
        self._write("fusion_selection.json", payload)
        return {"chosen": selection["chosen"], "fusion_viable": selection["fusion_viable"],
                "summary": selection["summary"]}

    def _run1_fusion_reference(self) -> list[dict[str, Any]] | None:
        """Run 1's stored generation checks for the same pairs (16 examples, loss-proxy era)."""
        if not self.full.reference_run:
            return None
        path = Path(self.full.reference_run) / "decision_study" / "utility_supervision.json"
        if not path.exists():
            return None
        out = []
        for item in json.loads(path.read_text()).get("measurements", []):
            check = item.get("generation_check_validation") or {}
            out.append({"new_adapter": item["new_adapter"], "old_adapter": item["old_adapter"], **check,
                        "note": "run 1: refactorize_svd, lambda 1, installed at 2x (D-017), 16 examples"})
        return out

    # -- phase 3: supervision -----------------------------------------------------
    def phase3_supervision(self) -> dict[str, Any]:
        ids = [a.adapter_id for a in self.study.initial_adapters]
        measurements = []
        for new_id, old_id in itertools.permutations(ids, 2):
            key = f"P3:{new_id}->{old_id}"
            try:
                measurements.append(self._unit(key, lambda a=new_id, b=old_id: self._pair(
                    self.selected, a, b, "sup", supervision=True)))
            except Exception as exc:
                self.failures.append({"phase": "P3", "unit": key, "error": f"{type(exc).__name__}: {exc}"})
                log.exception("supervision pair %s failed", key)
                cleanup(self.backbone.device)
        by_type: dict[str, list[bool]] = {}
        for m in measurements:
            by_type.setdefault(m["pair_type"], []).append(m["beneficial"])
        agreement = [m["beneficial"] == m["beneficial_loss_proxy"] for m in measurements]
        return {
            "n_pairs": len(measurements),
            "beneficial_fraction": (sum(m["beneficial"] for m in measurements) / len(measurements)
                                    if measurements else None),
            "beneficial_fraction_by_pair_type": {k: {"n": len(v), "beneficial_fraction": sum(v) / len(v)}
                                                 for k, v in by_type.items()},
            "proxy_label_agreement": sum(agreement) / len(agreement) if agreement else None,
        }

    def _supervision_measurements(self) -> list[dict[str, Any]]:
        ids = [a.adapter_id for a in self.study.initial_adapters]
        return [self.ckpt.unit(f"P3:{a}->{b}") for a, b in itertools.permutations(ids, 2)
                if self.ckpt.has_unit(f"P3:{a}->{b}")]

    # -- phase 4: utility network, gate, thresholds -------------------------------
    def phase4_utility(self) -> dict[str, Any]:
        measurements = self._supervision_measurements()
        supervision = SupervisionSet(fit_split="utility_train")
        gate_examples = []
        for m in measurements:
            vector = FeatureVector(**{**m["features"],
                                      "family_slices": {k: tuple(v) for k, v in m["features"]["family_slices"].items()}})
            supervision.examples.append(SupervisionExample(
                candidate_id=m["old_adapter"], task_id=m["new_task"], features=vector, targets=m["targets"],
                proxy=False, proxy_steps=None, beneficial=m["beneficial"]))
            gate_examples.append(GateExample(task_id=m["new_task"], candidate_id=m["old_adapter"],
                                             features=m["gate_features"], gap=m["reuse_gap_measured"]))
        supervision.n_pairs_considered = supervision.n_pairs_evaluated = len(measurements)

        cv = fit_with_cv(self.cfg, supervision, gate=self.full.fit_gate)
        paths = self.tracker.paths.utility_network
        if cv.fitted:
            cv.network.save(paths / "utility_network.pt", run_mode=self.cfg.run_mode.value,
                            feature_names=list(cv.normalizer.feature_names))
            cv.normalizer.save(paths / "normalizer.json")
            (paths / "thresholds.json").write_text(json.dumps(cv.thresholds.as_dict(), indent=2))
        gate_cfg = ReuseGateConfig.from_config(self.cfg.section("utility")["reuse_gate"])
        gate = ReuseGate.fit(gate_examples, gate_cfg, run_mode=self.cfg.run_mode.value)
        gate.save(paths / "reuse_gate.json")
        fixed = self._select_fixed_rule_threshold(supervision) if supervision.examples else {"threshold": None}
        (paths / "fixed_rule_threshold.json").write_text(json.dumps(fixed, indent=2, default=str))
        self._restore_frozen()
        report = {
            "measurements": measurements,
            "supervision": supervision.as_dict(),
            "label_definition": "measured metric on VALIDATION (D-019); loss proxy kept as beneficial_loss_proxy",
            "utility_network": cv.as_dict(),
            "amaun_uses": "utility network" if cv.fitted else "probe_rule (automatic fallback)",
            "fallback_reason": None if cv.fitted else cv.reason,
            "reuse_gate": {**gate.as_dict(), "target": "relative measured gap s(A_cand)/s(A_dedicated) - 1",
                           "in_sample": [{"task": e.task_id, "candidate": e.candidate_id, "measured_gap": e.gap,
                                          "predicted_gap": gate.predict(e.features)} for e in gate_examples]},
            "fixed_rule_threshold": fixed,
        }
        self._write("utility_supervision.json", report)
        return {"fitted": cv.fitted, "reason": cv.reason,
                "thresholds": cv.thresholds.as_dict() if cv.thresholds else None,
                "oof_decision_accuracy": cv.oof_decision_accuracy, "per_head_oof_mse": cv.per_head_oof_mse,
                "fixed_rule_threshold": fixed.get("threshold"), "reuse_gate_trained": gate.trained}

    def _restore_frozen(self) -> None:
        """Reload everything Phase 4 froze (also after a resume)."""
        paths = self.tracker.paths.utility_network
        mode = self.cfg.run_mode.value
        if (paths / "utility_network.pt").exists():
            self.network, _ = UtilityNetwork.load(paths / "utility_network.pt", expected_run_mode=mode)
            self.network.eval()
            self.normalizer = FeatureNormalizer.load(paths / "normalizer.json", expected_run_mode=mode)
            self.thresholds = Thresholds(**json.loads((paths / "thresholds.json").read_text()))
        else:
            self.network = self.normalizer = self.thresholds = None
        self.reuse_gate = (ReuseGate.load(paths / "reuse_gate.json", expected_run_mode=mode)
                           if (paths / "reuse_gate.json").exists() else None)
        fixed = paths / "fixed_rule_threshold.json"
        self.fixed_rule_threshold = json.loads(fixed.read_text()) if fixed.exists() else {"threshold": None}

    # -- decisions (phases 5 and 6) -------------------------------------------------
    def probes(self, task_id: str, candidates: Sequence[AdapterEntry]) -> dict[str, Any]:
        """SUPPORT probes: loss (gate features) and measured score for the backbone and candidates."""
        n = self.full.probe_examples
        gate_features = self.runner.reuse_gate_features(task_id, candidates, probe_examples=n)
        backbone = {"loss": next(iter(gate_features.values()))["base_loss"] if gate_features else float("nan"),
                    "score": self._measure(None, task_id, split=SplitName.SUPPORT, n=n,
                                           label=f"probe:{task_id}:backbone")["primary_score"]}
        if not gate_features:
            from amaun.manager.reuse_gate import support_probe_batches
            batches = support_probe_batches(self.backbone.tokenizer, self.tasks[task_id], self.splits[task_id], count=n)
            backbone["loss"] = probe_quality(self.peft_model, None, batches, self.backbone.device, ece_bins=10)["loss"]
        scores = {e.adapter_id: {"loss": gate_features[e.adapter_id]["cand_loss"],
                                 "score": self._measure(e.delta, task_id, split=SplitName.SUPPORT, n=n,
                                                        label=f"probe:{task_id}:{e.adapter_id}")["primary_score"]}
                  for e in candidates}
        return {"backbone": backbone, "candidates": scores, "gate_features": gate_features}

    def decide(self, *, task_id: str, task_index: int, temp_delta: AdapterDelta,
               candidates: Sequence[AdapterEntry], policies: Sequence[str], log_as: str) -> dict[str, Any]:
        """Every requested policy's choice for one arrival.  Reads SUPPORT only (no TEST)."""
        if not candidates:
            return {"decisions": {p: {"option": criteria.CREATE, "reason": "repository empty"} for p in policies},
                    "probes": None}
        probes = self.probes(task_id, candidates)
        backbone, cand = probes["backbone"], probes["candidates"]
        gate = None
        if self.reuse_gate is not None and any(p in policies for p in ("amaun",)):
            gate = self.reuse_gate.decide(probes["gate_features"], run_mode=self.cfg.run_mode.value)
        vectors: dict[str, FeatureVector] = {}
        if any(p in policies for p in ("amaun", "amaun_train_first", "fixed_rule", "always_merge")):
            contexts = self.runner._build_contexts(task_id, task_index, temp_delta, candidates=list(candidates))
            vectors = {cid: self.runner.extractor.extract(ctx) for cid, ctx in contexts.items()}
            del contexts
        utilities = None
        if self.network is not None and vectors:
            scored = score_candidates(network=self.network, normalizer=self.normalizer, vectors=vectors,
                                      run_mode=self.cfg.run_mode.value)
            utilities = {s.adapter_id: s.utility for s in scored}
        margin = float(self.cfg.get("manager.reuse_prescreen_margin", 0.0))
        prescreen = {cid: (v.as_dict().get("performance.L_val_gap") is not None
                           and v.as_dict()["performance.L_val_gap"] <= margin) for cid, v in vectors.items()}
        rule = criteria.probe_rule(backbone, cand, equivalence=self.full.equivalence)
        decisions: dict[str, dict[str, Any]] = {}
        for policy in policies:
            if policy in ("amaun", "amaun_train_first"):
                if self.network is None:
                    decisions[policy] = {**rule, "fallback": "probe_rule (utility network not fitted)"}
                    continue
                if policy == "amaun":
                    reuse = gate.adapter_id if gate is not None and gate.fired else None
                else:   # D-020: REUSE from the post-training evidence
                    best = max(utilities, key=lambda c: (utilities[c], c))
                    reuse = best if utilities[best] >= self.thresholds.tau_reuse and prescreen.get(best) else None
                decisions[policy] = criteria.amaun_order(
                    reuse_choice=reuse, backbone=backbone, candidates=cand, utilities=utilities,
                    tau_u=self.thresholds.tau_u, equivalence=self.full.equivalence)
            elif policy == "probe_rule":
                decisions[policy] = rule
            elif policy == "always_separate":
                decisions[policy] = {"option": criteria.CREATE, "reason": "policy: always create"}
            elif policy == "reuse_only":
                best = criteria.best_by_probe_score(cand)
                decisions[policy] = {"option": f"REUSE:{best}", "reason": "best SUPPORT probe score"}
            elif policy == "fuse_only":
                best = criteria.best_by_loss_gain(cand, backbone["loss"])
                decisions[policy] = {"option": f"FUSE:{best}", "reason": "best SUPPORT probe loss gain"}
            elif policy == "always_merge":
                strategy = AlwaysMergeStrategy(self.study.merge_feature,
                                               higher_is_more_similar=self.study.merge_feature_higher_is_more_similar)
                ranked = strategy.score(vectors, run_mode=self.cfg.run_mode.value)
                decisions[policy] = {"option": f"FUSE:{ranked[0].adapter_id}",
                                     "reason": f"nearest by {self.study.merge_feature}"}
            elif policy == "fixed_rule":
                threshold = (self.fixed_rule_threshold or {}).get("threshold")
                if threshold is None:
                    decisions[policy] = {"option": None, "reason": "fixed-rule threshold unavailable",
                                         "measured": False}
                    continue
                sim = self.cfg.section("manager").get("similarity") or {}
                strategy = SimilaritySelectionStrategy(SimilarityConfig(
                    str(sim.get("feature", "semantic.d_task_sq")), float(threshold),
                    bool(sim.get("higher_is_more_similar", False))))
                decision = strategy.decide(task_id=task_id, task_index=task_index,
                                           scores=strategy.score(vectors, run_mode=self.cfg.run_mode.value),
                                           repository_size=len(candidates), temp_adapter_trained=True,
                                           config_hash=self.cfg.config_hash)
                decisions[policy] = {"option": self._option_key(decision), "reason": decision.reason}
        for policy, decision in decisions.items():
            self.tracker.log_decision(task_id=task_id, task_index=task_index, strategy=policy, study=log_as,
                                      option=decision.get("option"), reason=decision.get("reason"),
                                      fallback=decision.get("fallback"))
        return {"decisions": decisions, "probes": {"backbone": backbone, "candidates": cand},
                "gate": gate.as_dict() if gate is not None else None, "utilities": utilities,
                "prescreen": prescreen,
                "features": {cid: {"values": v.as_dict(), "missing": list(v.missing)} for cid, v in vectors.items()}}

    # -- phase 5: evaluation tasks ------------------------------------------------
    def _temp(self, task_id: str) -> dict[str, Any]:
        """The evaluation task's dedicated adapter (trained once; reloaded after a resume)."""
        if task_id in self.temp:
            return self.temp[task_id]
        temp_id = f"temp-{task_id}"
        task = self.tasks[task_id]

        def train() -> dict[str, Any]:
            delta, record = train_adapter(
                backbone=self.backbone, task=task, guard=trainer_guard(self.splits[task_id]),
                lora_spec=self.runner.lora_spec, train_cfg=self.runner.train_cfg, resolved=self.cfg,
                adapter_id=temp_id, budget=self.runner.budget)
            entry = AdapterEntry(delta=delta, metadata=self.runner._metadata_for(
                temp_id, task, delta, record.as_dict(), AdapterOrigin.TRAINED, [task_id]))
            self.repository.candidates.put(entry)       # candidate store: never part of R_c
            return {"adapter_id": temp_id, "path": str(entry.path), "steps": record.steps,
                    "wall_seconds": record.wall_seconds, "final_train_loss": record.final_train_loss,
                    "train_rows": len(self.splits[task_id][SplitName.TRAIN]),
                    "peak_memory_bytes": record.peak_memory_bytes, "loss_trajectory": record.loss_trajectory,
                    "content_hash": delta.content_hash()}

        record = self._unit(f"P5:train:{task_id}", train)
        path = Path(record["path"])
        delta = AdapterDelta.load(path / "delta.pt")
        metadata = AdapterMetadata.load(path / "metadata.json")
        if delta.content_hash() != record["content_hash"]:
            raise PreconditionError(f"temporary adapter {temp_id} on disk does not match its checkpoint record")
        entry = AdapterEntry(delta=delta, metadata=metadata, path=path)
        self.temp[task_id] = {"delta": delta, "entry": entry, "record": record}
        return self.temp[task_id]

    def phase5_task(self, spec: Any, index: int) -> dict[str, Any]:
        task_id = spec.task_id
        self.runner.ensure_representation(task_id)
        temp = self._temp(task_id)
        candidates = self.repository.entries()

        # 1. Eligibility on VALIDATION, before any TEST read.
        n_val = self._examples("eligibility", task_id)
        zs_val = self._measure(None, task_id, split=SplitName.VALIDATION, n=n_val, label=f"eligibility:{task_id}:zero_shot")
        temp_val = self._measure(temp["delta"], task_id, split=SplitName.VALIDATION, n=n_val,
                                 label=f"eligibility:{task_id}:dedicated")
        eligibility = criteria.eligibility(temp_val["primary_score"], zs_val["primary_score"],
                                           threshold=self.full.eligibility_threshold)

        # 2. Every policy decides (SUPPORT only) and is logged.
        decided = self.decide(task_id=task_id, task_index=index, temp_delta=temp["delta"], candidates=candidates,
                              policies=self.full.policies, log_as="phase5")
        decisions = decided["decisions"]

        # 3. TEST: the measured option set.
        probes = decided["probes"]
        top = criteria.top_k_by_loss_gain(probes["candidates"], probes["backbone"]["loss"], self.full.fuse_top_k)
        chosen_fuse = sorted({d["option"].partition(":")[2] for d in decisions.values()
                              if d.get("option") and d["option"].startswith("FUSE:")})
        fuse_set = list(dict.fromkeys(top + chosen_fuse))
        keys = [criteria.CREATE, criteria.BACKBONE] + [f"REUSE:{e.adapter_id}" for e in candidates] + \
               [f"FUSE:{c}" for c in fuse_set]
        options: dict[str, dict[str, Any]] = {}
        for key in keys:
            unit = f"P5:{task_id}:option:{key}"
            try:
                options[key] = self._unit(unit, lambda k=key: self._measure_option(task_id, k, temp, candidates))
            except Exception as exc:
                self.failures.append({"phase": "P5", "unit": unit, "error": f"{type(exc).__name__}: {exc}"})
                log.exception("option %s for %s failed", key, task_id)
                cleanup(self.backbone.device)
        unmeasured = sorted({e.adapter_id for e in candidates} - set(fuse_set))
        unmeasured_chosen = [p for p, d in decisions.items()
                             if d.get("option") and d["option"].startswith("FUSE:")
                             and d["option"].partition(":")[2] in unmeasured]
        if unmeasured_chosen:
            raise PreconditionError(f"{task_id}: policies {unmeasured_chosen} chose an unmeasured FUSE option")

        tables = {self._cap_label(cap): self._score_task(task_id, decisions, options, candidates, cap)
                  for cap in self.full.caps}
        payload = {
            "task_id": task_id, "kind": spec.kind, "related_to": spec.related_to,
            "temporary_adapter": {k: v for k, v in temp["record"].items() if k != "loss_trajectory"},
            "eligibility": {**eligibility, "examples": n_val, "metric": temp_val["primary_score_name"]},
            "decisions": decisions, "probes": probes, "reuse_gate": decided["gate"],
            "utilities": decided["utilities"], "prescreen": decided["prescreen"], "features": decided["features"],
            "measured_fuse_set": {"top_k_by_probe_loss_gain": top, "chosen_by_policies": chosen_fuse,
                                  "not_measured": unmeasured, "no_policy_chose_an_unmeasured_option": True},
            "options": options,
            "policy_outcomes": tables,
        }
        self._write(f"evaluation_{task_id}.json", payload)
        return {"eligibility": eligibility["status"], "decisions": {p: d.get("option") for p, d in decisions.items()},
                "tables": {cap: {"oracle": t["oracle"]["option"],
                                 "policies": {p: {k: v for k, v in r.items() if k in
                                                  ("executed", "decision_correct", "quality_shortfall",
                                                   "cost_excess", "forgetting", "acceptable")}
                                              for p, r in t["policies"].items()}}
                           for cap, t in tables.items()}}

    @staticmethod
    def _cap_label(cap: int | None) -> str:
        return "no_cap" if cap is None else f"cap_{cap}"

    def _measure_option(self, task_id: str, key: str, temp: Mapping[str, Any],
                        candidates: Sequence[AdapterEntry]) -> dict[str, Any]:
        outcome, _, candidate_id = key.partition(":")
        n = self._examples("test", task_id)
        base_bytes = sum(e.delta.storage_bytes() for e in candidates)
        train_seconds = float(temp["record"]["wall_seconds"])
        option: dict[str, Any] = {"key": key, "outcome": outcome, "candidate_id": candidate_id or None,
                                  "old_task_scores": {}, "old_task_standalone": {}, "old_drops": {}, "fusion": None}
        if outcome == criteria.CREATE:
            new = self._measure(temp["delta"], task_id, split=SplitName.TEST, n=n, label=f"option:{task_id}:{key}")
            option.update(adapters_after=len(candidates) + 1, training_seconds=train_seconds,
                          stored_bytes=base_bytes + temp["delta"].storage_bytes())
        elif outcome == criteria.BACKBONE:
            new = self._measure(None, task_id, split=SplitName.TEST, n=n, label=f"option:{task_id}:{key}")
            option.update(adapters_after=len(candidates), training_seconds=0.0, stored_bytes=base_bytes)
        elif outcome == "REUSE":
            entry = next(e for e in candidates if e.adapter_id == candidate_id)
            new = self._measure(entry.delta, task_id, split=SplitName.TEST, n=n, label=f"option:{task_id}:{key}")
            option.update(adapters_after=len(candidates), training_seconds=0.0, stored_bytes=base_bytes)
        else:
            entry = next(e for e in candidates if e.adapter_id == candidate_id)
            fused, fusion = self._fuse_with(self.selected, entry, temp["delta"], task_id,
                                            f"eval-{task_id}-fuse-{candidate_id}")
            new = self._measure(fused, task_id, split=SplitName.TEST, n=n, label=f"option:{task_id}:{key}")
            for t in entry.metadata.tasks_served:
                post = self._measure(fused, t, split=SplitName.TEST, n=self._examples("test", t),
                                     label=f"option:{task_id}:{key}:old:{t}")["primary_score"]
                pre = self._measure(entry.delta, t, split=SplitName.TEST, n=self._examples("test", t),
                                    label=f"standalone:test:{entry.adapter_id}")["primary_score"]
                option["old_task_scores"][t] = post
                option["old_task_standalone"][t] = pre
                option["old_drops"][t] = criteria.relative_drop(pre, post)
            option.update(adapters_after=len(candidates), training_seconds=train_seconds + fusion["wall_seconds"],
                          stored_bytes=base_bytes - entry.delta.storage_bytes() + fused.storage_bytes(),
                          fusion=fusion)
            del fused
            cleanup(self.backbone.device)
        create_seconds = train_seconds
        option.update(
            new_task_score=new["primary_score"], new_task_metric=new["primary_score_name"],
            new_task_loss=new["mean_token_loss"], metrics=new["metrics"], per_example=new["per_example_primary"],
            predictions_file=new["predictions_file"],
            training_seconds_spent=option["training_seconds"],
            training_seconds_saved_vs_create=create_seconds - option["training_seconds"],
        )
        return option

    def _score_task(self, task_id: str, decisions: Mapping[str, Mapping[str, Any]],
                    options: Mapping[str, Mapping[str, Any]], candidates: Sequence[AdapterEntry],
                    cap: int | None) -> dict[str, Any]:
        """Items 6-7 for one task under one cap setting, from the measured options."""
        at_cap = cap is not None and len(candidates) >= cap
        measures = {k: criteria.OptionMeasure(
            key=k, new_score=o["new_task_score"], old_drops=o["old_drops"], adapters_after=o["adapters_after"],
            training_seconds=o["training_seconds"], stored_bytes=o["stored_bytes"], per_example=o["per_example"])
            for k, o in options.items()}
        if criteria.CREATE not in measures:
            return {"error": "CREATE was not measured; nothing can be judged", "oracle": {"option": None},
                    "policies": {}}
        create = measures[criteria.CREATE]
        oracle = criteria.choose_oracle(measures, at_cap=at_cap, equivalence=self.full.equivalence,
                                        forgetting=self.full.forgetting)
        acceptability = {k: {**criteria.acceptability(m, create, equivalence=self.full.equivalence,
                                                      forgetting=self.full.forgetting,
                                                      resamples=self.full.bootstrap_resamples),
                             "feasible": criteria.feasible(m, at_cap=at_cap)}
                         for k, m in measures.items()}
        # Secondary, run-1 definitions: unweighted mean of the new and every seed task's score.
        standalone = {t: self._measure(e.delta, t, split=SplitName.TEST, n=self._examples("test", t),
                                       label=f"standalone:test:{e.adapter_id}")["primary_score"]
                      for e in candidates for t in e.metadata.tasks_served}
        average = {k: (o["new_task_score"] + sum(o["old_task_scores"].get(t, s) for t, s in standalone.items()))
                   / (1 + len(standalone)) for k, o in options.items()}
        feasible_keys = [k for k in options if acceptability[k]["feasible"]]
        oracle_by_average = max(feasible_keys, key=lambda k: (average[k], -options[k]["adapters_after"]))
        separate = decisions.get("always_separate", {}).get("option")
        separate_key = criteria.executed_option(separate, at_cap=at_cap) if separate else None
        policies = {}
        for name, decision in decisions.items():
            chosen = decision.get("option")
            if chosen is None:
                policies[name] = {"chosen": None, "measured": False, "reason": decision.get("reason")}
                continue
            executed = criteria.executed_option(chosen, at_cap=at_cap)
            if executed not in measures:
                policies[name] = {"chosen": chosen, "executed": executed, "measured": False,
                                  "reason": "option not measured"}
                continue
            quality = criteria.decision_quality(executed, measures, oracle["option"],
                                                equivalence=self.full.equivalence, forgetting=self.full.forgetting)
            ci = None
            if separate_key in measures and executed != separate_key:
                ci = criteria.paired_bootstrap_ci(measures[executed].per_example, measures[separate_key].per_example,
                                                  resamples=self.full.bootstrap_resamples)
            flags = []
            if not quality["acceptable"]:
                flags.append("unacceptable: below 0.98 x CREATE or > 1% forgetting")
            if quality["cost_excess"]["adapters"] > 0 or quality["cost_excess"]["training_seconds"] > 0:
                flags.append("costlier than the oracle")
            policies[name] = {
                "chosen": chosen, "executed": executed, "infeasible_create_replaced": chosen != executed,
                "measured": True, **quality, "flags": flags, "ci_vs_always_separate": ci,
                "new_task_score": measures[executed].new_score,
                "average_performance": average[executed],
                "regret_vs_oracle": average[oracle_by_average] - average[executed],
                "fallback": decision.get("fallback"),
            }
        return {"at_cap": at_cap, "cap": cap, "oracle": oracle, "acceptability": acceptability,
                "average_performance": average, "oracle_option": oracle_by_average,
                "oracle_option_note": "run-1 definition (best average performance), secondary",
                "policies": policies}

    def phase5_evaluation(self) -> dict[str, Any]:
        results = {}
        for index, spec in enumerate(self.study.evaluation_tasks, start=1):
            key = f"P5:{spec.task_id}"
            try:
                results[spec.task_id] = self._unit(key, lambda s=spec, i=index: self.phase5_task(s, i))
            except Exception as exc:
                self.failures.append({"phase": "P5", "unit": key, "error": f"{type(exc).__name__}: {exc}"})
                log.exception("evaluation task %s failed", spec.task_id)
                cleanup(self.backbone.device)
        comparison = self._compare_full(results)
        self._write("policy_comparison.json", comparison)
        return {"tasks": results, "policy_comparison": comparison}

    def _compare_full(self, results: Mapping[str, Any]) -> dict[str, Any]:
        """Policy tables over the eligible tasks, per cap setting (items 7, 10)."""
        eligible = [t for t, r in results.items() if r.get("eligibility") == "eligible"]
        out: dict[str, Any] = {"eligible_tasks": eligible,
                               "non_discriminating_tasks": [t for t in results if t not in eligible]}
        for cap in self.full.caps:
            label = self._cap_label(cap)
            table = {}
            for policy in self.full.policies:
                rows, diffs_a, diffs_b = [], [], []
                for task_id in eligible:
                    payload = json.loads((self.tracker.paths.root / "decision_study" /
                                          f"evaluation_{task_id}.json").read_text())
                    row = payload["policy_outcomes"][label]["policies"].get(policy) or {}
                    rows.append((task_id, row))
                    sep = payload["policy_outcomes"][label]["policies"].get("always_separate") or {}
                    if row.get("measured") and sep.get("measured"):
                        diffs_a += payload["options"][row["executed"]]["per_example"]
                        diffs_b += payload["options"][sep["executed"]]["per_example"]
                measured = [r for _, r in rows if r.get("measured")]

                def mean(values):
                    values = list(values)
                    return sum(values) / len(values) if values else None
                table[policy] = {
                    "decisions": {t: r.get("executed", r.get("chosen")) for t, r in rows},
                    "n_measured": len(measured),
                    "decision_correct": mean(1.0 if r["decision_correct"] else 0.0 for r in measured),
                    "acceptable": mean(1.0 if r["acceptable"] else 0.0 for r in measured),
                    "mean_quality_shortfall": mean(r["quality_shortfall"] for r in measured),
                    "mean_cost_excess_adapters": mean(r["cost_excess"]["adapters"] for r in measured),
                    "mean_cost_excess_training_seconds": mean(r["cost_excess"]["training_seconds"] for r in measured),
                    "mean_forgetting": mean(r["forgetting"] for r in measured),
                    "max_forgetting": max((r["forgetting"] for r in measured), default=None),
                    "mean_average_performance": mean(r["average_performance"] for r in measured),
                    "mean_regret_vs_oracle": mean(r["regret_vs_oracle"] for r in measured),
                    "pooled_ci_vs_always_separate": (criteria.paired_bootstrap_ci(
                        diffs_a, diffs_b, resamples=self.full.bootstrap_resamples)
                        if diffs_a and policy != "always_separate" else None),
                }
            out[label] = table
        out["caveat"] = ("one seed: CIs resample TEST examples only; scores pooled across tasks mix metrics "
                         "and are descriptive")
        return out

    # -- phase 6: sequential stream -----------------------------------------------
    def phase6_stream(self) -> dict[str, Any]:
        from amaun.continual.sequential_stream import run_stream

        runs = {}
        for policy in self.full.stream["policies"]:
            for cap in self.full.stream["caps"]:
                name = f"{policy}_{self._cap_label(cap)}"
                key = f"P6:{name}"
                try:
                    runs[name] = self._unit(key, lambda p=policy, c=cap, n=name: run_stream(self, policy=p, cap=c, name=n))
                except Exception as exc:
                    self.failures.append({"phase": "P6", "unit": key, "error": f"{type(exc).__name__}: {exc}"})
                    log.exception("stream run %s failed", name)
                    cleanup(self.backbone.device)
                    continue
                self._write(f"stream_{name}.json", runs[name])
        return {name: {k: r[k] for k in ("policy", "cap", "fallback", "decision_counts", "final_repository_size",
                                         "final_repository_bytes", "mean_final_score")}
                for name, r in runs.items()}

    # -- phase 7: memory audit and repeatability ----------------------------------
    def phase7_audit(self) -> dict[str, Any]:
        new_id, old_id = self.full.fusion_pair_adapters[0], self.full.fusion_pair_adapters[1]
        new, old = self.repository.retrieve(new_id), self.repository.retrieve(old_id)
        task_id = new.metadata.tasks_served[0]
        n = self.study.memory_cycle_eval_examples
        post_cleanup, cycles = [], []
        for cycle in range(self.study.memory_cycles):
            self.monitor.record("cycle:start", cycle=cycle)
            standalone = self._evaluate(new.delta, self.tasks[task_id], utility_guard(self.splits[task_id]),
                                        split=SplitName.VALIDATION, n=n)
            fused, fusion = self._fuse_with(self.selected, old, new.delta, task_id, f"cycle{cycle}-fused")
            self.monitor.record("cycle:fused", cycle=cycle)
            result = self._evaluate(fused, self.tasks[task_id], utility_guard(self.splits[task_id]),
                                    split=SplitName.VALIDATION, n=n)
            cycles.append({"cycle": cycle, "standalone_score": standalone.primary_score,
                           "fused_score": result.primary_score, "fused_content_hash": fused.content_hash(),
                           "fusion_wall_seconds": fusion["wall_seconds"],
                           "peak_allocated_bytes": fusion["peak_allocated_bytes"]})
            del fused
            cleanup(self.backbone.device)
            post_cleanup.append(self.monitor.record("cycle:after_cleanup", cycle=cycle, count_tensors=True))
        report = stability_report(post_cleanup, allocated_tolerance_bytes=self.study.allocated_tolerance_bytes,
                                  tensor_tolerance=self.study.tensor_tolerance) if post_cleanup else {"verdict": "not_run"}
        scores = [c["fused_score"] for c in cycles]
        report.update({
            "cycle_definition": "load adapter -> evaluate -> fuse+KD (selected config) -> evaluate fused -> release",
            "variant": self.selected.name if self.selected else None, "cycles": cycles,
            "fused_score_spread": (max(scores) - min(scores)) if scores else None,
            "identical_fused_adapters": len({c["fused_content_hash"] for c in cycles}) == 1 if cycles else None,
        })
        self._write("memory_audit.json", report)
        return {k: report.get(k) for k in ("verdict", "fused_score_spread", "identical_fused_adapters")}

    # -- orchestration ------------------------------------------------------------
    def _restore(self, upto: str) -> None:
        """After a resume, rebuild the in-memory state that completed phases left behind."""
        if self.ckpt.has_unit("phase:P2"):
            self.selected = self.full.variant(self.ckpt.unit("phase:P2")["chosen"])
        if self.ckpt.phase_done("P4"):
            self._restore_frozen()
        for assignment in self.study.initial_adapters:
            self.runner.ensure_representation(assignment.task_id)

    def run(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "run_mode": self.cfg.run_mode.value, "is_reportable": self.cfg.run_mode.is_reportable,
            "experiment_id": self.cfg.experiment_id, "seed": self.cfg.seed, "config_hash": self.cfg.config_hash,
            "backbone": self.cfg.backbone.model_id, "time_budget_hours": self.study.time_budget_hours,
            "assignments": {"initial_adapters": [a.__dict__ for a in self.study.initial_adapters],
                            "evaluation_tasks": [t.__dict__ for t in self.study.evaluation_tasks]},
        }
        self._write("assignments.json", summary["assignments"])
        self._restore("all")
        steps = (("P0", self.phase0_preflight), ("P1", self.phase1_seeds), ("P2", self.phase2_fusion_selection),
                 ("P3", self.phase3_supervision), ("P4", self.phase4_utility), ("P5", self.phase5_evaluation),
                 ("P6", self.phase6_stream), ("P7", self.phase7_audit))
        for phase, function in steps:
            if self.ckpt.phase_done(phase):
                summary[phase] = self.ckpt.unit(f"phase:{phase}")
                continue
            if not self.clock.allows(phase):
                summary["aborted"] = f"time budget exhausted before {phase}"
                break
            self.monitor.record(f"{phase}:start", count_tensors=True)
            start = time.perf_counter()
            result = function()
            seconds = time.perf_counter() - start
            self.clock.record(phase, seconds)
            # Wall time per phase across sessions (a resumed phase adds its remaining time).
            spent = self.ckpt.state.setdefault("phase_seconds", {})
            spent[phase] = spent.get(phase, 0.0) + seconds
            summary[phase] = result
            if phase == "P0" and not result["ok"]:
                summary["aborted"] = "preflight failed: see decision_study/preflight_full.json; nothing was run"
                break
            self.ckpt.save_unit(f"phase:{phase}", result, elapsed=self._elapsed())
            self.ckpt.mark_phase(phase, elapsed=self._elapsed())
        if "aborted" not in summary:
            assert_fingerprint_unchanged(self.backbone.model, self.backbone.fingerprint, phase="full_study")
            summary["backbone_fingerprint_unchanged"] = True
        projection = (self.ckpt.unit("phase:P0") or summary.get("P0") or {}).get("time_projection") or {}
        summary["projected_vs_actual_hours"] = {
            phase: {"projected": (projection.get("per_phase_hours") or {}).get(phase),
                    "actual": self.ckpt.state.get("phase_seconds", {}).get(phase, 0.0) / 3600.0}
            for phase in PHASES}
        summary["resumes"] = self.ckpt.resumes
        self._finish(summary)
        return summary

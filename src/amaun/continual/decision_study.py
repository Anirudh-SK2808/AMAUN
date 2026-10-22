"""The L4 decision study: is the learned utility network a better integration decision-maker
than simple policies, on similar-task arrivals?

Phases, in order (``configs/experiments/l4_decision_study.yaml``):

0. **Preflight.**  One worst-case training step, fusion step and generation per task at the
   configured batch size, to surface an OOM *before* hours of work.  A failure is reported with
   the exact error and the smallest adjustment; no setting is changed automatically.
1. **Initial adapters.**  Each ``decision_study.initial_adapters`` entry is trained on its own
   task's TRAIN split from theta_0 and registered.  The task-to-adapter assignment, the training
   rows' content hashes, and any data shared between adapters are recorded.
2. **Utility supervision.**  For every ordered pair ``(new = A_i, old = A_j)`` of initial
   adapters, the integration is actually performed (layer-wise fusion + KD) and measured on
   VALIDATION.  Targets use the existing definition (``manager.targets.measure_outcome_targets``,
   loss-based proxy scores).  The normaliser, ``g_omega`` and ``tau`` are fitted on these
   measurements only, then frozen.  The fixed-rule baseline's threshold is selected on the
   same measurements, so both learned and fixed rules see identical evidence.
3. **Evaluation tasks.**  Each evaluation task trains its own temporary adapter, then every
   policy decides against the *initial* repository with frozen parameters.  Afterwards every
   option (CREATE, REUSE_j, FUSE_j) is measured on TEST, so each policy's choice is scored on
   real outcomes and compared with the others.  Evaluation tasks are independent: none sees
   another's integration.
4. **Memory audit.**  Repeated load -> fuse -> evaluate -> cleanup cycles with allocator and
   live-tensor snapshots, judged by :func:`amaun.utils.resources.stability_report`.

Leakage: phases 1-2 read TRAIN / VALIDATION / SUPPORT only.  TEST is read in phase 1 (standalone
scores, for reporting) and phase 3 (after every decision is frozen and logged).  Nothing measured
on TEST is ever passed back into a fit, a threshold, or a decision.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.lora import extract_delta, load_delta_into, zero_delta_
from amaun.adapters.metadata import AdapterOrigin
from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.baselines.strategies import (
    AlwaysMergeStrategy,
    AmaunStrategy,
    SeparateLoRAStrategy,
    SimilarityConfig,
    SimilaritySelectionStrategy,
)
from amaun.config import ResolvedConfig, RunMode
from amaun.continual.runner import ContinualRunner
from amaun.data.loaders import BatchPlan, GuardedLoader
from amaun.data.preprocess import IGNORE_INDEX, collate, masked_cross_entropy, tokenize_example
from amaun.data.registry import TaskSpec
from amaun.data.splits import Example, SplitName, TaskSplits, evaluator_guard, trainer_guard, utility_guard
from amaun.errors import PreconditionError
from amaun.evaluation.evaluator import Evaluator
from amaun.features.extractor import FeatureExtractor
from amaun.fusion.optimize import run_fusion_kd
from amaun.manager.bootstrap import _loss_to_score, _validation_loss, fit_utility_model
from amaun.manager.decision import Decision, Outcome
from amaun.manager.reuse_gate import (
    GateExample,
    ReuseGate,
    ReuseGateConfig,
    gate_reuse_decision,
    reuse_gap,
)
from amaun.manager.targets import SupervisionExample, SupervisionSet, measure_outcome_targets
from amaun.manager.threshold import select_thresholds
from amaun.manager.utility_network import OUTCOME_HEADS
from amaun.models.backbone import (
    LoadedBackbone,
    _is_adapter_parameter,
    assert_backbone_frozen,
    assert_fingerprint_unchanged,
)
from amaun.models.generation import GenerationConfig
from amaun.training.lora_trainer import train_adapter
from amaun.utils.logging import get_logger
from amaun.utils.resources import ResourceMonitor, cleanup, memory_snapshot, stability_report
from amaun.utils.tracking import ExperimentTracker

log = get_logger("continual.decision_study")

# ``amaun`` follows manager.decision_order (with prescreen_then_train: REUSE gate before training,
# else the train-first decision).  ``amaun_train_first`` is always the train-first decision with the
# same utility network, so the two orders are compared on identical measured options.
POLICIES = ("amaun", "amaun_train_first", "always_merge", "always_separate", "fixed_rule")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterAssignment:
    adapter_id: str
    task_id: str
    role: str


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    kind: str                  # similar | different
    related_to: str | None     # adapter id the task is expected to relate to, for reporting only


@dataclass
class StudyConfig:
    initial_adapters: list[AdapterAssignment]
    evaluation_tasks: list[EvaluationTask]
    policies: list[str]
    required_backbone: str | None
    time_budget_hours: float
    supervision_generation_examples: int
    supervision_loss_examples: int
    retention_split: SplitName
    outcome_tolerance: float
    min_supervision_examples: int
    evaluate_zero_shot: bool
    measure_all_options: bool
    preflight: bool
    memory_cycles: int
    memory_cycle_fusion_steps: int
    memory_cycle_eval_examples: int
    allocated_tolerance_bytes: int
    tensor_tolerance: int
    merge_feature: str
    merge_feature_higher_is_more_similar: bool

    @classmethod
    def from_config(cls, cfg: ResolvedConfig) -> "StudyConfig":
        section = cfg.section("decision_study")
        adapters = [
            AdapterAssignment(str(a["adapter_id"]), str(a["task"]), str(a.get("role", "")))
            for a in section["initial_adapters"]
        ]
        if len({a.adapter_id for a in adapters}) != len(adapters):
            raise PreconditionError("decision_study.initial_adapters has duplicate adapter ids")
        tasks = [
            EvaluationTask(str(t["task"]), str(t["kind"]), t.get("related_to"))
            for t in (section.get("evaluation_tasks") or [])
        ]
        for task in tasks:
            if task.kind not in ("similar", "different"):
                raise PreconditionError(f"evaluation task '{task.task_id}' kind must be similar|different")
        initial_tasks = {a.task_id for a in adapters}
        overlap = initial_tasks & {t.task_id for t in tasks}
        if overlap:
            raise PreconditionError(
                f"evaluation task(s) {sorted(overlap)} are also initial-adapter tasks; an evaluation "
                f"task must be held out from utility supervision"
            )
        policies = list(section.get("policies") or POLICIES)
        unknown = [p for p in policies if p not in POLICIES]
        if unknown:
            raise PreconditionError(f"unknown policies {unknown}; known: {list(POLICIES)}")
        memory = section.get("memory_audit") or {}
        merge = section.get("always_merge") or {}
        return cls(
            initial_adapters=adapters,
            evaluation_tasks=tasks,
            policies=policies,
            required_backbone=section.get("required_backbone"),
            time_budget_hours=float(section["time_budget_hours"]),
            supervision_generation_examples=int(section["supervision_generation_examples"]),
            supervision_loss_examples=int(section["supervision_loss_examples"]),
            retention_split=SplitName(section["retention_split"]),
            outcome_tolerance=float(section["outcome_tolerance"]),
            min_supervision_examples=int(section["min_supervision_examples"]),
            evaluate_zero_shot=bool(section.get("evaluate_zero_shot", True)),
            measure_all_options=bool(section.get("measure_all_options", True)),
            preflight=bool(section.get("preflight", True)),
            memory_cycles=int(memory.get("cycles", 0)),
            memory_cycle_fusion_steps=int(memory.get("fusion_steps", 2)),
            memory_cycle_eval_examples=int(memory.get("eval_examples", 4)),
            allocated_tolerance_bytes=int(float(memory.get("allocated_tolerance_mb", 32)) * 2**20),
            tensor_tolerance=int(memory.get("tensor_tolerance", 8)),
            merge_feature=str(merge.get("feature", "semantic.d_task_sq")),
            merge_feature_higher_is_more_similar=bool(merge.get("higher_is_more_similar", False)),
        )


class _PlaceholderStrategy:
    """Lets a ContinualRunner be built for its probes and guards; it never decides."""

    strategy_id = "placeholder"
    requires_utility_network = False

    def score(self, vectors, *, run_mode):
        return []

    def decide(self, **kwargs):
        raise RuntimeError("the decision study makes decisions through explicit policies")


class _Clock:
    """Wall-clock budget: phases that would start after the deadline are skipped and recorded."""

    def __init__(self, hours: float) -> None:
        self.start = time.perf_counter()
        self.budget = hours * 3600.0
        self.phases: dict[str, float] = {}
        self.skipped: list[dict[str, str]] = []

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def allows(self, what: str) -> bool:
        if self.elapsed < self.budget:
            return True
        self.skipped.append({"item": what, "reason": "time_budget_exhausted",
                             "elapsed_seconds": f"{self.elapsed:.0f}"})
        log.error("time budget exhausted before '%s' (%.0fs elapsed); skipping", what, self.elapsed)
        return False

    def record(self, phase: str, seconds: float) -> None:
        self.phases[phase] = self.phases.get(phase, 0.0) + seconds



def worst_case_examples(tokenizer: Any, task: TaskSpec, examples: Sequence[Example],
                        count: int) -> tuple[list[Example], Example]:
    """The ``count`` examples with the most tokens, and the one with the longest prompt.

    Both are measured after the task's own truncation, i.e. on what a batch actually allocates.
    Character length is not a proxy for that: ``middle`` truncation caps every long prompt at
    ``max_input_tokens``, so the longest-by-characters rows can be shorter in tokens than rows
    with a short prompt and a long target (FinQA: 774 vs 884 tokens at batch 4).  Ties keep
    the split's order, so the selection is deterministic.
    """
    lengths = []
    for example in examples:
        input_ids, labels = tokenize_example(
            tokenizer, example, max_input_tokens=task.max_input_tokens,
            max_target_tokens=task.max_target_tokens, prompt_truncation=task.prompt_truncation,
        )
        lengths.append((len(input_ids), sum(1 for label in labels if label == IGNORE_INDEX)))
    order = sorted(range(len(examples)), key=lambda i: lengths[i][0], reverse=True)
    longest_prompt = max(range(len(examples)), key=lambda i: lengths[i][1])
    return [examples[i] for i in order[:count]], examples[longest_prompt]


# --------------------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------------------


@dataclass
class _Option:
    """One measured integration option for an evaluation task."""

    key: str
    outcome: str
    candidate_id: str | None
    new_task_score: float | None = None
    new_task_loss: float | None = None
    old_task_scores: dict[str, float] = field(default_factory=dict)
    old_task_losses: dict[str, float | None] = field(default_factory=dict)
    fused_bytes: int | None = None
    integration_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    adapters_retained: int = 0
    repository_bytes: int = 0
    error: str | None = None
    fusion: dict[str, Any] | None = None


class DecisionStudy:
    def __init__(
        self,
        *,
        cfg: ResolvedConfig,
        backbone: LoadedBackbone,
        tasks: Sequence[TaskSpec],
        splits: Mapping[str, TaskSplits],
        tracker: ExperimentTracker,
    ) -> None:
        self.cfg = cfg
        self.study = StudyConfig.from_config(cfg)
        self.backbone = backbone
        self.tasks = {task.task_id: task for task in tasks}
        self.splits = dict(splits)
        self.tracker = tracker
        self.clock = _Clock(self.study.time_budget_hours)
        self.monitor = ResourceMonitor(backbone.device, tracker.paths.root / "decision_study" / "resources.jsonl")
        self.failures: list[dict[str, Any]] = []

        missing = [a.task_id for a in self.study.initial_adapters if a.task_id not in self.tasks]
        missing += [t.task_id for t in self.study.evaluation_tasks if t.task_id not in self.tasks]
        if missing:
            raise PreconditionError(f"decision study references tasks with no data: {sorted(set(missing))}")
        if (self.study.required_backbone and cfg.run_mode is not RunMode.DUMMY
                and cfg.backbone.model_id != self.study.required_backbone):
            raise PreconditionError(
                f"decision_study.required_backbone is {self.study.required_backbone} but run_mode "
                f"'{cfg.run_mode.value}' resolves to {cfg.backbone.model_id}; refusing to mix tiers"
            )

        self.repository = AdapterRepository(
            tracker.paths.adapters, backbone_fingerprint=backbone.fingerprint,
            journal_path=tracker.paths.repository_events,
        )
        generation = GenerationConfig.from_config(cfg.section("generation"))
        batch_size = int(cfg.get("training.batch_size"))
        self.final_evaluator = Evaluator(
            backbone, generation, max_examples=int(cfg.get("evaluation.max_test_examples")),
            batch_size=batch_size,
        )
        self.check_evaluator = Evaluator(
            backbone, generation, max_examples=self.study.supervision_generation_examples,
            batch_size=batch_size,
        )
        self.runner = ContinualRunner(
            cfg=cfg, backbone=backbone, tasks=tasks, splits=splits, repository=self.repository,
            strategy=_PlaceholderStrategy(), extractor=FeatureExtractor(cfg.section("features")),
            evaluator=self.final_evaluator, tracker=tracker, arm_id="decision_study",
        )
        self.peft_model = self.runner.peft_model
        self._test_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._val_loss_cache: dict[tuple[str, str], float] = {}
        self._val_gen_cache: dict[tuple[str, str], float] = {}

    # -- small measured primitives ----------------------------------------------
    def _write(self, name: str, payload: Any) -> None:
        self.tracker.write_json(f"decision_study/{name}", payload)

    def _evaluate(self, delta: AdapterDelta | None, task_id: str, *, split: SplitName,
                  evaluator: Evaluator, label: str) -> dict[str, Any]:
        task = self.tasks[task_id]
        if delta is None:
            zero_delta_(self.peft_model)
        else:
            load_delta_into(self.peft_model, delta)
        guard = evaluator_guard(self.splits[task_id]) if split is SplitName.TEST else utility_guard(self.splits[task_id])
        self.monitor.record(f"eval:{label}:before")
        result = evaluator.evaluate(task, guard, model=self.peft_model,
                                    adapter_id=None if delta is None else delta.adapter_id, split=split)
        self.monitor.record(f"eval:{label}:after")
        record = {
            "task_id": task_id, "adapter_id": result.adapter_id, "split": split.value,
            "primary_score": result.primary_score, "primary_score_name": result.primary_score_name,
            "metrics": result.metrics, "mean_token_loss": result.mean_token_loss,
            "n_examples": result.n_examples, "wall_seconds": result.wall_seconds,
            "sample_predictions": result.sample_predictions,
        }
        self.tracker.log_metric(phase=f"decision_study:{label}", **{k: v for k, v in record.items()
                                                                  if k != "sample_predictions"})
        return record

    def _test_score(self, delta: AdapterDelta | None, task_id: str, label: str) -> dict[str, Any]:
        key = (delta.adapter_id if delta is not None else "<zero-shot>", task_id)
        if key not in self._test_cache:
            self._test_cache[key] = self._evaluate(delta, task_id, split=SplitName.TEST,
                                                   evaluator=self.final_evaluator, label=label)
        return self._test_cache[key]

    def _val_loss(self, delta: AdapterDelta, task_id: str) -> float:
        key = (delta.adapter_id, task_id)
        if key not in self._val_loss_cache:
            task = self.tasks[task_id]
            self._val_loss_cache[key] = _validation_loss(
                self.peft_model, delta, task, self.splits[task_id], self.backbone.tokenizer,
                self.backbone.device, max_examples=self.study.supervision_loss_examples,
                batch_size=int(self.cfg.get("training.batch_size")),
            )
        return self._val_loss_cache[key]

    def _val_generation(self, delta: AdapterDelta, task_id: str) -> float:
        key = (delta.adapter_id, task_id)
        if key not in self._val_gen_cache:
            self._val_gen_cache[key] = self._evaluate(
                delta, task_id, split=SplitName.VALIDATION, evaluator=self.check_evaluator,
                label=f"check:{delta.adapter_id}:{task_id}",
            )["primary_score"]
        return self._val_gen_cache[key]

    def _fuse(self, old: AdapterEntry, new_delta: AdapterDelta, new_task_id: str, fused_id: str,
              *, steps: int | None = None) -> tuple[AdapterDelta, dict[str, Any]]:
        """Layer-wise fusion + KD of ``(A_old = old, A_temp = new_delta)`` -- Algorithm 2."""
        new_task = self.tasks[new_task_id]
        fusion_cfg = self.runner.fusion_cfg
        if steps is not None:
            fusion_cfg = type(fusion_cfg)(**{**fusion_cfg.__dict__, "steps": steps})
        loader = GuardedLoader(
            trainer_guard(self.splits[new_task_id]), SplitName.TRAIN, new_task, self.backbone.tokenizer,
            BatchPlan(batch_size=self.runner.train_cfg.batch_size, shuffle=True, seed=self.cfg.seed),
        )
        # islice: only the batches fusion will use are tokenised (list(loader) collated all 1 500).
        task_batches = list(itertools.islice(iter(loader), max(1, fusion_cfg.steps)))
        retention = self._retention_batches(old)
        self.monitor.record(f"fuse:{fused_id}:before")
        fused, record = run_fusion_kd(
            backbone=self.backbone, peft_model=self.peft_model, components=[old.delta, new_delta],
            task_batches=task_batches, retention_batches=retention, fusion_cfg=fusion_cfg,
            kd_cfg=self.runner.kd_cfg, fused_adapter_id=fused_id, default_rank=self.runner.lora_spec.r,
        )
        del task_batches, retention
        cleanup(self.backbone.device)
        self.monitor.record(f"fuse:{fused_id}:after")
        payload = record.as_dict()
        self.tracker.write_json(f"fusion_logs/{fused_id}.json",
                                {**payload, "alpha_summary": record.alpha_summary()})
        return fused, {
            "fused_adapter_id": fused_id, "steps": record.steps, "wall_seconds": record.wall_seconds,
            "kd": record.kd, "alpha_summary": record.alpha_summary(),
            "alpha_moved_from_init": record.alpha_moved_from_init,
            "final_loss": record.loss_trajectory[-1] if record.loss_trajectory else None,
        }

    def _retention_batches(self, old: AdapterEntry) -> list:
        """KD ``D_rep``: the old adapter's own tasks, from ``decision_study.retention_split``.

        TRAIN by default, so VALIDATION -- on which phase 2 measures old-task retention -- is
        never also the data KD was optimised on.
        """
        limit = int(self.runner.kd_cfg.retention_examples)
        tasks = [t for t in old.metadata.tasks_served if t in self.tasks]
        if not tasks or limit <= 0:
            return []
        per_task = max(1, -(-limit // len(tasks)))
        batches = []
        for task_id in tasks:
            task = self.tasks[task_id]
            examples = list(trainer_guard(self.splits[task_id]).get(self.study.retention_split))[:per_task]
            for start in range(0, len(examples), 2):
                batches.append(collate(
                    self.backbone.tokenizer, examples[start : start + 2],
                    max_input_tokens=task.max_input_tokens, max_target_tokens=task.max_target_tokens,
                    prompt_truncation=task.prompt_truncation,
                ))
        return batches

    # -- phase 0 -----------------------------------------------------------------
    def preflight(self) -> dict[str, Any]:
        """Worst-case memory at the configured batch size; report, never adjust.

        Per task, on the longest TRAIN examples by token count (:func:`worst_case_examples`):
        one LoRA forward+backward at ``training.batch_size`` with ``training.gradient_checkpointing``
        applied as ``train_adapter`` applies it, one fusion+KD step (two freshly initialised
        deltas, so it runs before any adapter exists; fusion is never checkpointed), and one
        greedy generation of up to ``max_new_tokens`` from the longest prompt.
        """
        device = self.backbone.device
        batch_size = self.runner.train_cfg.batch_size
        checkpointing = self.runner.train_cfg.gradient_checkpointing
        task_ids = [a.task_id for a in self.study.initial_adapters] + [t.task_id for t in self.study.evaluation_tasks]
        report: dict[str, Any] = {"batch_size": batch_size,
                                  "gradient_checkpointing": checkpointing,
                                  "selection": "longest TRAIN examples by token count after truncation",
                                  "checks": [], "ok": True}
        halve = (f"training.batch_size: {max(1, batch_size // 2)} with training.gradient_accumulation_steps: "
                 f"{self.runner.train_cfg.gradient_accumulation_steps * 2}")
        adjustments = {
            "train_step": ("set training.gradient_checkpointing: true (same values, slower training); or "
                           f"keep the effective batch and halve the micro-batch: {halve}") if not checkpointing
                          else f"keep the effective batch and halve the micro-batch: {halve}",
            "fusion_step": (f"fusion is not checkpointed and has no accumulation: {halve} would fit it, but "
                            "also halves the examples each fusion step sees"),
            "generation": "generation runs one example at a time; reduce the task's max_input_tokens",
        }
        fresh = extract_delta(self.peft_model, adapter_id="preflight-old")
        other = AdapterDelta(layers=fresh.layers, adapter_id="preflight-new")
        fusion_cfg = type(self.runner.fusion_cfg)(**{**self.runner.fusion_cfg.__dict__, "steps": 1})
        for task_id in dict.fromkeys(task_ids):
            task = self.tasks[task_id]
            train = list(trainer_guard(self.splits[task_id]).get(SplitName.TRAIN))
            longest, longest_prompt = worst_case_examples(self.backbone.tokenizer, task, train, batch_size)

            def batch_of(examples):
                return collate(self.backbone.tokenizer, examples,
                               max_input_tokens=task.max_input_tokens,
                               max_target_tokens=task.max_target_tokens,
                               prompt_truncation=task.prompt_truncation)

            def run_check(check: str) -> int:
                """One check; returns the sequence length used.  Every tensor is local to this
                frame, so after an OOM the traceback (and with it these tensors) is released when
                the ``except`` clause ends -- before the cache is emptied for the next check."""
                if check == "train_step":
                    batch = batch_of(longest).to(device)
                    self.peft_model.train()
                    if checkpointing:   # the same calls as train_adapter, so this measures training
                        self.peft_model.gradient_checkpointing_enable()
                        if hasattr(self.peft_model, "enable_input_require_grads"):
                            self.peft_model.enable_input_require_grads()
                        assert_backbone_frozen(self.peft_model)
                    try:
                        logits = self.peft_model(input_ids=batch.input_ids,
                                                 attention_mask=batch.attention_mask).logits
                        masked_cross_entropy(logits, batch.labels).backward()
                    finally:
                        if checkpointing and hasattr(self.backbone.model, "gradient_checkpointing_disable"):
                            self.backbone.model.gradient_checkpointing_disable()
                    self.peft_model.zero_grad(set_to_none=True)
                    self.peft_model.eval()
                    return int(batch.input_ids.shape[1])
                if check == "fusion_step":
                    task_batch = batch_of(longest)
                    run_fusion_kd(
                        backbone=self.backbone, peft_model=self.peft_model,
                        components=[fresh, other], task_batches=[task_batch],
                        retention_batches=[batch_of(longest[:2])], fusion_cfg=fusion_cfg,
                        kd_cfg=self.runner.kd_cfg, fused_adapter_id=f"preflight-{task_id}",
                        default_rank=self.runner.lora_spec.r,
                    )
                    return int(task_batch.input_ids.shape[1])
                zero_delta_(self.peft_model)
                with torch.inference_mode():
                    self.final_evaluator._generate_one(self.peft_model, task, longest_prompt)
                input_ids, labels = tokenize_example(
                    self.backbone.tokenizer, longest_prompt, max_input_tokens=task.max_input_tokens,
                    max_target_tokens=task.max_target_tokens, prompt_truncation=task.prompt_truncation)
                return sum(1 for label in labels if label == IGNORE_INDEX)

            for check in ("train_step", "fusion_step", "generation"):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                failed = False
                try:
                    seq_len = run_check(check)
                    peak = memory_snapshot(device, label=f"preflight:{task_id}:{check}")
                    report["checks"].append({"task_id": task_id, "check": check, "ok": True,
                                             "batch_size": len(longest), "sequence_length": seq_len,
                                             "peak_allocated_bytes": peak.get("gpu_peak_allocated_bytes"),
                                             "peak_reserved_bytes": peak.get("gpu_peak_reserved_bytes")})
                except torch.OutOfMemoryError as exc:   # also catches torch.cuda.OutOfMemoryError
                    failed = True
                    report["ok"] = False
                    report["checks"].append({
                        "task_id": task_id, "check": check, "ok": False,
                        "batch_size": len(longest),
                        "error": f"{type(exc).__name__}: {exc}",
                        "smallest_adjustment": adjustments[check],
                    })
                finally:
                    self.peft_model.zero_grad(set_to_none=True)
                    self.peft_model.eval()
                    # After an OOM, return cached blocks so the next check is measured cleanly
                    # rather than failing on the previous check's fragments.
                    cleanup(device, release_cache=failed)
                if failed:
                    break
        del fresh, other
        self._write("preflight.json", report)
        return report

    # -- phase 1 -----------------------------------------------------------------
    def train_initial_adapters(self) -> list[dict[str, Any]]:
        records = []
        train_ids: dict[str, set[str]] = {}
        for assignment in self.study.initial_adapters:
            if not self.clock.allows(f"train:{assignment.adapter_id}"):
                continue
            task = self.tasks[assignment.task_id]
            self.runner.ensure_representation(task.task_id)
            start = time.perf_counter()
            delta, record = train_adapter(
                backbone=self.backbone, task=task, guard=trainer_guard(self.splits[task.task_id]),
                lora_spec=self.runner.lora_spec, train_cfg=self.runner.train_cfg, resolved=self.cfg,
                adapter_id=assignment.adapter_id, budget=self.runner.budget,
            )
            entry = AdapterEntry(delta=delta, metadata=self.runner._metadata_for(
                assignment.adapter_id, task, delta, record.as_dict(), AdapterOrigin.TRAINED, [task.task_id],
            ))
            registered = self.repository.register(entry, bootstrap=True, task_index=0)
            self.clock.record("train_initial_adapters", time.perf_counter() - start)
            train_ids[assignment.adapter_id] = self.splits[task.task_id].hashes(SplitName.TRAIN)
            records.append({
                "adapter_id": assignment.adapter_id, "task_id": task.task_id, "role": assignment.role,
                "dataset": task.hf_id, "dataset_revision": task.hf_revision,
                "train_rows": len(self.splits[task.task_id][SplitName.TRAIN]),
                "steps": record.steps, "epochs_completed": record.epochs_completed,
                "final_train_loss": record.final_train_loss,
                "loss_trajectory": record.loss_trajectory,
                "validation_trajectory": record.validation_trajectory,
                "wall_seconds": record.wall_seconds, "tokens_seen": record.tokens_seen,
                "peak_memory_bytes": record.peak_memory_bytes, "memory": record.memory,
                "checkpoint": str(registered.path) if registered.path else None,
            })
        # Which adapters share training rows: a content-hash intersection, not an assumption.
        for item in records:
            mine = train_ids[item["adapter_id"]]
            item["shares_training_rows_with"] = {
                other: len(mine & ids) for other, ids in train_ids.items()
                if other != item["adapter_id"] and mine & ids
            }
        self._write("initial_adapters.json", records)

        standalone = {}
        for assignment in self.study.initial_adapters:
            if assignment.adapter_id not in self.repository:
                continue
            if not self.clock.allows(f"test:{assignment.adapter_id}"):
                continue
            entry = self.repository.retrieve(assignment.adapter_id)
            standalone[assignment.adapter_id] = self._test_score(
                entry.delta, assignment.task_id, label=f"standalone:{assignment.adapter_id}")
        self._write("standalone_test_scores.json", standalone)
        return records

    # -- phase 2 -----------------------------------------------------------------
    def utility_supervision(self) -> dict[str, Any]:
        entries = {a.adapter_id: self.repository.retrieve(a.adapter_id)
                   for a in self.study.initial_adapters if a.adapter_id in self.repository}
        supervision = SupervisionSet(fit_split="utility_train")
        measurements = []
        gate_section = self.cfg.section("utility").get("reuse_gate")
        gate_cfg = ReuseGateConfig.from_config(gate_section) if gate_section else None
        gate_examples: list[GateExample] = []
        score_scale = float(self.cfg.get("utility.supervision.loss_to_score_scale"))
        for index, (new_id, old_id) in enumerate(itertools.permutations(sorted(entries), 2)):
            if not self.clock.allows(f"supervision:{new_id}->{old_id}"):
                continue
            new, old = entries[new_id], entries[old_id]
            new_task, old_task = new.metadata.tasks_served[0], old.metadata.tasks_served[0]
            start = time.perf_counter()
            try:
                contexts = self.runner._build_contexts(new_task, index, new.delta, candidates=[old])
                vector = self.runner.extractor.extract(contexts[old_id])
                del contexts
                # Stage-1 gate evidence for the same pair: A_j's temp-free features on task i's
                # SUPPORT, labelled with the measured VALIDATION gap between reusing A_j and the
                # dedicated adapter A_i.  Same feature function as at decision time.
                gate_example = None
                if gate_cfg is not None:
                    gate_example = GateExample(
                        task_id=new_task, candidate_id=old_id,
                        features=self.runner.reuse_gate_features(
                            new_task, [old], probe_examples=gate_cfg.probe_examples)[old_id],
                        gap=reuse_gap(self._val_loss(old.delta, new_task), self._val_loss(new.delta, new_task),
                                      scale=score_scale),
                    )
                fused, fusion = self._fuse(old, new.delta, new_task, f"sup-{new_id}-into-{old_id}")
                pre_new = self._val_loss(new.delta, new_task)
                pre_old = self._val_loss(old.delta, old_task)
                post_new = self._val_loss(fused, new_task)
                post_old = self._val_loss(fused, old_task)
                forgetting = max(0.0, _loss_to_score(pre_old, scale=score_scale)
                                 - _loss_to_score(post_old, scale=score_scale))
                targets, beneficial = measure_outcome_targets(
                    primary_score_post=_loss_to_score(post_new, scale=score_scale),
                    forgetting_post=forgetting,
                    confidence_post=_loss_to_score(post_new, scale=score_scale),
                    memory_bytes_post=fused.storage_bytes(),
                    communication_bytes_post=fused.storage_bytes(),
                    long_term_loss_post=post_new if post_new == post_new else 0.0,
                    primary_score_baseline=_loss_to_score(pre_new, scale=score_scale),
                )
                generation_check = {
                    "examples": self.study.supervision_generation_examples,
                    "new_task_standalone": self._val_generation(new.delta, new_task),
                    "new_task_fused": self._val_generation(fused, new_task),
                    "old_task_standalone": self._val_generation(old.delta, old_task),
                    "old_task_fused": self._val_generation(fused, old_task),
                }
                self._val_loss_cache = {k: v for k, v in self._val_loss_cache.items() if k[0] != fused.adapter_id}
                del fused
            except Exception as exc:
                self.failures.append({"phase": "utility_supervision", "pair": f"{new_id}->{old_id}",
                                      "error": f"{type(exc).__name__}: {exc}"})
                log.exception("supervision pair %s->%s failed", new_id, old_id)
                cleanup(self.backbone.device)
                continue
            supervision.n_pairs_considered += 1
            supervision.n_pairs_evaluated += 1
            if gate_example is not None:
                gate_examples.append(gate_example)
            supervision.examples.append(SupervisionExample(
                candidate_id=old_id, task_id=new_task, features=vector, targets=targets,
                proxy=False, proxy_steps=None, beneficial=beneficial,
            ))
            measurements.append({
                "new_adapter": new_id, "new_task": new_task, "old_adapter": old_id, "old_task": old_task,
                "same_task": new_task == old_task,
                "validation_loss": {"new_standalone": pre_new, "new_fused": post_new,
                                    "old_standalone": pre_old, "old_fused": post_old},
                "forgetting_proxy": forgetting, "targets": targets, "beneficial": beneficial,
                "reuse_gate_example": ({"features": gate_example.features, "measured_gap": gate_example.gap}
                                       if gate_example is not None else None),
                "generation_check_validation": generation_check, "fusion": fusion,
                "features": vector.as_dict(), "missing_features": list(vector.missing),
                "wall_seconds": time.perf_counter() - start,
            })
            self.clock.record("utility_supervision", time.perf_counter() - start)

        report: dict[str, Any] = {"measurements": measurements, "supervision": supervision.as_dict()}
        n = len(supervision)
        limitations = []
        if n < self.study.min_supervision_examples:
            limitations.append(
                f"only {n} measured integrations (< decision_study.min_supervision_examples="
                f"{self.study.min_supervision_examples}); the utility network is fitted on too few "
                f"examples for its decisions to be claimed as generalising"
            )
        flags = [e.beneficial for e in supervision.examples]
        if flags and (all(flags) or not any(flags)):
            limitations.append(
                f"all {n} measured integrations were {'beneficial' if all(flags) else 'NOT beneficial'}: "
                f"there is no variation in the outcome label, so tau is set by the one-class policy "
                f"({self.cfg.get('manager.threshold.one_class_policy', 'median')}), not learned"
            )
        report["limitations"] = limitations
        if n == 0:
            report["fitted"] = False
            self._write("utility_supervision.json", report)
            return report

        network, normalizer, thresholds, record, utilities = fit_utility_model(self.cfg, supervision)
        network.save(self.tracker.paths.utility_network / "utility_network.pt",
                     run_mode=self.cfg.run_mode.value, feature_names=list(normalizer.feature_names))
        normalizer.save(self.tracker.paths.utility_network / "normalizer.json")
        features, targets_tensor = supervision.tensors(normalizer, self.cfg.run_mode.value, OUTCOME_HEADS)
        with torch.no_grad():
            predicted = network.predict_outcomes(features)
        for item, utility, pred in zip(measurements, utilities, predicted.tolist()):
            item["predicted_utility_in_sample"] = utility
            item["predicted_outcomes_in_sample"] = dict(zip(OUTCOME_HEADS, pred))

        fixed_rule_threshold = self._select_fixed_rule_threshold(supervision)
        self.network, self.normalizer, self.thresholds = network, normalizer, thresholds
        self.reuse_gate = None
        if gate_cfg is not None:
            self.reuse_gate = ReuseGate.fit(gate_examples, gate_cfg, run_mode=self.cfg.run_mode.value)
            self.reuse_gate.save(self.tracker.paths.utility_network / "reuse_gate.json")
            limitations.extend(f"reuse gate: {note}" for note in self.reuse_gate.notes)
            report["reuse_gate"] = {
                **self.reuse_gate.as_dict(),
                "in_sample": [{"task": e.task_id, "candidate": e.candidate_id, "measured_gap": e.gap,
                               "predicted_gap": self.reuse_gate.predict(e.features)} for e in gate_examples],
                "note": "in-sample fit only; phase 3 compares predictions with gaps measured on TEST",
            }
        self.fixed_rule_threshold = fixed_rule_threshold
        report.update({
            "fitted": True,
            "thresholds": thresholds.as_dict(),
            "fixed_rule_threshold": fixed_rule_threshold,
            "utility_training": record.as_dict(),
            "feature_dimension": len(normalizer.feature_names),
            "note": "in-sample predictions only: with this few examples there is no held-out "
                    "utility-network validation, so prediction error on unseen pairs is unmeasured "
                    "until phase 3",
        })
        self._write("utility_supervision.json", report)
        return report

    def _select_fixed_rule_threshold(self, supervision: SupervisionSet) -> dict[str, Any]:
        """Select the single-metric rule's threshold on the SAME measured evidence as tau_u."""
        manager = self.cfg.section("manager")
        sim = manager.get("similarity") or {}
        feature = str(sim.get("feature", "semantic.d_task_sq"))
        higher = bool(sim.get("higher_is_more_similar", False))
        rule = SimilaritySelectionStrategy(SimilarityConfig(feature, 0.0, higher))
        scores, flags = [], []
        for example in supervision.examples:
            if feature in example.features.missing:
                continue
            scores.append(rule.score({example.candidate_id: example.features}, run_mode=self.cfg.run_mode.value)[0].utility)
            flags.append(example.beneficial)
        if not scores:
            return {"feature": feature, "threshold": None,
                    "reason": f"'{feature}' was missing on every supervision example"}
        thresholds = select_thresholds(manager["threshold"], observed_utilities=scores, beneficial_flags=flags)
        return {"feature": feature, "higher_is_more_similar": higher, "threshold": thresholds.tau_u,
                "source": thresholds.source, "notes": thresholds.notes, "n_observations": len(scores)}

    # -- phase 3 -----------------------------------------------------------------
    def _policies(self) -> dict[str, Any]:
        built: dict[str, Any] = {}
        for name, order in (("amaun", str(self.cfg.get("manager.decision_order"))),
                            ("amaun_train_first", "train_then_decide")):
            if name in self.study.policies and getattr(self, "network", None) is not None:
                built[name] = AmaunStrategy(
                    network=self.network, normalizer=self.normalizer, thresholds=self.thresholds,
                    decision_order=order, allowed_outcomes=self.cfg.get("manager.allowed_outcomes"),
                    top_k=int(self.cfg.get("manager.retrieval.k")),
                    reuse_gate=getattr(self, "reuse_gate", None) if name == "amaun" else None,
                )
        if "always_merge" in self.study.policies:
            built["always_merge"] = AlwaysMergeStrategy(
                self.study.merge_feature, higher_is_more_similar=self.study.merge_feature_higher_is_more_similar)
        if "always_separate" in self.study.policies:
            built["always_separate"] = SeparateLoRAStrategy()
        threshold = getattr(self, "fixed_rule_threshold", {}).get("threshold")
        if "fixed_rule" in self.study.policies and threshold is not None:
            sim = self.cfg.section("manager").get("similarity") or {}
            built["fixed_rule"] = SimilaritySelectionStrategy(SimilarityConfig(
                str(sim.get("feature", "semantic.d_task_sq")), float(threshold),
                bool(sim.get("higher_is_more_similar", False))))
        return built

    @staticmethod
    def _option_key(decision: Decision) -> str:
        if decision.outcome is Outcome.CREATE:
            return "CREATE"
        return f"{decision.outcome.value}:{decision.selected_adapter_ids[0]}"

    def evaluate_task(self, spec: EvaluationTask, task_index: int) -> dict[str, Any]:
        task_id = spec.task_id
        payload: dict[str, Any] = {"task_id": task_id, "kind": spec.kind, "related_to": spec.related_to}
        if not self.clock.allows(f"evaluation:{task_id}"):
            payload["skipped"] = "time_budget_exhausted"
            return payload

        self.runner.ensure_representation(task_id)
        candidates = self.repository.entries()
        policies = self._policies()

        # Stage 1 -- the REUSE gate, BEFORE any training for this task (SUPPORT only).
        gate_decision = None
        gate_record: dict[str, Any] | None = None
        amaun = policies.get("amaun")
        if isinstance(amaun, AmaunStrategy) and amaun.gates_before_training:
            gate_start = time.perf_counter()
            gate_features = self.runner.reuse_gate_features(
                task_id, candidates, probe_examples=amaun.reuse_gate.config.probe_examples)
            gate_decision = amaun.reuse_gate.decide(gate_features, run_mode=self.cfg.run_mode.value)
            gate_record = {"decision": gate_decision.as_dict(), "features": gate_features,
                           "gate_seconds": time.perf_counter() - gate_start}
            if gate_decision.fired:
                decision = gate_reuse_decision(
                    gate_decision, task_id=task_id, task_index=task_index, thresholds=amaun.thresholds,
                    repository_size=len(self.repository), config_hash=self.cfg.config_hash)
                self.tracker.log_decision(**decision.as_dict(), strategy="amaun", study_task_kind=spec.kind,
                                          reuse_gate=gate_decision.as_dict())
                gate_record["decision_record"] = decision
        payload["reuse_gate"] = gate_record

        # Temporary adapter on the evaluation task's own TRAIN split (idea.md S3).  When the gate
        # fired, a deployment would stop here; the study still trains it so CREATE / FUSE can be
        # measured as counterfactuals, and reports the training time the gate would have saved.
        temp_id = f"temp-{task_id}"
        start = time.perf_counter()
        temp_delta, temp_record = train_adapter(
            backbone=self.backbone, task=self.tasks[task_id], guard=trainer_guard(self.splits[task_id]),
            lora_spec=self.runner.lora_spec, train_cfg=self.runner.train_cfg, resolved=self.cfg,
            adapter_id=temp_id, budget=self.runner.budget,
        )
        self.clock.record("train_temporary_adapters", time.perf_counter() - start)
        temp_entry = AdapterEntry(delta=temp_delta, metadata=self.runner._metadata_for(
            temp_id, self.tasks[task_id], temp_delta, temp_record.as_dict(), AdapterOrigin.TRAINED, [task_id]))
        self.repository.candidates.put(temp_entry)   # candidate store: never part of R_c
        payload["temporary_adapter"] = {
            "adapter_id": temp_id, "train_rows": len(self.splits[task_id][SplitName.TRAIN]),
            "steps": temp_record.steps, "final_train_loss": temp_record.final_train_loss,
            "wall_seconds": temp_record.wall_seconds, "peak_memory_bytes": temp_record.peak_memory_bytes,
        }

        if gate_record is not None:
            gate_record["training_seconds_saved_if_deployed"] = (
                temp_record.wall_seconds if gate_decision.fired else 0.0)
            gate_record["optimizer_steps_saved_if_deployed"] = temp_record.steps if gate_decision.fired else 0

        # Features from SUPPORT only; every policy decides BEFORE any TEST read for this task.
        contexts = self.runner._build_contexts(task_id, task_index, temp_delta)
        vectors = {cid: self.runner.extractor.extract(ctx) for cid, ctx in contexts.items()}
        del contexts
        prescreen = self.runner._reuse_prescreen(vectors)
        decisions: dict[str, Decision] = {}
        predictions: dict[str, Any] = {}
        for name, strategy in policies.items():
            if name == "amaun" and gate_record is not None and "decision_record" in gate_record:
                decisions[name] = gate_record.pop("decision_record")   # REUSE decided before training
                continue
            try:
                if isinstance(strategy, AmaunStrategy):
                    # After a gate that ran and did not fire, REUSE is already rejected.
                    strategy.set_prescreen(False if (name == "amaun" and gate_record is not None) else prescreen)
                scores = strategy.score(vectors, run_mode=self.cfg.run_mode.value)
                decision = strategy.decide(
                    task_id=task_id, task_index=task_index, scores=scores,
                    repository_size=len(self.repository), temp_adapter_trained=True,
                    config_hash=self.cfg.config_hash,
                )
            except Exception as exc:
                self.failures.append({"phase": "decision", "task": task_id, "policy": name,
                                      "error": f"{type(exc).__name__}: {exc}"})
                continue
            decisions[name] = decision
            if isinstance(strategy, AmaunStrategy) and not predictions:
                predictions = {s.adapter_id: {"utility": s.utility, "outcomes": s.outcomes} for s in scores}
            self.tracker.log_decision(**decision.as_dict(), strategy=name, study_task_kind=spec.kind)
        payload["decisions"] = {name: {"option": self._option_key(d), "reason": d.reason,
                                       "ranked": [c.as_dict() for c in d.ranked_candidates]}
                                for name, d in decisions.items()}
        payload["reuse_prescreen_passed"] = prescreen
        payload["features"] = {cid: {"values": v.as_dict(), "missing": list(v.missing)}
                               for cid, v in vectors.items()}

        # Measure options on TEST (after every decision above is frozen and logged).
        wanted = {"CREATE"} | {f"{o}:{e.adapter_id}" for e in candidates for o in ("REUSE", "FUSE")}
        if not self.study.measure_all_options:
            wanted = {self._option_key(d) for d in decisions.values()}
        options = self._measure_options(task_id, temp_delta, candidates, sorted(wanted))
        payload["options"] = {key: option.__dict__ for key, option in options.items()}
        payload["actual_vs_predicted"] = self._actual_vs_predicted(options, predictions)
        if gate_record is not None:
            gate_record["predicted_vs_actual"] = self._gate_vs_actual(gate_decision, options)
        payload["policy_outcomes"] = self._score_policies(task_id, decisions, options, candidates)

        self.repository.candidates.discard(temp_id)
        del temp_delta, temp_entry, vectors
        cleanup(self.backbone.device)
        self._write(f"evaluation_{task_id}.json", payload)
        return payload

    def _measure_options(self, task_id: str, temp_delta: AdapterDelta,
                         candidates: Sequence[AdapterEntry], keys: Sequence[str]) -> dict[str, _Option]:
        base_bytes = sum(e.delta.storage_bytes() for e in candidates)
        options: dict[str, _Option] = {}
        for key in keys:
            outcome, _, candidate_id = key.partition(":")
            option = _Option(key=key, outcome=outcome, candidate_id=candidate_id or None)
            options[key] = option
            if not self.clock.allows(f"option:{task_id}:{key}"):
                option.error = "skipped: time_budget_exhausted"
                continue
            try:
                eval_start = time.perf_counter()
                if outcome == "CREATE":
                    new = self._test_score(temp_delta, task_id, f"option:{task_id}:{key}")
                    option.adapters_retained = len(candidates) + 1
                    option.repository_bytes = base_bytes + temp_delta.storage_bytes()
                    serving = {}
                elif outcome == "REUSE":
                    entry = next(e for e in candidates if e.adapter_id == candidate_id)
                    new = self._test_score(entry.delta, task_id, f"option:{task_id}:{key}")
                    option.adapters_retained = len(candidates)
                    option.repository_bytes = base_bytes
                    serving = {}
                else:  # FUSE
                    entry = next(e for e in candidates if e.adapter_id == candidate_id)
                    fuse_start = time.perf_counter()
                    fused, option.fusion = self._fuse(entry, temp_delta, task_id, f"eval-{task_id}-fuse-{candidate_id}")
                    option.integration_seconds = time.perf_counter() - fuse_start
                    eval_start = time.perf_counter()
                    new = self._evaluate(fused, task_id, split=SplitName.TEST, evaluator=self.final_evaluator,
                                         label=f"option:{task_id}:{key}")
                    serving = {}
                    for t in entry.metadata.tasks_served:
                        old_eval = self._evaluate(fused, t, split=SplitName.TEST, evaluator=self.final_evaluator,
                                                  label=f"option:{task_id}:{key}:old:{t}")
                        serving[t] = old_eval["primary_score"]
                        option.old_task_losses[t] = old_eval["mean_token_loss"]
                    option.fused_bytes = fused.storage_bytes()
                    option.adapters_retained = len(candidates)
                    option.repository_bytes = base_bytes - entry.delta.storage_bytes() + fused.storage_bytes()
                    del fused
                    cleanup(self.backbone.device)
                option.new_task_score = new["primary_score"]
                option.new_task_loss = new["mean_token_loss"]
                for entry in candidates:
                    for t in entry.metadata.tasks_served:
                        option.old_task_scores[t] = serving.get(t, self._test_score(
                            entry.delta, t, f"standalone:{entry.adapter_id}")["primary_score"])
                option.evaluation_seconds = time.perf_counter() - eval_start
            except Exception as exc:
                option.error = f"{type(exc).__name__}: {exc}"
                self.failures.append({"phase": "option", "task": task_id, "option": key, "error": option.error})
                log.exception("option %s for %s failed", key, task_id)
                cleanup(self.backbone.device)
        return options

    def _gate_vs_actual(self, gate: Any, options: Mapping[str, _Option]) -> dict[str, Any]:
        """Predicted reuse gap beside the gap measured on TEST (same proxy, reporting only)."""
        scale = float(self.cfg.get("utility.supervision.loss_to_score_scale"))
        create = options.get("CREATE")
        rows = {}
        for candidate_id, predicted in gate.predicted_gaps.items():
            reuse = options.get(f"REUSE:{candidate_id}")
            if (reuse is None or create is None or reuse.error or create.error
                    or reuse.new_task_loss is None or create.new_task_loss is None):
                rows[candidate_id] = {"predicted_gap": predicted, "actual_gap": None,
                                      "reason": "REUSE or CREATE option not measured"}
                continue
            actual = reuse_gap(reuse.new_task_loss, create.new_task_loss, scale=scale)
            tol = self.reuse_gate.config.tolerance
            rows[candidate_id] = {
                "predicted_gap": predicted, "actual_gap_on_test": actual,
                "predicted_ok": predicted >= -tol, "actually_ok": actual >= -tol,
                "new_task_score_reuse": reuse.new_task_score, "new_task_score_separate": create.new_task_score,
            }
        return rows

    def _actual_vs_predicted(self, options: Mapping[str, _Option], predictions: Mapping[str, Any]) -> dict[str, Any]:
        """Predicted outcome vector beside the one MEASURED for the same FUSE, with both utilities.

        The actual vector is built with the supervision definition (``measure_outcome_targets``,
        loss->score proxy) from TEST measurements, and scalarised with the network's own
        ``psi`` and weights, so predicted and actual utility are directly comparable.
        """
        scale = float(self.cfg.get("utility.supervision.loss_to_score_scale"))
        network = getattr(self, "network", None)
        rows = {}
        for candidate_id, predicted in predictions.items():
            option = options.get(f"FUSE:{candidate_id}")
            if option is None or option.error or option.new_task_loss is None:
                rows[candidate_id] = {"predicted": predicted, "actual": None,
                                      "reason": "FUSE option not measured" if option is None else option.error}
                continue
            drops = []
            for t, post_loss in option.old_task_losses.items():
                pre_loss = self._test_cache.get((candidate_id, t), {}).get("mean_token_loss")
                if pre_loss is not None and post_loss is not None:
                    drops.append(max(0.0, _loss_to_score(pre_loss, scale=scale) - _loss_to_score(post_loss, scale=scale)))
            score = _loss_to_score(option.new_task_loss, scale=scale)
            actual, _ = measure_outcome_targets(
                primary_score_post=score,
                forgetting_post=sum(drops) / len(drops) if drops else 0.0,
                confidence_post=score,
                memory_bytes_post=int(option.fused_bytes or 0),
                communication_bytes_post=int(option.fused_bytes or 0),
                long_term_loss_post=option.new_task_loss,
            )
            actual_utility = None
            if network is not None:
                with torch.no_grad():
                    vector = torch.tensor([[actual[h] for h in network.heads]], dtype=torch.float32)
                    actual_utility = float((network.psi(vector) @ network.scalar_weights())[0])
            rows[candidate_id] = {
                "predicted": predicted,
                "actual_measured_on_test": {"outcomes": actual, "utility": actual_utility,
                                            "forgetting_measured": bool(drops)},
                "note": "actual outcomes use the supervision target definition (loss->score proxy) on TEST",
            }
        return rows

    def _score_policies(self, task_id: str, decisions: Mapping[str, Decision],
                        options: Mapping[str, _Option], candidates: Sequence[AdapterEntry]) -> dict[str, Any]:
        tol = self.study.outcome_tolerance
        standalone = {t: self._test_cache.get((e.adapter_id, t), {}).get("primary_score")
                      for e in candidates for t in e.metadata.tasks_served}
        create = options.get("CREATE")
        create_new = create.new_task_score if create and not create.error else None

        def summarise(option: _Option) -> dict[str, Any]:
            if option.error or option.new_task_score is None:
                return {"option": option.key, "measured": False, "reason": option.error}
            drops = [max(0.0, standalone[t] - s) for t, s in option.old_task_scores.items()
                     if standalone.get(t) is not None]
            scores = [option.new_task_score] + list(option.old_task_scores.values())
            return {
                "option": option.key, "measured": True,
                "new_task_score": option.new_task_score,
                "new_task_score_separate_adapter": create_new,
                "negative_transfer_vs_separate": (max(0.0, create_new - option.new_task_score)
                                                  if create_new is not None else None),
                "old_task_scores": option.old_task_scores,
                "old_task_retention": {t: (s / standalone[t] if standalone.get(t) else None)
                                       for t, s in option.old_task_scores.items()},
                "forgetting": sum(drops) / len(drops) if drops else 0.0,
                "average_performance": sum(scores) / len(scores),
                "adapters_retained": option.adapters_retained,
                "repository_bytes": option.repository_bytes,
                "integration_seconds": option.integration_seconds,
                "evaluation_seconds": option.evaluation_seconds,
            }

        summaries = {key: summarise(option) for key, option in options.items()}
        measured = {k: v for k, v in summaries.items() if v["measured"]}
        oracle = max(measured.values(), key=lambda s: (s["average_performance"], -s["adapters_retained"]),
                     default=None)
        result: dict[str, Any] = {"options": summaries,
                                  "oracle_option": oracle["option"] if oracle else None,
                                  "oracle_note": "best measured average performance on TEST; for "
                                                 "reporting regret only, never used by any policy"}
        create_summary = measured.get("CREATE")
        for name, decision in decisions.items():
            key = self._option_key(decision)
            chosen = summaries.get(key)
            if chosen is None or not chosen["measured"]:
                result[name] = {"option": key, "measured": False,
                                "reason": "option not measured" if chosen is None else chosen["reason"]}
                continue
            entry = {"option": key, **chosen,
                     "regret_vs_oracle": (oracle["average_performance"] - chosen["average_performance"]) if oracle else None}
            flags = []
            if key == "CREATE":
                better = [s["option"] for k, s in measured.items() if k != "CREATE"
                          and s["average_performance"] >= chosen["average_performance"] - tol
                          and s["forgetting"] <= tol]
                if better:
                    flags.append(f"unnecessary_create: {better} matched CREATE within tolerance with fewer adapters")
            elif create_summary is not None:
                if (chosen["average_performance"] < create_summary["average_performance"] - tol
                        or chosen["forgetting"] > tol):
                    flags.append("harmful_integration: worse than keeping a separate adapter")
            entry["flags"] = flags
            result[name] = entry
        return result

    # -- phase 4 -----------------------------------------------------------------
    def memory_audit(self) -> dict[str, Any]:
        entries = self.repository.entries()
        if self.study.memory_cycles <= 0 or len(entries) < 2:
            report = {"verdict": "not_run", "reason": "memory_audit.cycles <= 0 or < 2 adapters"}
            self._write("memory_audit.json", report)
            return report
        task_id = entries[0].metadata.tasks_served[0]
        evaluator = Evaluator(self.backbone, self.final_evaluator.generation,
                              max_examples=self.study.memory_cycle_eval_examples,
                              batch_size=self.final_evaluator.batch_size)
        post_cleanup = []
        for cycle in range(self.study.memory_cycles):
            if not self.clock.allows(f"memory_cycle:{cycle}"):
                break
            self.monitor.record("cycle:start", cycle=cycle)
            load_delta_into(self.peft_model, entries[0].delta)
            self.monitor.record("cycle:adapter_loaded", cycle=cycle)
            self._evaluate(entries[0].delta, task_id, split=SplitName.VALIDATION, evaluator=evaluator,
                           label=f"cycle{cycle}:standalone")
            fused, _ = self._fuse(entries[1], entries[0].delta, task_id, f"cycle{cycle}-fused",
                                  steps=self.study.memory_cycle_fusion_steps)
            self.monitor.record("cycle:fused", cycle=cycle)
            self._evaluate(fused, task_id, split=SplitName.VALIDATION, evaluator=evaluator,
                           label=f"cycle{cycle}:fused")
            del fused
            cleanup(self.backbone.device)
            post_cleanup.append(self.monitor.record("cycle:after_cleanup", cycle=cycle, count_tensors=True))
        report = stability_report(post_cleanup, allocated_tolerance_bytes=self.study.allocated_tolerance_bytes,
                                  tensor_tolerance=self.study.tensor_tolerance)
        report["cycle_definition"] = "load adapter -> evaluate -> fuse+KD -> evaluate fused -> release -> gc"
        self._write("memory_audit.json", report)
        return report

    # -- orchestration -----------------------------------------------------------
    def run(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "run_mode": self.cfg.run_mode.value, "is_reportable": self.cfg.run_mode.is_reportable,
            "experiment_id": self.cfg.experiment_id, "seed": self.cfg.seed,
            "backbone": self.cfg.backbone.model_id,
            "backbone_bytes": sum(p.numel() * p.element_size()
                                  for name, p in self.backbone.model.named_parameters()
                                  if not _is_adapter_parameter(name)),
            "assignments": {
                "initial_adapters": [a.__dict__ for a in self.study.initial_adapters],
                "evaluation_tasks": [t.__dict__ for t in self.study.evaluation_tasks],
            },
            "time_budget_hours": self.study.time_budget_hours,
        }
        self._write("assignments.json", summary["assignments"])
        if self.study.preflight:
            start = time.perf_counter()
            preflight = self.preflight()
            self.clock.record("preflight", time.perf_counter() - start)
            summary["preflight"] = {"ok": preflight["ok"]}
            if not preflight["ok"]:
                summary["aborted"] = "preflight OOM: see decision_study/preflight.json; no setting was changed"
                self._finish(summary)
                return summary

        self.monitor.record("phase1:start", count_tensors=True)
        summary["initial_adapters"] = [
            {k: r[k] for k in ("adapter_id", "task_id", "role", "train_rows", "steps",
                               "final_train_loss", "wall_seconds", "shares_training_rows_with")}
            for r in self.train_initial_adapters()
        ]
        if self.study.evaluate_zero_shot:
            summary["zero_shot_test"] = {}
            for task_id in [a.task_id for a in self.study.initial_adapters] + [t.task_id for t in self.study.evaluation_tasks]:
                if task_id in summary["zero_shot_test"] or not self.clock.allows(f"zero_shot:{task_id}"):
                    continue
                start = time.perf_counter()
                summary["zero_shot_test"][task_id] = self._test_score(None, task_id, f"zero_shot:{task_id}")["primary_score"]
                self.clock.record("zero_shot_evaluation", time.perf_counter() - start)

        self.monitor.record("phase2:start", count_tensors=True)
        supervision = self.utility_supervision()
        summary["utility_supervision"] = {k: supervision.get(k) for k in
                                          ("supervision", "thresholds", "fixed_rule_threshold", "limitations", "fitted")}

        self.monitor.record("phase3:start", count_tensors=True)
        evaluations = {}
        for index, spec in enumerate(self.study.evaluation_tasks, start=1):
            start = time.perf_counter()
            try:
                result = self.evaluate_task(spec, index)
            except Exception as exc:
                self.failures.append({"phase": "evaluation", "task": spec.task_id,
                                      "error": f"{type(exc).__name__}: {exc}"})
                log.exception("evaluation task %s failed", spec.task_id)
                cleanup(self.backbone.device)
                continue
            self.clock.record("evaluation_tasks", time.perf_counter() - start)
            evaluations[spec.task_id] = {"kind": spec.kind, "decisions": result.get("decisions"),
                                         "policy_outcomes": result.get("policy_outcomes"),
                                         "skipped": result.get("skipped")}
        summary["evaluation"] = evaluations
        summary["policy_comparison"] = self._compare_policies(evaluations)

        start = time.perf_counter()
        summary["memory_audit"] = self.memory_audit()
        self.clock.record("memory_audit", time.perf_counter() - start)

        assert_fingerprint_unchanged(self.backbone.model, self.backbone.fingerprint, phase="decision_study")
        summary["backbone_fingerprint_unchanged"] = True
        self._finish(summary)
        return summary

    def _compare_policies(self, evaluations: Mapping[str, Any]) -> dict[str, Any]:
        table: dict[str, Any] = {}
        for name in self.study.policies:
            rows = [ev["policy_outcomes"].get(name) for ev in evaluations.values()
                    if ev.get("policy_outcomes") and ev["policy_outcomes"].get(name)]
            measured = [r for r in rows if r.get("measured")]
            table[name] = {
                "decisions": {task: (ev["policy_outcomes"].get(name) or {}).get("option")
                              for task, ev in evaluations.items() if ev.get("policy_outcomes")},
                "n_measured": len(measured),
                "mean_average_performance": (sum(r["average_performance"] for r in measured) / len(measured)
                                             if measured else None),
                "mean_forgetting": sum(r["forgetting"] for r in measured) / len(measured) if measured else None,
                "mean_regret_vs_oracle": (sum(r["regret_vs_oracle"] for r in measured) / len(measured)
                                          if measured else None),
                "flags": [f"{task}: {flag}" for task, ev in evaluations.items()
                          for flag in ((ev.get("policy_outcomes") or {}).get(name) or {}).get("flags", [])],
            }
        table["caveat"] = (
            f"{len(evaluations)} evaluation task(s) x 1 seed: differences between policies are "
            f"descriptive, not statistically tested"
        )
        return table

    def _finish(self, summary: dict[str, Any]) -> None:
        summary["durations_seconds"] = dict(self.clock.phases)
        summary["total_wall_seconds"] = self.clock.elapsed
        summary["fits_time_budget"] = self.clock.elapsed <= self.clock.budget and not self.clock.skipped
        summary["skipped"] = list(self.clock.skipped)
        summary["failures"] = list(self.failures)
        summary["budget"] = self.runner.budget.as_dict()
        self._write("summary.json", summary)

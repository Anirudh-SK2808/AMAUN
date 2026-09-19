"""M10 -- the continual-learning runner: one client's task stream.

Executes the control flow of ``docs/master_prompt.md`` S5 exactly:

    task arrival -> support set -> task representation -> candidate enumeration
    -> 7-family features -> normalisation -> utility -> REUSE | FUSE | CREATE
    -> repository update -> compression -> federated round -> evaluation

The runner owns the **split discipline**: it builds a manager guard (SUPPORT), a trainer guard
(TRAIN/VALIDATION), and an evaluator guard (TEST) per task, and hands each component only its
own guard.  Nothing else in the pipeline constructs a guard, so the isolation is centralised
and reviewable in one place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.lora import LoRASpec, ensure_lora, extract_delta, load_delta_into, zero_delta_
from amaun.adapters.metadata import AdapterMetadata, AdapterOrigin
from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.baselines.strategies import AmaunStrategy, ManagementStrategy
from amaun.config import ResolvedConfig
from amaun.data.loaders import BatchPlan, GuardedLoader, fixed_reference_batch
from amaun.data.preprocess import collate
from amaun.data.registry import TaskSpec
from amaun.data.splits import (
    SplitName,
    TaskSplits,
    evaluator_guard,
    manager_guard,
    trainer_guard,
)
from amaun.errors import PreconditionError
from amaun.evaluation.cl_metrics import PerformanceMatrix
from amaun.evaluation.efficiency import snapshot as efficiency_snapshot
from amaun.evaluation.evaluator import Evaluator
from amaun.features.extractor import (
    CandidateContext,
    FeatureExtractor,
    probe_gradients,
    probe_latency,
    probe_quality,
    probe_semantic_divergence,
)
from amaun.fusion.layerwise import FusionConfig
from amaun.fusion.optimize import run_fusion_kd
from amaun.manager.compression import (
    CompressionMechanism,
    CompressionReport,
    plan_compression,
    should_compress,
)
from amaun.manager.decision import Decision, Outcome
from amaun.manager.task_representation import TaskRepresentation, build_task_representation
from amaun.models.backbone import LoadedBackbone, assert_fingerprint_unchanged
from amaun.training.budget import ArmBudget
from amaun.training.kd import KDConfig
from amaun.training.lora_trainer import TrainingConfig, train_adapter
from amaun.utils.logging import get_logger
from amaun.utils.tracking import ExperimentTracker

log = get_logger("continual.runner")


@dataclass
class TaskEvent:
    """One task arrival, fully recorded."""

    task_index: int
    task_id: str
    decision: dict[str, Any]
    outcome: str
    active_adapter_id: str | None
    n_candidates: int
    feature_dimension: int | None
    training: dict[str, Any] | None = None
    fusion: dict[str, Any] | None = None
    compression: dict[str, Any] | None = None
    efficiency: dict[str, Any] = field(default_factory=dict)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    #: VALIDATION scores used as continual-feature inputs (never TEST) -- see _prior_scores.
    validation_scores: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class RunnerResult:
    events: list[TaskEvent]
    matrix: PerformanceMatrix
    repository: AdapterRepository
    budget: ArmBudget
    task_representations: dict[str, dict[str, Any]]

    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.outcome] = counts.get(event.outcome, 0) + 1
        return counts


class ContinualRunner:
    """Drives the sequential task stream for one client."""

    def __init__(
        self,
        *,
        cfg: ResolvedConfig,
        backbone: LoadedBackbone,
        tasks: Sequence[TaskSpec],
        splits: Mapping[str, TaskSplits],
        repository: AdapterRepository,
        strategy: ManagementStrategy,
        extractor: FeatureExtractor,
        evaluator: Evaluator,
        tracker: ExperimentTracker,
        arm_id: str,
    ) -> None:
        self.cfg = cfg
        self.backbone = backbone
        self.tasks = {task.task_id: task for task in tasks}
        self.task_order = [task.task_id for task in tasks]
        self.splits = dict(splits)
        self.repository = repository
        self.strategy = strategy
        self.extractor = extractor
        self.evaluator = evaluator
        self.tracker = tracker
        self.arm_id = arm_id
        self.budget = ArmBudget(arm_id=arm_id)

        self.lora_spec = LoRASpec.from_config(cfg.section("lora"))
        self.train_cfg = TrainingConfig.from_config(cfg.section("training"))
        self.fusion_cfg = FusionConfig.from_config(
            cfg.section("fusion"), lora_learning_rate=self.train_cfg.learning_rate
        )
        self.kd_cfg = KDConfig.from_config(cfg.section("kd"))

        self.peft_model = ensure_lora(backbone.model, self.lora_spec)
        self.baseline_fingerprint = backbone.fingerprint
        self.representations: dict[str, TaskRepresentation] = {}
        self.matrix = PerformanceMatrix(task_order=[], score_name="primary_score")
        # Continual-learning FEATURES must not be derived from the TEST-filled matrix above:
        # that would let evaluation data influence a decision (master-prompt constraint 13).
        # This parallel history is filled from VALIDATION only and is what the manager reads.
        self.validation_history = PerformanceMatrix(task_order=[], score_name="primary_score")
        self.events: list[TaskEvent] = []
        self._stream: list[str] = []
        self._cumulative_train_seconds = 0.0

    # -- guards ------------------------------------------------------------------
    def _manager_guard(self, task_id: str):
        return manager_guard(self.splits[task_id])

    def _trainer_guard(self, task_id: str):
        return trainer_guard(self.splits[task_id])

    def _evaluator_guard(self, task_id: str):
        return evaluator_guard(self.splits[task_id])

    # -- bootstrap ---------------------------------------------------------------
    def bootstrap_repository(self, seed_task_ids: Sequence[str]) -> list[AdapterEntry]:
        """Train seed adapters **independently** from theta_0 and register all of them.

        D-002 option O1.  Each adapter gets a fresh injection, a fresh optimiser, and a seed
        derived from its own id, so training order cannot affect the result.  Registration
        uses ``bootstrap=True``, which is journaled -- a reviewer can see exactly which
        adapters bypassed the decision module and that it was by design.
        """
        entries: list[AdapterEntry] = []
        for task_id in seed_task_ids:
            task = self.tasks[task_id]
            adapter_id = f"seed-{task_id}"
            delta, record = train_adapter(
                backbone=self.backbone, task=task, guard=self._trainer_guard(task_id),
                lora_spec=self.lora_spec, train_cfg=self.train_cfg, resolved=self.cfg,
                adapter_id=adapter_id, budget=self.budget,
            )
            self._cumulative_train_seconds += record.wall_seconds
            entry = AdapterEntry(
                delta=delta,
                metadata=self._metadata_for(
                    adapter_id, task, delta, record.as_dict(), AdapterOrigin.TRAINED, [task_id]
                ),
            )
            self.repository.register(entry, bootstrap=True, task_index=0)
            entries.append(entry)
            self.tracker.log_metric(
                phase="bootstrap", task_id=task_id, adapter_id=adapter_id,
                steps=record.steps, final_train_loss=record.final_train_loss,
                wall_seconds=record.wall_seconds,
            )
        log.info("bootstrapped repository with %d independently trained adapters", len(entries))
        return entries

    def _metadata_for(
        self,
        adapter_id: str,
        task: TaskSpec,
        delta: AdapterDelta,
        training_config: dict[str, Any],
        origin: AdapterOrigin,
        tasks_served: Sequence[str],
        parents: Sequence[str] = (),
    ) -> AdapterMetadata:
        representation = self.representations.get(task.task_id)
        return AdapterMetadata(
            adapter_id=adapter_id,
            task_name=task.task_name,
            task_type=task.task_type.value,
            dataset=task.hf_id or f"synthetic:{task.task_id}",
            dataset_revision=task.hf_revision,
            model_version=str(self.cfg.backbone.model_id),
            backbone_fingerprint=self.backbone.fingerprint,
            run_mode=self.cfg.run_mode.value,
            lora_rank=delta.rank,
            target_modules=list(self.lora_spec.target_modules),
            training_configuration=training_config,
            task_embedding=list(representation.embedding) if representation else None,
            parameter_count=delta.parameter_count,
            storage_bytes=delta.storage_bytes(),
            origin=origin.value,
            parent_adapter_ids=list(parents),
            tasks_served=list(tasks_served),
        )

    # -- probes ------------------------------------------------------------------
    def _probe_batches(self, task_id: str, *, count: int) -> list:
        """Support-set batches used for every candidate's probes (manager guard only)."""
        task = self.tasks[task_id]
        guard = self._manager_guard(task_id)
        examples = list(guard.get(SplitName.SUPPORT))[:count]
        if not examples:
            return []
        return [
            collate(
                self.backbone.tokenizer, examples[i : i + 2],
                max_input_tokens=task.max_input_tokens,
                max_target_tokens=task.max_target_tokens,
            )
            for i in range(0, len(examples), 2)
        ]

    def _build_contexts(
        self, task_id: str, task_index: int, temp_delta: AdapterDelta
    ) -> dict[str, CandidateContext]:
        """Extract features for every candidate pair ``(A_temp, A_i)``."""
        features_cfg = self.cfg.section("features")
        probe_count = int(features_cfg.get("semantic", {}).get("hidden_probe_examples", 8))
        eval_count = int(features_cfg.get("performance", {}).get("max_eval_examples", 8))
        ece_bins = int(features_cfg.get("performance", {}).get("ece_bins", 10))
        latency_repeats = int(features_cfg.get("resource", {}).get("latency_probe_batches", 2))
        gradient_cfg = features_cfg.get("gradient", {})

        probe = self._probe_batches(task_id, count=max(probe_count, eval_count))
        reference = None
        if probe:
            reference = fixed_reference_batch(
                self._manager_guard(task_id), SplitName.SUPPORT, self.tasks[task_id],
                self.backbone.tokenizer,
                size=int(gradient_cfg.get("reference_batch_size", 4)),
                seed=int(gradient_cfg.get("reference_seed", 99)),
            )

        temp_quality = probe_quality(
            self.peft_model, temp_delta, probe, self.backbone.device, ece_bins=ece_bins
        ) if probe else {}
        temp_gradients = (
            probe_gradients(self.peft_model, temp_delta, reference, self.backbone.device)
            if reference is not None else {}
        )

        representation = self.representations.get(task_id)
        contexts: dict[str, CandidateContext] = {}
        for entry in self.repository.entries():
            divergence = (
                probe_semantic_divergence(
                    self.peft_model, temp_delta, entry.delta, probe[:1], self.backbone.device
                ) if probe else {}
            )
            candidate_quality = probe_quality(
                self.peft_model, entry.delta, probe, self.backbone.device, ece_bins=ece_bins
            ) if probe else {}
            candidate_gradients = (
                probe_gradients(self.peft_model, entry.delta, reference, self.backbone.device)
                if reference is not None else {}
            )
            latency = (
                probe_latency(
                    self.peft_model, entry.delta, probe[0], self.backbone.device, repeats=latency_repeats
                ) if probe else None
            )
            contexts[entry.adapter_id] = CandidateContext(
                temp_id=temp_delta.adapter_id,
                temp_delta=temp_delta,
                candidate_id=entry.adapter_id,
                candidate_delta=entry.delta,
                candidate_metadata=entry.metadata,
                task_index=task_index,
                total_events=max(1, len(self.events) + 1),
                task_embedding=representation.embedding if representation else None,
                candidate_task_embedding=entry.metadata.task_embedding,
                hidden_discrepancy=divergence.get("hidden_discrepancy"),
                output_kl=divergence.get("output_kl"),
                output_kl_reverse=divergence.get("output_kl_reverse"),
                temp_gradients=temp_gradients,
                candidate_gradients=candidate_gradients,
                temp_quality=temp_quality,
                candidate_quality=candidate_quality,
                prior_scores_pre=self._prior_scores(before=True),
                prior_scores_post=self._prior_scores(before=False),
                new_task_score_pre=None,
                new_task_score_post=None,
                repository_bytes=float(self.repository.memory_bytes()),
                repository_size=len(self.repository),
                candidate_latency_seconds=latency,
            )
        return contexts

    def _prior_scores(self, *, before: bool) -> dict[str, float]:
        """Prior-task scores for the continual feature family, from VALIDATION only.

        These values feed ``z_cl`` and therefore the utility network and the decision.  Reading
        them from ``self.matrix`` -- which the evaluator fills from TEST -- would let evaluation
        data influence adapter selection, violating master-prompt constraint 13 and idea.md
        S16.5.  ``self.validation_history`` is the leakage-safe source; it is filled by
        :meth:`_record_validation_scores` through a guard that permits VALIDATION and not TEST.
        """
        rows = self.validation_history.task_order
        if len(rows) < 2:
            return {}
        row = rows[-2] if before else rows[-1]
        return {
            task: value
            for task in rows[:-1]
            if (value := self.validation_history.get(row, task)) is not None
        }

    def _record_validation_scores(self, current_task_id: str) -> dict[str, float]:
        """Score every task seen so far on VALIDATION, for the continual features.

        Uses a utility guard (TRAIN / VALIDATION / SUPPORT) so a TEST read is impossible here by
        construction, and caps the example count because these scores are decision inputs, not
        reported metrics.
        """
        from amaun.data.splits import utility_guard

        scores: dict[str, float] = {}
        if current_task_id not in self.validation_history.task_order:
            self.validation_history.task_order.append(current_task_id)
        for task_id in self.validation_history.task_order:
            adapter_id = self._adapter_serving(task_id)
            if adapter_id and adapter_id in self.repository:
                load_delta_into(self.peft_model, self.repository.retrieve(adapter_id).delta)
            else:
                continue
            try:
                result = self.evaluator.evaluate(
                    self.tasks[task_id], utility_guard(self.splits[task_id]),
                    model=self.peft_model, adapter_id=adapter_id, split=SplitName.VALIDATION,
                )
            except Exception as exc:  # a missing split must not abort the run
                log.warning("validation scoring failed for '%s': %s", task_id, exc)
                continue
            self.validation_history.record(current_task_id, task_id, result.primary_score)
            scores[task_id] = result.primary_score
        return scores

    # -- the task event ----------------------------------------------------------
    def run_task(self, task_id: str, task_index: int) -> TaskEvent:
        """Process one arriving task end to end."""
        task = self.tasks[task_id]
        start = time.perf_counter()
        log.info("=== task %d: %s (%s) ===", task_index, task_id, task.task_name)

        # 1. Task representation -- SUPPORT only.
        representation = build_task_representation(
            task_id, self._manager_guard(task_id),
            config=self.cfg.section("task_repr"), backbone=self.backbone,
            descriptor=task.description,
        )
        self.representations[task_id] = representation

        # 2. Train the temporary adapter (idea.md S5.4 line 1).  It goes to the candidate
        #    store, never to R_c (master-prompt constraint 10).
        temp_id = f"temp-{task_id}-{task_index}"
        temp_delta, train_record = train_adapter(
            backbone=self.backbone, task=task, guard=self._trainer_guard(task_id),
            lora_spec=self.lora_spec, train_cfg=self.train_cfg, resolved=self.cfg,
            adapter_id=temp_id, budget=self.budget,
        )
        self._cumulative_train_seconds += train_record.wall_seconds
        temp_entry = AdapterEntry(
            delta=temp_delta,
            metadata=self._metadata_for(
                temp_id, task, temp_delta, train_record.as_dict(), AdapterOrigin.TRAINED, [task_id]
            ),
        )
        self.repository.candidates.put(temp_entry)

        # 3. Features and scoring.
        contexts = self._build_contexts(task_id, task_index, temp_delta)
        vectors = {cid: self.extractor.extract(ctx) for cid, ctx in contexts.items()}
        feature_dimension = len(next(iter(vectors.values()))) if vectors else None
        scores = self.strategy.score(vectors, run_mode=self.cfg.run_mode.value) if vectors else []

        # 4. Decide.
        if isinstance(self.strategy, AmaunStrategy):
            self.strategy.set_prescreen(self._reuse_prescreen(vectors))
        decision = self.strategy.decide(
            task_id=task_id, task_index=task_index, scores=scores,
            repository_size=len(self.repository), temp_adapter_trained=True,
            config_hash=self.cfg.config_hash,
        )
        self.tracker.log_decision(**decision.as_dict(), strategy=self.strategy.strategy_id,
                                  feature_dimension=feature_dimension)

        # 5. Execute the decision.
        fusion_payload: dict[str, Any] | None = None
        active_adapter_id: str | None
        if decision.outcome is Outcome.REUSE:
            active_adapter_id = decision.selected_adapter_ids[0]
            self.repository.mark_reuse(active_adapter_id, task_id, task_index)
            self.repository.candidates.discard(temp_id)   # A_temp is thrown away

        elif decision.outcome is Outcome.FUSE:
            components = [
                self.repository.retrieve(adapter_id).delta
                for adapter_id in decision.selected_adapter_ids
            ] + [temp_delta]
            fused_id = f"fused-{task_id}-{task_index}"
            task_batches = list(
                GuardedLoader(
                    self._trainer_guard(task_id), SplitName.TRAIN, task, self.backbone.tokenizer,
                    BatchPlan(batch_size=self.train_cfg.batch_size, shuffle=True, seed=self.cfg.seed),
                )
            )[: max(1, self.fusion_cfg.steps)]
            retention = self._retention_batches(exclude_task=task_id)
            fused_delta, fusion_record = run_fusion_kd(
                backbone=self.backbone, peft_model=self.peft_model, components=components,
                task_batches=task_batches, retention_batches=retention,
                fusion_cfg=self.fusion_cfg, kd_cfg=self.kd_cfg, fused_adapter_id=fused_id,
                default_rank=self.lora_spec.r, budget=self.budget,
            )
            fusion_payload = fusion_record.as_dict()
            self.tracker.write_json(
                f"fusion_logs/{fused_id}.json", {**fusion_payload, "alpha_summary": fusion_record.alpha_summary()}
            )
            fused_entry = AdapterEntry(
                delta=fused_delta,
                metadata=self._metadata_for(
                    fused_id, task, fused_delta, {"fusion": fusion_record.refactorization},
                    AdapterOrigin.FUSED, [task_id], parents=decision.selected_adapter_ids + [temp_id],
                ),
            )
            self.repository.replace_with_fused(
                replaced_id=decision.selected_adapter_ids[0], fused_entry=fused_entry,
                authorised_by=decision, task_index=task_index,
            )
            self.repository.candidates.discard(temp_id)
            active_adapter_id = fused_id

        else:  # CREATE
            self.repository.register(temp_entry, authorised_by=decision, task_index=task_index)
            active_adapter_id = temp_id

        # 6. Compression, if the memory budget is exceeded.
        compression_payload = self._maybe_compress(task_index, decision)

        # 7. Evaluate on TEST -- the first and only test access for this row.
        self.matrix.task_order.append(task_id)
        future: list[str] = []
        if bool(self.cfg.get("evaluation.evaluate_unseen_for_fwt", False)):
            seen = set(self.matrix.task_order)
            future = [t for t in self._stream if t not in seen]
        evaluations = self._evaluate_seen_tasks(task_id, active_adapter_id, future_task_ids=future)

        efficiency = efficiency_snapshot(
            self.repository, task_index=task_index, n_tasks_seen=len(self.matrix.task_order),
            trainable_parameters_this_event=train_record.trainable_parameters,
            training_seconds_cumulative=self._cumulative_train_seconds,
        )

        # Fill the leakage-safe validation history for the NEXT task's continual features.
        validation_scores = self._record_validation_scores(task_id)

        event = TaskEvent(
            task_index=task_index,
            task_id=task_id,
            decision=decision.as_dict(),
            outcome=decision.outcome.value,
            active_adapter_id=active_adapter_id,
            n_candidates=len(vectors),
            feature_dimension=feature_dimension,
            training=train_record.as_dict(),
            fusion=fusion_payload,
            compression=compression_payload,
            efficiency=efficiency.as_dict(),
            evaluations=evaluations,
            validation_scores=validation_scores,
            wall_seconds=time.perf_counter() - start,
        )
        self.events.append(event)
        self.tracker.log_metric(
            phase="task_event", task_index=task_index, task_id=task_id,
            outcome=decision.outcome.value, active_adapter_id=active_adapter_id,
            **{f"eff_{k}": v for k, v in efficiency.as_dict().items() if isinstance(v, (int, float))},
        )
        self.repository.snapshot(
            self.tracker.paths.repository_snapshots / f"task_{task_index:03d}.json",
            label=f"after {task_id}",
        )

        # The backbone must be untouched after every event.
        assert_fingerprint_unchanged(
            self.backbone.model, self.baseline_fingerprint, phase=f"task_event:{task_id}"
        )
        return event

    def _reuse_prescreen(self, vectors: Mapping[str, Any]) -> bool:
        """Support-set pre-screen for the REUSE outcome (D-003 O1).

        Passes when some existing adapter's support-set loss is already no worse than the
        freshly trained temporary adapter's by more than a configured margin.  Uses SUPPORT
        only, and returns False when the evidence is unavailable rather than guessing.
        """
        margin = float(self.cfg.get("manager.reuse_prescreen_margin", 0.0))
        for vector in vectors.values():
            values = vector.as_dict()
            gap = values.get("performance.L_val_gap")
            if gap is None:
                continue
            if gap <= margin:      # candidate loss <= temp loss (+ margin): reuse is defensible
                return True
        return False

    def _retention_batches(self, *, exclude_task: str) -> list:
        """``D_rep`` for KD: prior tasks' TRAIN/VALIDATION data, never TEST."""
        limit = self.kd_cfg.retention_examples
        batches = []
        for task_id in self.matrix.task_order:
            if task_id == exclude_task or len(batches) * 2 >= limit:
                continue
            task = self.tasks[task_id]
            guard = self._trainer_guard(task_id)
            examples = list(guard.get(SplitName.VALIDATION))[:2] or list(guard.get(SplitName.TRAIN))[:2]
            if examples:
                batches.append(
                    collate(
                        self.backbone.tokenizer, examples,
                        max_input_tokens=task.max_input_tokens,
                        max_target_tokens=task.max_target_tokens,
                    )
                )
        return batches

    def _maybe_compress(self, task_index: int, decision: Decision) -> dict[str, Any] | None:
        """Utility-based repository compression when ``Mem(R_c) >= R_max`` (idea.md S7.2)."""
        max_bytes = self.cfg.get("repository.max_bytes", None)
        triggered, reason = should_compress(self.repository, int(max_bytes) if max_bytes else None)
        if not triggered:
            return {"triggered": False, "reason": reason}

        mechanism = CompressionMechanism(
            self.cfg.get("repository.compression.mechanism", "utility")
        )
        min_adapters = int(self.cfg.get("repository.min_adapters", 2))
        before = self.repository.memory_bytes()

        if mechanism is CompressionMechanism.UTILITY and isinstance(self.strategy, AmaunStrategy):
            def _directed_utility(source: AdapterEntry, target: AdapterEntry) -> float:
                """Utility of integrating ``source`` into ``target``, using the SAME mechanism
                as selection (idea.md S16.7)."""
                ctx = CandidateContext(
                    temp_id=source.adapter_id, temp_delta=source.delta,
                    candidate_id=target.adapter_id, candidate_delta=target.delta,
                    candidate_metadata=target.metadata, task_index=task_index,
                    total_events=max(1, len(self.events)),
                    task_embedding=source.metadata.task_embedding,
                    candidate_task_embedding=target.metadata.task_embedding,
                    repository_bytes=float(self.repository.memory_bytes()),
                    repository_size=len(self.repository),
                )
                vector = self.extractor.extract(ctx)
                scored = self.strategy.score(
                    {target.adapter_id: vector}, run_mode=self.cfg.run_mode.value
                )
                return scored[0].utility if scored else float("-inf")

            def pair_utility(left: AdapterEntry, right: AdapterEntry) -> float:
                """Symmetric pair utility.

                idea.md S7.2 writes ``argmax_{i != j} U_merge(A_i, A_j)`` over UNORDERED pairs,
                but the feature vector is directional: historical, performance and resource
                features describe the *candidate* only, so U(i,j) != U(j,i). Taking the mean of
                both directions makes the score symmetric, so the chosen pair does not depend on
                the repository's iteration order.
                """
                forward = _directed_utility(left, right)
                backward = _directed_utility(right, left)
                finite = [v for v in (forward, backward) if v > float("-inf")]
                return sum(finite) / len(finite) if finite else float("-inf")
        else:
            from amaun.manager.compression import heuristic_pair_utility

            def pair_utility(left: AdapterEntry, right: AdapterEntry) -> float:
                return heuristic_pair_utility(left, right, task_index=task_index)

        plan, evaluated = plan_compression(
            self.repository, pair_utility=pair_utility, mechanism=mechanism, min_adapters=min_adapters
        )
        report = CompressionReport(
            triggered=True, reason=reason,
            repository_bytes_before=before, repository_bytes_after=before,
            pairs_evaluated=evaluated,
        )
        if plan is None:
            report.reason = f"{reason}; but no mergeable pair (|R_c|<={min_adapters})"
            return report.as_dict()

        left = self.repository.retrieve(plan.first_id)
        right = self.repository.retrieve(plan.second_id)
        merged_id = f"compressed-{plan.first_id}+{plan.second_id}-t{task_index}"
        from amaun.fusion.refactorize import FusionStorage, fuse_to_delta

        merged_delta, refactor = fuse_to_delta(
            [left.delta, right.delta], [0.5, 0.5],
            storage=FusionStorage(self.fusion_cfg.storage),
            target_rank=self.fusion_cfg.refactorize_rank or self.lora_spec.r,
            adapter_id=merged_id,
        )
        merged_entry = AdapterEntry(
            delta=merged_delta,
            metadata=self._metadata_for(
                merged_id, self.tasks[self.task_order[0]], merged_delta,
                {"compression": refactor.as_dict()}, AdapterOrigin.COMPRESSED, [],
                parents=[plan.first_id, plan.second_id],
            ),
        )
        self.repository.replace_pair_with_merged(
            first_id=plan.first_id, second_id=plan.second_id,
            merged_entry=merged_entry, authorised_by=decision,
        )
        report.steps.append(plan)
        report.repository_bytes_after = self.repository.memory_bytes()
        return report.as_dict()

    def _evaluate_seen_tasks(
        self,
        current_task_id: str,
        active_adapter_id: str | None,
        *,
        future_task_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Fill row ``S[current][*]`` using TEST.

        This is the only TEST access in the whole pipeline, and it happens *after* the decision
        for this row is frozen.

        ``future_task_ids`` additionally evaluates not-yet-arrived tasks, which is what forward
        transfer requires (``FWT = mean_j S[j-1][j] - b_j``).  It is opt-in via
        ``evaluation.evaluate_unseen_for_fwt`` because it means the evaluator reads a future
        task's TEST split: legitimate, since these scores are recorded for reporting only and are
        never read back by the manager (``_prior_scores`` looks at past tasks only), but it is an
        exposure worth making explicit rather than implicit.
        """
        results = []
        for task_id in list(self.matrix.task_order) + list(future_task_ids):
            adapter_id = self._adapter_serving(task_id) or active_adapter_id
            if adapter_id and adapter_id in self.repository:
                load_delta_into(self.peft_model, self.repository.retrieve(adapter_id).delta)
            elif adapter_id == active_adapter_id and active_adapter_id in self.repository.candidates:
                load_delta_into(self.peft_model, self.repository.candidates.get(active_adapter_id).delta)
            else:
                zero_delta_(self.peft_model)

            result = self.evaluator.evaluate(
                self.tasks[task_id], self._evaluator_guard(task_id),
                model=self.peft_model, adapter_id=adapter_id,
            )
            # Recorded in the matrix but NOT added to task_order: a future task has not been
            # learned, and the CL metrics derive their task set from task_order.
            self.matrix.record(current_task_id, task_id, result.primary_score)
            if adapter_id and adapter_id in self.repository:
                self.repository.record_evaluation(
                    adapter_id, task_id, result.metrics, len(self.matrix.task_order) - 1
                )
            results.append(result.as_dict())
            self.tracker.log_metric(
                phase="evaluation", after_task=current_task_id, evaluated_task=task_id,
                adapter_id=adapter_id, primary_score=result.primary_score,
                primary_score_name=result.primary_score_name, **result.metrics,
            )
        return results

    def _adapter_serving(self, task_id: str) -> str | None:
        """Which repository adapter currently serves ``task_id``.

        Several adapters can legitimately claim a task after a fusion or a reuse, so the tie
        is broken explicitly and deterministically: the most recently created adapter wins
        (highest ``created_at_task_index``, then adapter id). Relying on repository iteration
        order would make evaluation depend on dict ordering.
        """
        candidates = [
            entry for entry in self.repository.entries()
            if task_id in entry.metadata.tasks_served
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda e: (e.metadata.created_at_task_index or -1, e.adapter_id)
        )
        return candidates[-1].adapter_id

    # -- the stream --------------------------------------------------------------
    def run(self, stream: Sequence[str]) -> RunnerResult:
        """Run the full task stream."""
        if not stream:
            raise PreconditionError("the evaluated task stream is empty")
        self._stream = list(stream)
        for index, task_id in enumerate(stream):
            if task_id not in self.tasks:
                raise PreconditionError(f"stream references unknown task '{task_id}'")
            self.run_task(task_id, index)

        self.matrix.save(self.tracker.paths.performance_matrix)
        return RunnerResult(
            events=self.events,
            matrix=self.matrix,
            repository=self.repository,
            budget=self.budget,
            task_representations={k: v.as_dict() for k, v in self.representations.items()},
        )

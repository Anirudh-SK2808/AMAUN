"""M7 -- bootstrapping the utility network's own supervision (idea.md S9).

The targets do not exist a priori.  This module generates them by **actually performing**
candidate integrations on a *decision-development* task stream and measuring the outcome:

    for each dev task:
        train A_temp
        for each sampled candidate A_i:
            extract z_i
            perform a (possibly truncated) fusion of (A_i, A_temp)
            MEASURE the post-integration outcome on VALIDATION  -> y_i
    fit the normaliser on these z          (never on TEST -- idea.md S16.5)
    train g_omega on (z, y)                (never on TEST -- idea.md S16.6)
    select tau from the observed (U, beneficial) pairs   (D-005 dev_stream)

Everything here reads TRAIN / VALIDATION / SUPPORT through a utility guard.  No TEST handle
exists in this module, so the leakage constraint holds structurally rather than by care.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.lora import load_delta_into
from amaun.adapters.repository import AdapterRepository
from amaun.config import ResolvedConfig
from amaun.data.preprocess import collate, masked_cross_entropy
from amaun.data.registry import TaskSpec
from amaun.data.splits import SplitName, TaskSplits, manager_guard, trainer_guard, utility_guard
from amaun.errors import ManagerError
from amaun.features.extractor import CandidateContext, FeatureExtractor
from amaun.features.normalizer import FeatureNormalizer
from amaun.fusion.layerwise import FusionConfig
from amaun.fusion.optimize import run_fusion_kd
from amaun.manager.decision import Thresholds
from amaun.manager.targets import (
    PairSampling,
    SupervisionExample,
    SupervisionSet,
    estimate_proxy_bias,
    measure_outcome_targets,
    sample_pairs,
)
from amaun.manager.threshold import select_thresholds
from amaun.manager.utility_network import (
    OUTCOME_HEADS,
    UtilityNetwork,
    UtilityNetworkConfig,
    train_utility_network,
)
from amaun.training.kd import KDConfig
from amaun.training.lora_trainer import TrainingConfig, train_adapter
from amaun.utils.logging import get_logger

log = get_logger("manager.bootstrap")


@dataclass
class BootstrapResult:
    network: UtilityNetwork
    normalizer: FeatureNormalizer
    thresholds: Thresholds
    supervision: SupervisionSet
    training_record: dict[str, Any]
    feature_names: list[str]
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.as_dict(),
            "supervision": self.supervision.as_dict(),
            "utility_training": self.training_record,
            "feature_dimension": len(self.feature_names),
            "feature_names": self.feature_names,
            "wall_seconds": self.wall_seconds,
            "notes": list(self.notes),
        }


@torch.no_grad()
def _validation_loss(model: Any, delta: AdapterDelta, task: TaskSpec, splits: TaskSplits,
                     tokenizer: Any, device: str, *, max_examples: int) -> float:
    """Mean masked CE on VALIDATION with ``delta`` installed. Never touches TEST."""
    guard = utility_guard(splits)
    examples = list(guard.get(SplitName.VALIDATION))[:max_examples]
    if not examples:
        return float("nan")
    load_delta_into(model, delta)
    model.eval()
    batch = collate(
        tokenizer, examples,
        max_input_tokens=task.max_input_tokens, max_target_tokens=task.max_target_tokens,
    ).to(device)
    logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
    return float(masked_cross_entropy(logits, batch.labels))


def _loss_to_score(loss: float) -> float:
    """Map a validation loss to a bounded score in (0,1] for the ``primary_score`` head.

    A proxy for task score during bootstrapping: generating and scoring text for every
    candidate pair would multiply the already-dominant bootstrap cost (D-010).  This is a
    monotone transform of loss, so it preserves the *ordering* the utility head needs, and the
    fact that it is a proxy is recorded in the bootstrap notes so it cannot be mistaken for a
    measured task metric.
    """
    import math

    if not math.isfinite(loss):
        return 0.0
    return float(math.exp(-max(0.0, loss) / 10.0))


def bootstrap_utility_network(
    *,
    cfg: ResolvedConfig,
    backbone: Any,
    peft_model: Any,
    repository: AdapterRepository,
    dev_tasks: Sequence[TaskSpec],
    splits: Mapping[str, TaskSplits],
    extractor: FeatureExtractor,
) -> BootstrapResult:
    """Generate supervision, fit the normaliser, train ``g_omega``, and select thresholds."""
    if repository.is_empty:
        raise ManagerError(
            "utility bootstrapping needs a non-empty repository: there are no candidate pairs to "
            "observe. Run the bootstrap_six protocol first, or supply seed adapters."
        )
    if not dev_tasks:
        raise ManagerError(
            "no decision-development tasks supplied. Selecting thresholds on the evaluated "
            "stream would be selection on test data (docs/decision_register.md D-005)."
        )

    start = time.perf_counter()
    supervision_cfg = cfg.section("utility")["supervision"]
    lora_spec_cfg = cfg.section("lora")
    from amaun.adapters.lora import LoRASpec

    lora_spec = LoRASpec.from_config(lora_spec_cfg)
    train_cfg = TrainingConfig.from_config(cfg.section("training"))
    fusion_cfg = FusionConfig.from_config(cfg.section("fusion"), lora_learning_rate=train_cfg.learning_rate)
    kd_cfg = KDConfig.from_config(cfg.section("kd"))
    max_eval = int(cfg.get("features.performance.max_eval_examples", 8))

    proxy_steps = int(supervision_cfg["proxy_steps"])
    proxy_validation_frac = float(supervision_cfg.get("proxy_validation_frac", 0.0))
    max_pairs = int(supervision_cfg["max_pairs_per_event"])
    sampling = PairSampling(supervision_cfg["pair_sampling"])

    supervision = SupervisionSet(fit_split="utility_train")
    full_examples: list[SupervisionExample] = []
    notes = [
        "primary_score targets during bootstrap are a monotone transform of VALIDATION loss, "
        "not a generated-text metric: scoring text for every candidate pair would multiply the "
        "dominant bootstrap cost (D-010). Ordering is preserved; absolute values are proxies.",
    ]

    for task in dev_tasks:
        task_splits = splits[task.task_id]
        temp_id = f"devtemp-{task.task_id}"
        temp_delta, _ = train_adapter(
            backbone=backbone, task=task, guard=trainer_guard(task_splits),
            lora_spec=lora_spec, train_cfg=train_cfg, resolved=cfg, adapter_id=temp_id,
        )
        temp_loss = _validation_loss(
            peft_model, temp_delta, task, task_splits, backbone.tokenizer, backbone.device,
            max_examples=max_eval,
        )
        baseline_score = _loss_to_score(temp_loss)

        # Features for every candidate (cheap relative to fusion).
        vectors = {}
        for entry in repository.entries():
            ctx = CandidateContext(
                temp_id=temp_id, temp_delta=temp_delta,
                candidate_id=entry.adapter_id, candidate_delta=entry.delta,
                candidate_metadata=entry.metadata, task_index=0,
                total_events=max(1, len(dev_tasks)),
                candidate_task_embedding=entry.metadata.task_embedding,
                temp_quality={"loss": temp_loss},
                candidate_quality={
                    "loss": _validation_loss(
                        peft_model, entry.delta, task, task_splits, backbone.tokenizer,
                        backbone.device, max_examples=max_eval,
                    )
                },
                repository_bytes=float(repository.memory_bytes()),
                repository_size=len(repository),
            )
            vectors[entry.adapter_id] = extractor.extract(ctx)

        chosen = sample_pairs(
            vectors, policy=sampling, max_pairs=max_pairs, seed=cfg.seed
        )
        supervision.n_pairs_considered += len(vectors)

        guard = utility_guard(task_splits)
        task_examples = list(guard.get(SplitName.TRAIN))[:4]
        task_batches = [
            collate(
                backbone.tokenizer, task_examples[i : i + 2],
                max_input_tokens=task.max_input_tokens, max_target_tokens=task.max_target_tokens,
            )
            for i in range(0, len(task_examples), 2)
        ] or None
        if task_batches is None:
            raise ManagerError(f"dev task '{task.task_id}' has no TRAIN examples for fusion")
        retention = [task_batches[0]]

        for position, candidate_id in enumerate(chosen):
            entry = repository.retrieve(candidate_id)
            run_full = (
                proxy_validation_frac > 0.0
                and position < max(1, int(round(proxy_validation_frac * len(chosen))))
            )
            steps = fusion_cfg.steps if run_full else proxy_steps
            trial_cfg = FusionConfig(**{**fusion_cfg.__dict__, "steps": steps})

            fused_delta, record = run_fusion_kd(
                backbone=backbone, peft_model=peft_model,
                components=[entry.delta, temp_delta],
                task_batches=task_batches, retention_batches=retention,
                fusion_cfg=trial_cfg, kd_cfg=kd_cfg,
                fused_adapter_id=f"bootstrap-{candidate_id}-{task.task_id}",
                default_rank=lora_spec.r,
            )
            supervision.n_pairs_evaluated += 1

            # --- MEASURE the post-integration outcome (this is the supervision signal) -----
            post_loss = _validation_loss(
                peft_model, fused_delta, task, task_splits, backbone.tokenizer, backbone.device,
                max_examples=max_eval,
            )
            candidate_prior_loss = _validation_loss(
                peft_model, entry.delta, task, task_splits, backbone.tokenizer, backbone.device,
                max_examples=max_eval,
            )
            # Forgetting: how much worse the fused adapter is on the candidate's own prior
            # task(s) than the candidate was. Positive means capability was lost.
            forgetting = 0.0
            prior_task_ids = [t for t in entry.metadata.tasks_served if t in splits]
            if prior_task_ids:
                drops = []
                for prior_id in prior_task_ids:
                    prior_task = next((t for t in dev_tasks if t.task_id == prior_id), None)
                    if prior_task is None:
                        continue
                    before = _loss_to_score(
                        _validation_loss(
                            peft_model, entry.delta, prior_task, splits[prior_id],
                            backbone.tokenizer, backbone.device, max_examples=max_eval,
                        )
                    )
                    after = _loss_to_score(
                        _validation_loss(
                            peft_model, fused_delta, prior_task, splits[prior_id],
                            backbone.tokenizer, backbone.device, max_examples=max_eval,
                        )
                    )
                    drops.append(max(0.0, before - after))
                forgetting = sum(drops) / len(drops) if drops else 0.0

            targets, beneficial = measure_outcome_targets(
                primary_score_post=_loss_to_score(post_loss),
                forgetting_post=forgetting,
                confidence_post=_loss_to_score(post_loss),
                memory_bytes_post=fused_delta.storage_bytes(),
                communication_bytes_post=fused_delta.storage_bytes(),
                long_term_loss_post=post_loss if post_loss == post_loss else 0.0,
                primary_score_baseline=baseline_score,
            )
            example = SupervisionExample(
                candidate_id=candidate_id, task_id=task.task_id,
                features=vectors[candidate_id], targets=targets,
                proxy=not run_full, proxy_steps=None if run_full else proxy_steps,
                beneficial=beneficial,
            )
            supervision.examples.append(example)
            if run_full:
                full_examples.append(example)

    if not supervision.examples:
        raise ManagerError("bootstrapping produced zero supervision examples")

    supervision.proxy_bias = estimate_proxy_bias(
        [e for e in supervision.examples if e.proxy], full_examples
    )
    if not full_examples and proxy_validation_frac > 0:
        notes.append(
            "no full-convergence pairs were run, so proxy bias could not be measured; "
            "utility targets are proxy-only (assumption A-12 unverified for this run)"
        )

    # --- fit the normaliser on THIS split only (idea.md S4.7 / S16.5) --------------------
    normalizer = FeatureNormalizer(epsilon=float(cfg.get("utility.normalisation.epsilon", 1e-8)))
    normalizer.fit(
        [example.features for example in supervision.examples],
        split=str(cfg.get("utility.normalisation.fit_split", "utility_train")),
        run_mode=cfg.run_mode.value,
    )

    features, targets_tensor = supervision.tensors(normalizer, cfg.run_mode.value, OUTCOME_HEADS)
    network = UtilityNetwork(
        features.shape[1], UtilityNetworkConfig.from_config(cfg.section("utility")["network"])
    )
    record = train_utility_network(
        network, features, targets_tensor,
        config=cfg.section("utility")["training"],
        fit_split=supervision.fit_split, run_mode=cfg.run_mode.value, seed=cfg.seed,
    )

    # --- select thresholds from the observed (U, beneficial) pairs (D-005) ---------------
    network.eval()
    with torch.no_grad():
        utilities = [float(u) for u in network.utility(features)]
    thresholds = select_thresholds(
        cfg.section("manager")["threshold"],
        observed_utilities=utilities,
        beneficial_flags=[example.beneficial for example in supervision.examples],
    )

    log.info(
        "utility bootstrap: %d examples (%d proxy), final L_U=%.5f, tau_u=%.4f tau_reuse=%.4f",
        len(supervision), sum(1 for e in supervision.examples if e.proxy),
        record.final_loss or float("nan"), thresholds.tau_u, thresholds.tau_reuse,
    )
    return BootstrapResult(
        network=network,
        normalizer=normalizer,
        thresholds=thresholds,
        supervision=supervision,
        training_record=record.as_dict(),
        feature_names=list(normalizer.feature_names),
        wall_seconds=time.perf_counter() - start,
        notes=notes,
    )

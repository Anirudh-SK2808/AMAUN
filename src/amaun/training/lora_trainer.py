"""M4 -- the single LoRA/QLoRA training loop.

There is exactly one training loop in AMAUN.  Every arm -- the six seed adapters, CREATE,
the post-fusion KD optimisation, and every baseline -- goes through it, so no arm can
accidentally get a different optimiser, schedule, or masking rule.

**The independence guarantee** (brief S1, reviewer falsification question 2).  Training
adapter *k* must not depend on whether adapters ``1..k-1`` were trained first in the same
process.  Three things make that true:

1. A *fresh* PEFT injection per adapter (no reused adapter state).
2. A *fresh* optimiser per adapter (no carried-over moments).
3. A seed derived from ``(base_seed, adapter_id)``, applied immediately before training, so
   the RNG stream does not depend on call order.

``tests/test_adapter_independence.py`` verifies this by training the same adapter in two
different orders and comparing deltas bit-for-bit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.lora import (
    LoRASpec,
    ensure_lora,
    extract_delta,
    reset_lora_parameters,
    trainable_parameter_count,
    trainable_parameters,
)
from amaun.config import ResolvedConfig
from amaun.data.loaders import BatchPlan, GuardedLoader
from amaun.data.preprocess import masked_cross_entropy
from amaun.data.registry import TaskSpec
from amaun.data.splits import SplitAccessGuard, SplitName
from amaun.errors import ConfigError
from amaun.models.backbone import (
    LoadedBackbone,
    assert_backbone_frozen,
    assert_fingerprint_unchanged,
)
from amaun.training.budget import ArmBudget
from amaun.utils.logging import get_logger
from amaun.utils.seed import derive_seed, set_all_seeds

log = get_logger("training.lora")


@dataclass
class TrainingConfig:
    """All training hyperparameters.  Constructed from ``configs/training/*.yaml``."""

    learning_rate: float
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int | None
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    scheduler: str                # "cosine" | "linear" | "constant"
    gradient_checkpointing: bool
    early_stopping_patience: int | None
    eval_every_steps: int | None
    log_every_steps: int

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "TrainingConfig":
        required = (
            "learning_rate", "epochs", "batch_size", "gradient_accumulation_steps",
            "warmup_ratio", "weight_decay", "max_grad_norm", "scheduler",
            "gradient_checkpointing", "log_every_steps",
        )
        missing = [key for key in required if key not in section]
        if missing:
            raise ConfigError(f"training config missing required keys: {missing}")
        return cls(
            learning_rate=float(section["learning_rate"]),
            epochs=float(section["epochs"]),
            batch_size=int(section["batch_size"]),
            gradient_accumulation_steps=int(section["gradient_accumulation_steps"]),
            max_steps=int(section["max_steps"]) if section.get("max_steps") else None,
            warmup_ratio=float(section["warmup_ratio"]),
            weight_decay=float(section["weight_decay"]),
            max_grad_norm=float(section["max_grad_norm"]),
            scheduler=str(section["scheduler"]),
            gradient_checkpointing=bool(section["gradient_checkpointing"]),
            early_stopping_patience=(
                int(section["early_stopping_patience"]) if section.get("early_stopping_patience") else None
            ),
            eval_every_steps=int(section["eval_every_steps"]) if section.get("eval_every_steps") else None,
            log_every_steps=int(section["log_every_steps"]),
        )


@dataclass
class TrainingRecord:
    """Everything observed during one adapter's training.

    ``loss_trajectory`` is required by idea.md S3 step 5 as an input to Module 2's feature
    extraction, so it is a first-class output rather than a log line.
    """

    adapter_id: str
    task_id: str
    seed: int
    steps: int = 0
    epochs_completed: float = 0.0
    loss_trajectory: list[dict[str, float]] = field(default_factory=list)
    validation_trajectory: list[dict[str, float]] = field(default_factory=list)
    final_train_loss: float | None = None
    best_validation_loss: float | None = None
    wall_seconds: float = 0.0
    tokens_seen: int = 0
    examples_seen: int = 0
    trainable_parameters: int = 0
    peak_memory_bytes: int | None = None
    early_stopped: bool = False
    backbone_fingerprint_before: str = ""
    backbone_fingerprint_after: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig, total_steps: int):
    warmup = max(1, int(cfg.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        if cfg.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        if cfg.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        if cfg.scheduler == "constant":
            return 1.0
        raise ConfigError(f"unknown scheduler '{cfg.scheduler}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_loss(model: Any, loader: GuardedLoader, device: str, *, max_batches: int | None = None) -> float:
    """Mean masked CE over a loader.  Used for early stopping and for ``L_val`` features."""
    model.eval()
    total, count = 0.0, 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = batch.to(device)
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        total += float(masked_cross_entropy(logits, batch.labels))
        count += 1
    model.train()
    return total / max(1, count)


def train_adapter(
    *,
    backbone: LoadedBackbone,
    task: TaskSpec,
    guard: SplitAccessGuard,
    lora_spec: LoRASpec,
    train_cfg: TrainingConfig,
    resolved: ResolvedConfig,
    adapter_id: str,
    budget: ArmBudget | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> tuple[AdapterDelta, TrainingRecord]:
    """Train exactly one adapter on exactly one task's TRAIN split, from ``theta_0``."""
    if bool(resolved.get("model.backbone_trainable", False)):
        raise ConfigError(
            "model.backbone_trainable=true is declared, but full-backbone fine-tuning (baseline "
            "B1) is NOT implemented in this trainer. Running it through the LoRA path would "
            "silently produce a LoRA arm mislabelled as full fine-tuning, which would corrupt "
            "the baseline comparison. See docs/acceptance_criteria.md H1 and "
            "docs/decision_register.md D-007."
        )

    seed = derive_seed(resolved.seed, adapter_id)
    set_all_seeds(seed)

    fingerprint_before = backbone.fingerprint
    # One injection per backbone (see adapters.lora.ensure_lora); independence comes from
    # re-initialising the factors under a seed derived from this adapter's id, plus a fresh
    # optimiser below.  Training order therefore cannot affect the result (invariant I4.3).
    peft_model = ensure_lora(backbone.model, lora_spec, adapter_name="default")
    reset_lora_parameters(peft_model, seed=seed, adapter_name="default")
    # Re-seed AFTER injection: the very first ensure_lora call performs PEFT's own
    # initialisation and consumes global RNG, so without this the dropout stream (and hence the
    # trained weights) would depend on whether this adapter happened to be trained first.
    # Order-independence is asserted by tests/test_adapter_independence.py.
    set_all_seeds(seed)

    if train_cfg.gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
        # Both calls above are known to re-enable grads on base modules; re-assert (I1.3).
        assert_backbone_frozen(peft_model)

    params = trainable_parameters(peft_model)
    if not params:
        raise ConfigError("no trainable parameters after LoRA injection")
    # Invariant I4.1: the optimiser may only ever see adapter parameters.
    optimizer = torch.optim.AdamW(params, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    plan = BatchPlan(batch_size=train_cfg.batch_size, shuffle=True, seed=seed)
    train_loader = GuardedLoader(guard, SplitName.TRAIN, task, backbone.tokenizer, plan)
    val_loader = None
    if SplitName.VALIDATION in guard.permitted:
        val_loader = GuardedLoader(
            guard, SplitName.VALIDATION, task, backbone.tokenizer,
            BatchPlan(batch_size=train_cfg.batch_size, shuffle=False, seed=seed),
        )

    batches_per_epoch = max(1, len(train_loader))
    steps_per_epoch = max(1, batches_per_epoch // train_cfg.gradient_accumulation_steps)
    total_steps = train_cfg.max_steps or max(1, int(steps_per_epoch * train_cfg.epochs))
    scheduler = _build_scheduler(optimizer, train_cfg, total_steps)

    record = TrainingRecord(
        adapter_id=adapter_id,
        task_id=task.task_id,
        seed=seed,
        trainable_parameters=trainable_parameter_count(peft_model),
        backbone_fingerprint_before=fingerprint_before,
        config={
            "lora": lora_spec.as_dict(),
            "training": {
                "learning_rate": train_cfg.learning_rate,
                "epochs": train_cfg.epochs,
                "batch_size": train_cfg.batch_size,
                "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
                "max_steps": train_cfg.max_steps,
                "total_steps_planned": total_steps,
                "scheduler": train_cfg.scheduler,
                "warmup_ratio": train_cfg.warmup_ratio,
                "weight_decay": train_cfg.weight_decay,
                "max_grad_norm": train_cfg.max_grad_norm,
                "gradient_checkpointing": train_cfg.gradient_checkpointing,
            },
            "run_mode": resolved.run_mode.value,
            "backbone": resolved.backbone.model_id,
            "max_input_tokens": task.max_input_tokens,
            "max_target_tokens": task.max_target_tokens,
        },
    )

    peft_model.train()
    start = time.perf_counter()
    step = 0
    accumulated = 0
    best_val = math.inf
    patience_left = train_cfg.early_stopping_patience
    device = backbone.device
    epoch = 0.0

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            batch = batch.to(device)
            logits = peft_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
            loss = masked_cross_entropy(logits, batch.labels)
            (loss / train_cfg.gradient_accumulation_steps).backward()

            record.tokens_seen += int(batch.attention_mask.sum())
            record.examples_seen += len(batch)
            if budget is not None:
                budget.add_batch(tokens=int(batch.attention_mask.sum()), examples=len(batch))
            accumulated += 1

            if accumulated >= train_cfg.gradient_accumulation_steps:
                # Invariant I4.2: verify the frozen backbone before *every* optimiser step.
                assert_backbone_frozen(peft_model)
                torch.nn.utils.clip_grad_norm_(params, train_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                step += 1
                if budget is not None:
                    budget.add_steps()
                loss_value = float(loss.detach())
                record.loss_trajectory.append({"step": step, "loss": loss_value,
                                               "lr": scheduler.get_last_lr()[0]})
                if on_step is not None:
                    on_step(step, loss_value)
                if step % train_cfg.log_every_steps == 0:
                    log.info("[%s] step %d/%d loss=%.4f", adapter_id, step, total_steps, loss_value)

                if (
                    val_loader is not None
                    and train_cfg.eval_every_steps
                    and step % train_cfg.eval_every_steps == 0
                ):
                    val_loss = evaluate_loss(peft_model, val_loader, device, max_batches=8)
                    record.validation_trajectory.append({"step": step, "val_loss": val_loss})
                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        patience_left = train_cfg.early_stopping_patience
                    elif patience_left is not None:
                        patience_left -= 1
                        if patience_left <= 0:
                            record.early_stopped = True
                            log.info("[%s] early stopping at step %d", adapter_id, step)
                            step = total_steps
                            break
        epoch += 1.0
        if epoch > max(1.0, train_cfg.epochs) * 100:  # guards against a zero-length loader
            break

    record.steps = step
    record.epochs_completed = step / max(1, steps_per_epoch)
    record.wall_seconds = time.perf_counter() - start
    record.final_train_loss = record.loss_trajectory[-1]["loss"] if record.loss_trajectory else None
    record.best_validation_loss = None if best_val is math.inf else best_val

    from amaun.utils.profiling import peak_memory_bytes

    record.peak_memory_bytes = peak_memory_bytes(device)

    delta = extract_delta(peft_model, adapter_id=adapter_id)

    # The injection is deliberately left in place: it is shared, and the next adapter resets
    # the factors rather than re-wrapping the model.  Calling PEFT's unload() here would strip
    # the LoRA layers out of the shared backbone and invalidate every other handle to it.
    if train_cfg.gradient_checkpointing and hasattr(backbone.model, "gradient_checkpointing_disable"):
        backbone.model.gradient_checkpointing_disable()

    # Invariant I1.2: the backbone must be bit-identical after training.
    assert_fingerprint_unchanged(backbone.model, fingerprint_before, phase=f"train:{adapter_id}")
    record.backbone_fingerprint_after = fingerprint_before

    if budget is not None:
        budget.wall_seconds += record.wall_seconds
        budget.trainable_parameters = max(budget.trainable_parameters, record.trainable_parameters)

    log.info(
        "[%s] done: %d steps, final loss %s, %.1fs, backbone unchanged",
        adapter_id, record.steps, f"{record.final_train_loss:.4f}" if record.final_train_loss else "n/a",
        record.wall_seconds,
    )
    return delta, record

"""M12 -- the evaluator: the only component with TEST access (invariant I12.1).

Constructed from an evaluator guard permitting ``{TEST}`` and nothing else.  It is never
handed to the manager, and nothing in ``manager/`` imports this module (enforced by the
dependency rule in ``docs/architecture.md``), so there is no path from a decision to test data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from amaun.adapters.delta import AdapterDelta
from amaun.data.preprocess import collate, encode_prompt, masked_cross_entropy
from amaun.data.registry import TaskSpec
from amaun.data.splits import Example, SplitAccessGuard, SplitName
from amaun.errors import EvaluationError, SplitAccessViolation
from amaun.evaluation.task_metrics import compute_with_examples
from amaun.models.generation import GenerationConfig
from amaun.utils.logging import get_logger

log = get_logger("evaluation.evaluator")


@dataclass
class EvaluationResult:
    task_id: str
    adapter_id: str | None
    metrics: dict[str, float]
    primary_score: float
    primary_score_name: str
    n_examples: int
    mean_token_loss: float | None
    wall_seconds: float
    generation_config: dict[str, Any] = field(default_factory=dict)
    sample_predictions: list[dict[str, str]] = field(default_factory=list)
    #: Every example's prediction, reference and per-metric scores (l4_full_study item 10), in
    #: split order, so paired comparisons and bootstrap CIs can be computed later.
    per_example: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class Evaluator:
    """Generates on TEST and computes the task's declared metrics."""

    def __init__(
        self,
        backbone: Any,
        generation: GenerationConfig,
        *,
        max_examples: int | None = None,
        batch_size: int = 4,
        keep_samples: int = 3,
        scorers: dict[str, Any] | None = None,
    ) -> None:
        self.backbone = backbone
        self.generation = generation
        self.max_examples = max_examples
        self.batch_size = batch_size
        self.keep_samples = keep_samples
        #: Model-based metrics (e.g. ``{"bertscore_f1": BERTScorer}``), shared by every arm.
        self.scorers = dict(scorers or {})

    @torch.no_grad()
    def evaluate(
        self,
        task: TaskSpec,
        guard: SplitAccessGuard,
        *,
        model: Any,
        adapter_id: str | None,
        split: SplitName | str = SplitName.TEST,
        max_examples: int | None = None,
    ) -> EvaluationResult:
        """Evaluate whatever adapter is currently installed on ``model``.

        ``max_examples`` overrides the evaluator's cap for this call (per-task TEST sizes).
        """
        if SplitName(split) not in guard.permitted:
            raise SplitAccessViolation(
                f"evaluator was given a guard permitting "
                f"{sorted(p.value for p in guard.permitted)}, not '{SplitName(split).value}'"
            )
        examples: list[Example] = list(guard.get(split))
        cap = max_examples if max_examples is not None else self.max_examples
        if cap is not None:
            examples = examples[:cap]
        if not examples:
            raise EvaluationError(f"task '{task.task_id}' has no examples in split '{split}'")

        tokenizer = self.backbone.tokenizer
        device = self.backbone.device
        model.eval()
        start = time.perf_counter()

        predictions: list[str] = []
        references: list[str] = [example.target for example in examples]
        losses: list[float] = []

        for begin in range(0, len(examples), self.batch_size):
            chunk = examples[begin : begin + self.batch_size]

            # Teacher-forced loss (for perplexity and as a decoding-free signal).
            batch = collate(
                tokenizer, chunk,
                max_input_tokens=task.max_input_tokens,
                max_target_tokens=task.max_target_tokens,
                prompt_truncation=task.prompt_truncation,
            ).to(device)
            logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
            losses.append(float(masked_cross_entropy(logits, batch.labels)))

            if self.generation.batch_size <= 1:
                for example in chunk:
                    predictions.append(self._generate_one(model, task, example))
        if self.generation.batch_size > 1:
            predictions = self._generate_batched(model, task, examples)

        mean_loss = sum(losses) / len(losses) if losses else None
        metrics, per_metric = compute_with_examples(task.metrics, predictions, references,
                                                    mean_token_loss=mean_loss, scorers=self.scorers)
        if task.primary_score not in metrics:
            raise EvaluationError(
                f"task '{task.task_id}' declares primary_score='{task.primary_score}' but the "
                f"computed metrics are {sorted(metrics)}; the primary score must be one of the "
                f"task's own metrics (docs/decision_register.md D-009)"
            )
        primary = float(metrics[task.primary_score])

        return EvaluationResult(
            task_id=task.task_id,
            adapter_id=adapter_id,
            metrics=metrics,
            primary_score=primary,
            primary_score_name=task.primary_score,
            n_examples=len(examples),
            mean_token_loss=mean_loss,
            wall_seconds=time.perf_counter() - start,
            generation_config=self.generation.as_dict(),
            sample_predictions=[
                {"prompt": e.prompt[:200], "reference": r[:200], "prediction": p[:200]}
                for e, r, p in list(zip(examples, references, predictions))[: self.keep_samples]
            ],
            per_example=[
                {"index": i, "example_id": e.example_id, "prediction": p, "reference": r,
                 "scores": {name: values[i] for name, values in per_metric.items()}}
                for i, (e, r, p) in enumerate(zip(examples, references, predictions))
            ],
        )

    def _generate_batched(self, model: Any, task: TaskSpec, examples: Sequence[Example]) -> list[str]:
        """Greedy generation ``generation.batch_size`` prompts at a time (D-026).

        Prompts are encoded exactly as in :meth:`_generate_one`, ordered by length so a batch pads
        little, left-padded (a decoder continues from the last position), and decoded per row
        after the shared prompt width.  Predictions are returned in the examples' order.
        """
        tokenizer = self.backbone.tokenizer
        device = self.backbone.device
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise EvaluationError("batched generation needs a pad or eos token id")
        encoded = [encode_prompt(tokenizer, e.prompt, max_input_tokens=task.max_input_tokens,
                                 truncation=task.prompt_truncation) for e in examples]
        kwargs = self.generation.to_kwargs(pad_token_id=pad_id,
                                           eos_token_id=getattr(tokenizer, "eos_token_id", None))
        kwargs["max_new_tokens"] = min(kwargs["max_new_tokens"], task.max_target_tokens)
        order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]), i))
        predictions: list[str | None] = [None] * len(encoded)
        size = self.generation.batch_size
        for begin in range(0, len(order), size):
            rows = order[begin : begin + size]
            width = max(len(encoded[i]) for i in rows)
            input_ids = torch.tensor([[pad_id] * (width - len(encoded[i])) + encoded[i] for i in rows],
                                     dtype=torch.long, device=device)
            attention = torch.tensor([[0] * (width - len(encoded[i])) + [1] * len(encoded[i]) for i in rows],
                                     dtype=torch.long, device=device)
            output = model.generate(input_ids=input_ids, attention_mask=attention, **kwargs)
            for row, index in enumerate(rows):
                predictions[index] = tokenizer.decode(output[row][width:], skip_special_tokens=True).strip()
        return [p if p is not None else "" for p in predictions]

    def _generate_one(self, model: Any, task: TaskSpec, example: Example) -> str:
        tokenizer = self.backbone.tokenizer
        device = self.backbone.device
        # The same encoder as training (collate -> tokenize_example), so the prompt the model is
        # evaluated on is the prompt it was trained on, including the truncation policy.
        ids = encode_prompt(
            tokenizer, example.prompt, max_input_tokens=task.max_input_tokens,
            truncation=task.prompt_truncation,
        )
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        encoded = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        kwargs = self.generation.to_kwargs(
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        kwargs["max_new_tokens"] = min(kwargs["max_new_tokens"], task.max_target_tokens)
        output = model.generate(**encoded, **kwargs)
        generated = output[0][encoded["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def zero_shot_baselines(
    evaluator: Evaluator,
    tasks: Sequence[TaskSpec],
    guards: dict[str, SplitAccessGuard],
    *,
    model: Any,
) -> dict[str, float]:
    """``b_j`` -- the frozen backbone with no adapter, for forward transfer (assumption A-06).

    Measured once and cached.  The caller must ensure no adapter delta is installed; use
    :func:`amaun.adapters.lora.zero_delta_` first.
    """
    baselines: dict[str, float] = {}
    for task in tasks:
        guard = guards.get(task.task_id)
        if guard is None:
            continue
        result = evaluator.evaluate(task, guard, model=model, adapter_id=None)
        baselines[task.task_id] = result.primary_score
        log.info("zero-shot baseline b_%s = %.4f", task.task_id, result.primary_score)
    return baselines

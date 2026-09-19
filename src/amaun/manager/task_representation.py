"""M5 -- task representation from the SUPPORT set only.

Brief S7 requires a *configurable* representation mechanism, and brief S5 restricts the
manager to the support/calibration set.  This module takes a
:class:`~amaun.data.splits.SplitAccessGuard` permitting ``{SUPPORT}``, so no encoder here can
reach TRAIN, VALIDATION, or TEST even by mistake.

The default encoder (``backbone_mean_hidden``) reuses the single resident frozen backbone,
which matters for two reasons: it adds no second model to a 24 GB box, and it keeps the
encoder "fixed and identical across methods" as idea.md S4.1 requires -- a sentence encoder
would have to be identical across arms too, and would be one more thing to hold constant.
"""

from __future__ import annotations

import enum
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from amaun.data.splits import Example, SplitAccessGuard, SplitName
from amaun.errors import ManagerError


class TaskEncoder(str, enum.Enum):
    BACKBONE_MEAN_HIDDEN = "backbone_mean_hidden"
    SENTENCE_ENCODER = "sentence_encoder"
    TOKEN_STATS = "token_stats"
    HYBRID = "hybrid"


@dataclass
class TaskRepresentation:
    """``(embedding, stats, descriptor)`` for one task."""

    task_id: str
    embedding: list[float]
    stats: dict[str, float]
    descriptor: str
    encoder: str
    n_support_examples: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "encoder": self.encoder,
            "embedding_dim": len(self.embedding),
            "stats": self.stats,
            "descriptor": self.descriptor,
            "n_support_examples": self.n_support_examples,
            "provenance": self.provenance,
        }


def support_set_statistics(examples: Sequence[Example]) -> dict[str, float]:
    """Distributional descriptors of the support set (brief S7 'support-set statistics')."""
    if not examples:
        return {}
    prompt_lengths = [len(example.prompt.split()) for example in examples]
    target_lengths = [len(example.target.split()) for example in examples]
    tokens = Counter(word for example in examples for word in example.prompt.split())
    total_tokens = sum(tokens.values())

    def summarise(values: list[int], prefix: str) -> dict[str, float]:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_min": float(min(values)),
            f"{prefix}_max": float(max(values)),
        }

    return {
        "n_examples": float(len(examples)),
        **summarise(prompt_lengths, "prompt_words"),
        **summarise(target_lengths, "target_words"),
        "type_token_ratio": len(tokens) / max(1, total_tokens),
        "vocabulary_size": float(len(tokens)),
        "target_to_prompt_ratio": (sum(target_lengths) / max(1, sum(prompt_lengths))),
    }


@torch.no_grad()
def _backbone_mean_hidden(
    backbone: Any, examples: Sequence[Example], *, pooling: str, max_tokens: int
) -> list[float]:
    """Mean (or last) last-layer hidden state over support prompts."""
    model, tokenizer, device = backbone.model, backbone.tokenizer, backbone.device
    model.eval()
    accumulated: torch.Tensor | None = None
    counted = 0
    for example in examples:
        encoded = tokenizer(
            example.prompt, return_tensors="pt", truncation=True, max_length=max_tokens
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = model(**encoded, output_hidden_states=True)
        hidden = output.hidden_states[-1][0].float()   # [seq, hidden]
        if pooling == "last":
            pooled = hidden[-1]
        elif pooling == "mean":
            pooled = hidden.mean(dim=0)
        else:
            raise ManagerError(f"unknown pooling '{pooling}'")
        accumulated = pooled if accumulated is None else accumulated + pooled
        counted += 1
    if accumulated is None or counted == 0:
        raise ManagerError("cannot encode an empty support set")
    return (accumulated / counted).cpu().tolist()


def _token_stats_embedding(stats: Mapping[str, float]) -> list[float]:
    """Encoder-free embedding: the support statistics themselves, in fixed key order.

    This is the ``A12`` ablation (no semantic task representation) and doubles as the
    fallback when no encoder is available.
    """
    return [float(stats[key]) for key in sorted(stats)]


def build_task_representation(
    task_id: str,
    guard: SplitAccessGuard,
    *,
    config: Mapping[str, Any],
    backbone: Any | None = None,
    descriptor: str = "",
) -> TaskRepresentation:
    """Build a task representation from SUPPORT only."""
    if SplitName.SUPPORT not in guard.permitted:
        raise ManagerError(
            f"task representation requires a guard permitting SUPPORT; got "
            f"{sorted(p.value for p in guard.permitted)}"
        )
    encoder = TaskEncoder(config.get("encoder", TaskEncoder.BACKBONE_MEAN_HIDDEN.value))
    max_examples = int(config.get("max_support_examples", 32))
    examples = list(guard.get(SplitName.SUPPORT))[:max_examples]
    if not examples:
        raise ManagerError(f"task '{task_id}' has an empty support set")

    stats = support_set_statistics(examples)
    pooling = str(config.get("pooling", "mean"))
    max_tokens = int(config.get("max_encoder_tokens", 256))

    if encoder is TaskEncoder.TOKEN_STATS:
        embedding = _token_stats_embedding(stats)
    elif encoder is TaskEncoder.BACKBONE_MEAN_HIDDEN:
        if backbone is None:
            raise ManagerError("backbone_mean_hidden requires the loaded backbone")
        embedding = _backbone_mean_hidden(backbone, examples, pooling=pooling, max_tokens=max_tokens)
    elif encoder is TaskEncoder.HYBRID:
        if backbone is None:
            raise ManagerError("hybrid encoder requires the loaded backbone")
        embedding = _backbone_mean_hidden(backbone, examples, pooling=pooling, max_tokens=max_tokens)
        embedding = embedding + _token_stats_embedding(stats)
    else:  # SENTENCE_ENCODER
        raise ManagerError(
            "task_repr.encoder='sentence_encoder' is not implemented: it would introduce a "
            "second resident model and an additional dependency that must then be held "
            "identical across every arm (idea.md S4.1). Use backbone_mean_hidden, or add the "
            "encoder as an explicit, logged experimental variable first."
        )

    if config.get("normalise", True):
        norm = math.sqrt(sum(value * value for value in embedding))
        if norm > 0:
            embedding = [value / norm for value in embedding]

    return TaskRepresentation(
        task_id=task_id,
        embedding=embedding,
        stats=stats,
        descriptor=descriptor or task_id,
        encoder=encoder.value,
        n_support_examples=len(examples),
        provenance={
            "split_read": "support",
            "pooling": pooling,
            "max_support_examples": max_examples,
            "normalised": bool(config.get("normalise", True)),
        },
    )

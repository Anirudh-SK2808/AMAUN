"""M6 -- feature extraction orchestration and the probe protocol.

Builds one :class:`CandidateContext` per pair ``(A_temp, A_i)`` (idea.md S4:
``P_t = {(A_temp, A_i) | A_i in R_c}``), runs the enabled families, and assembles a named
:class:`FeatureVector`.

Probe protocol -- the parts that make features *comparable* across candidates:

- **Semantic and performance probes** run on the same support batch for every candidate.
- **Gradient probes** use the single fixed reference minibatch of idea.md S4.3.
- **Latency probes** run a fixed number of forward passes, so ``Latency_i`` is measured
  under identical conditions per candidate.

The extractor holds only a manager guard (SUPPORT) plus precomputed history, so it has no
path to TEST even in principle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.lora import load_delta_into, zero_delta_
from amaun.adapters.metadata import AdapterMetadata
from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.data.preprocess import TokenizedBatch, masked_cross_entropy
from amaun.features.base import FeatureFamily, FeatureVector
from amaun.features.continual import ContinualFeatures
from amaun.features.gradient import GradientFeatures
from amaun.features.historical import HistoricalFeatures
from amaun.features.parameter import ParameterFeatures
from amaun.features.performance import PerformanceFeatures, confidence_entropy_ece
from amaun.features.resource import ResourceFeatures
from amaun.features.semantic import SemanticFeatures
from amaun.utils.logging import get_logger

log = get_logger("features.extractor")

FAMILY_REGISTRY: dict[str, type[FeatureFamily]] = {
    "semantic": SemanticFeatures,
    "parameter": ParameterFeatures,
    "gradient": GradientFeatures,
    "performance": PerformanceFeatures,
    "continual": ContinualFeatures,
    "historical": HistoricalFeatures,
    "resource": ResourceFeatures,
}

# Fixed family order, so the feature vector layout is stable regardless of config dict order.
FAMILY_ORDER = ("semantic", "parameter", "gradient", "performance", "continual", "historical", "resource")


@dataclass
class CandidateContext:
    """Everything the seven families need for one candidate pair."""

    temp_id: str
    temp_delta: AdapterDelta
    candidate_id: str
    candidate_delta: AdapterDelta
    candidate_metadata: AdapterMetadata
    task_index: int = 0
    total_events: int = 1

    task_embedding: Sequence[float] | None = None
    candidate_task_embedding: Sequence[float] | None = None
    hidden_discrepancy: float | None = None
    output_kl: float | None = None
    output_kl_reverse: float | None = None

    temp_gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    candidate_gradients: dict[str, torch.Tensor] = field(default_factory=dict)

    temp_quality: dict[str, float] = field(default_factory=dict)
    candidate_quality: dict[str, float] = field(default_factory=dict)

    prior_scores_pre: dict[str, float] = field(default_factory=dict)
    prior_scores_post: dict[str, float] = field(default_factory=dict)
    new_task_score_pre: float | None = None
    new_task_score_post: float | None = None

    repository_bytes: float = 0.0
    repository_size: int = 0
    candidate_latency_seconds: float | None = None


class FeatureExtractor:
    """Runs the enabled families and assembles named feature vectors."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        toggles = self.config.get("families") or {}
        self.enabled = [name for name in FAMILY_ORDER if toggles.get(name, False)]
        if not self.enabled:
            raise ValueError(
                "at least one feature family must be enabled; a decision on zero features is "
                "not a decision"
            )
        self.sentinel = float(self.config.get("missing_value_sentinel", 0.0))
        self.families: list[FeatureFamily] = [
            FAMILY_REGISTRY[name](self.config.get(name) or {}) for name in self.enabled
        ]
        log.info("feature families enabled: %s", self.enabled)

    def extract(self, ctx: CandidateContext) -> FeatureVector:
        names: list[str] = []
        values: list[float] = []
        slices: dict[str, tuple[int, int]] = {}
        missing: list[str] = []

        for family in self.families:
            start = len(names)
            produced = family.compute(ctx)
            for key in sorted(produced):
                value = produced[key]
                full_name = f"{family.name}.{key}"
                names.append(full_name)
                if value is None:
                    # Invariant I6.6: substitute a sentinel AND record the substitution, so
                    # a feature that is silently always-missing shows up in the logs.
                    missing.append(full_name)
                    values.append(self.sentinel)
                else:
                    values.append(float(value))
            slices[family.name] = (start, len(names))

        return FeatureVector(names=names, values=values, family_slices=slices, missing=missing)

    @property
    def dimension_signature(self) -> str:
        return "+".join(self.enabled)


# --------------------------------------------------------------------------------------
# Probes -- the measurements the families consume
# --------------------------------------------------------------------------------------


@torch.no_grad()
def probe_quality(
    model: Any,
    delta: AdapterDelta,
    batches: Sequence[TokenizedBatch],
    device: str,
    *,
    ece_bins: int,
) -> dict[str, float]:
    """Loss, confidence, entropy, and ECE for one adapter on a fixed batch set."""
    load_delta_into(model, delta)
    model.eval()
    losses, confidences, entropies, eces, accuracies = [], [], [], [], []
    for batch in batches:
        moved = batch.to(device)
        logits = model(input_ids=moved.input_ids, attention_mask=moved.attention_mask).logits
        losses.append(float(masked_cross_entropy(logits, moved.labels)))
        stats = confidence_entropy_ece(logits.cpu(), moved.labels.cpu(), bins=ece_bins)
        confidences.append(stats["confidence"])
        entropies.append(stats["entropy"])
        eces.append(stats["ece"])
        accuracies.append(stats["token_accuracy"])

    def mean(values: list[float]) -> float:
        finite = [v for v in values if v == v]  # drop NaN
        return sum(finite) / len(finite) if finite else float("nan")

    return {
        "loss": mean(losses),
        "confidence": mean(confidences),
        "entropy": mean(entropies),
        "ece": mean(eces),
        "token_accuracy": mean(accuracies),
    }


@torch.no_grad()
def probe_semantic_divergence(
    model: Any,
    temp_delta: AdapterDelta,
    candidate_delta: AdapterDelta,
    batches: Sequence[TokenizedBatch],
    device: str,
) -> dict[str, float]:
    """``d_hidden`` and ``D_out`` (idea.md S4.1) on a shared probe batch.

    Both adapters are evaluated on the *same* inputs through the *same* single resident
    backbone, switching only the installed delta -- so the comparison isolates the adapter.
    """
    model.eval()
    hidden_gaps, kl_forward, kl_reverse = [], [], []
    for batch in batches:
        moved = batch.to(device)
        load_delta_into(model, temp_delta)
        temp_out = model(
            input_ids=moved.input_ids, attention_mask=moved.attention_mask, output_hidden_states=True
        )
        temp_logits = temp_out.logits.detach().float()
        temp_hidden = temp_out.hidden_states[-1].detach().float()

        load_delta_into(model, candidate_delta)
        cand_out = model(
            input_ids=moved.input_ids, attention_mask=moved.attention_mask, output_hidden_states=True
        )
        cand_logits = cand_out.logits.detach().float()
        cand_hidden = cand_out.hidden_states[-1].detach().float()

        mask = moved.attention_mask.unsqueeze(-1).float()
        squared = ((temp_hidden - cand_hidden) ** 2 * mask).sum(dim=-1)
        token_count = moved.attention_mask.sum().clamp(min=1)
        hidden_gaps.append(float(squared.sum() / token_count))

        temp_log = torch.log_softmax(temp_logits, dim=-1)
        cand_log = torch.log_softmax(cand_logits, dim=-1)
        temp_prob = temp_log.exp()
        cand_prob = cand_log.exp()
        flat_mask = moved.attention_mask.bool()
        # idea.md S4.1: D_out = KL(p_i || p_temp), i.e. candidate as the reference.
        kl_forward.append(float(((cand_prob * (cand_log - temp_log)).sum(-1))[flat_mask].mean()))
        kl_reverse.append(float(((temp_prob * (temp_log - cand_log)).sum(-1))[flat_mask].mean()))

    return {
        "hidden_discrepancy": sum(hidden_gaps) / max(1, len(hidden_gaps)),
        "output_kl": sum(kl_forward) / max(1, len(kl_forward)),
        "output_kl_reverse": sum(kl_reverse) / max(1, len(kl_reverse)),
    }


def probe_gradients(
    model: Any,
    delta: AdapterDelta,
    batch: TokenizedBatch,
    device: str,
) -> dict[str, torch.Tensor]:
    """Gradient of the task loss w.r.t. the installed adapter's parameters.

    Uses the fixed reference minibatch (idea.md S4.3).  The *same* batch is passed for both
    adapters of a pair by the caller, which is what makes alignment comparable.
    """
    from amaun.adapters.lora import iter_lora_modules

    load_delta_into(model, delta)
    model.train()
    model.zero_grad(set_to_none=True)
    moved = batch.to(device)
    logits = model(input_ids=moved.input_ids, attention_mask=moved.attention_mask).logits
    loss = masked_cross_entropy(logits, moved.labels)
    loss.backward()

    gradients: dict[str, torch.Tensor] = {}
    for key, module in iter_lora_modules(model):
        for factor in ("lora_A", "lora_B"):
            parameter = getattr(module, factor)["default"].weight
            if parameter.grad is not None:
                gradients[f"{key}.{factor}"] = parameter.grad.detach().float().cpu().clone()
    model.zero_grad(set_to_none=True)
    model.eval()
    return gradients


@torch.no_grad()
def probe_latency(
    model: Any, delta: AdapterDelta, batch: TokenizedBatch, device: str, *, repeats: int
) -> float:
    """Mean forward-pass wall time with this adapter installed (``Latency_i``)."""
    load_delta_into(model, delta)
    model.eval()
    moved = batch.to(device)
    model(input_ids=moved.input_ids, attention_mask=moved.attention_mask)  # warm-up
    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        model(input_ids=moved.input_ids, attention_mask=moved.attention_mask)
    return (time.perf_counter() - start) / max(1, repeats)

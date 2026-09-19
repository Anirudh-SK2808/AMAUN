"""M7.4 -- generating utility-network supervision (idea.md S9).

    y_i = [ s_post, F_post, C_post, Mem_post, Comm_post, L_post ]^T
    L_U = sum_i || g_omega(z_i) - y_i ||^2

The targets do not exist a priori: they come from **actually performing** candidate
integrations and measuring the outcome.  That is the expensive part of the whole project
(D-010): ``O(T * n)`` fusion runs for the bootstrap.

The sampling/proxy policy is therefore explicit and logged:

- ``max_pairs_per_event`` bounds how many candidate pairs are evaluated per task event.
- ``pair_sampling`` chooses which pairs (``stratified_by_feature`` spreads the sample across
  the feature space rather than clustering on near-duplicates).
- ``proxy_steps`` truncates the fusion run used to produce a target -- cheap but biased.
- ``proxy_validation_frac`` of sampled pairs are *also* run to full length, which is what makes
  the proxy bias measurable rather than assumed negligible.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from amaun.errors import ManagerError
from amaun.features.base import FeatureVector
from amaun.manager.utility_network import OUTCOME_HEADS
from amaun.utils.logging import get_logger

log = get_logger("manager.targets")


class PairSampling(str, enum.Enum):
    ALL = "all"
    RANDOM = "random"
    STRATIFIED_BY_FEATURE = "stratified_by_feature"


@dataclass
class SupervisionExample:
    """One ``(z_i, y_i)`` observation with its provenance."""

    candidate_id: str
    task_id: str
    features: FeatureVector
    targets: dict[str, float]
    proxy: bool
    proxy_steps: int | None
    beneficial: bool

    def target_vector(self, heads: Sequence[str] = OUTCOME_HEADS) -> list[float]:
        missing = [head for head in heads if head not in self.targets]
        if missing:
            raise ManagerError(f"supervision example is missing target heads {missing}")
        return [float(self.targets[head]) for head in heads]


@dataclass
class SupervisionSet:
    examples: list[SupervisionExample] = field(default_factory=list)
    n_pairs_considered: int = 0
    n_pairs_evaluated: int = 0
    proxy_bias: dict[str, float] = field(default_factory=dict)
    fit_split: str = "utility_train"

    def __len__(self) -> int:
        return len(self.examples)

    def tensors(self, normalizer: Any, run_mode: str, heads: Sequence[str] = OUTCOME_HEADS):
        if not self.examples:
            raise ManagerError("no supervision examples: the utility network cannot be trained")
        features = torch.tensor(
            [normalizer.transform(example.features, run_mode=run_mode) for example in self.examples],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [example.target_vector(heads) for example in self.examples], dtype=torch.float32
        )
        return features, targets

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_examples": len(self.examples),
            "n_pairs_considered": self.n_pairs_considered,
            "n_pairs_evaluated": self.n_pairs_evaluated,
            "n_proxy": sum(1 for e in self.examples if e.proxy),
            "n_full": sum(1 for e in self.examples if not e.proxy),
            "proxy_bias": dict(self.proxy_bias),
            "fit_split": self.fit_split,
            "beneficial_fraction": (
                sum(1 for e in self.examples if e.beneficial) / len(self.examples)
                if self.examples else None
            ),
        }


def sample_pairs(
    vectors: Mapping[str, FeatureVector],
    *,
    policy: PairSampling | str,
    max_pairs: int,
    seed: int,
) -> list[str]:
    """Choose which candidate pairs to actually evaluate (D-010)."""
    policy = PairSampling(policy)
    candidate_ids = sorted(vectors)
    if policy is PairSampling.ALL or len(candidate_ids) <= max_pairs:
        return candidate_ids
    if policy is PairSampling.RANDOM:
        return sorted(random.Random(seed).sample(candidate_ids, k=max_pairs))

    # stratified_by_feature: sort candidates along their first principal-ish axis (here, the
    # L2 norm of the feature vector) and take evenly spaced picks, so the sample spans the
    # feature space instead of clustering on near-duplicate candidates.
    def magnitude(adapter_id: str) -> float:
        values = vectors[adapter_id].values
        return sum(value * value for value in values) ** 0.5

    ordered = sorted(candidate_ids, key=magnitude)
    step = len(ordered) / max_pairs
    picked = {ordered[min(len(ordered) - 1, int(index * step))] for index in range(max_pairs)}
    return sorted(picked)


def measure_outcome_targets(
    *,
    primary_score_post: float,
    forgetting_post: float,
    confidence_post: float,
    memory_bytes_post: int,
    communication_bytes_post: int,
    long_term_loss_post: float,
    primary_score_baseline: float | None = None,
) -> tuple[dict[str, float], bool]:
    """Assemble ``y_i`` from *measured* post-integration outcomes, plus a beneficial flag.

    ``beneficial`` is what threshold selection (D-005) consumes: an integration counts as
    beneficial when it did not lose primary score relative to the not-integrating baseline and
    did not increase forgetting. Defining it from measurements, not from the prediction,
    is what keeps threshold selection independent of the network being thresholded.
    """
    targets = {
        "primary_score": float(primary_score_post),
        "forgetting": float(forgetting_post),
        "confidence": float(confidence_post),
        # Scaled to MB so the head's magnitude is comparable to the others; raw byte counts
        # would dominate the L2 objective purely through units.
        "memory": float(memory_bytes_post) / 1e6,
        "communication": float(communication_bytes_post) / 1e6,
        "long_term_loss": float(long_term_loss_post),
    }
    beneficial = forgetting_post <= 0.0 + 1e-9
    if primary_score_baseline is not None:
        beneficial = beneficial and primary_score_post >= primary_score_baseline - 1e-9
    return targets, bool(beneficial)


def estimate_proxy_bias(
    proxy: Sequence[SupervisionExample], full: Sequence[SupervisionExample]
) -> dict[str, float]:
    """Mean per-head difference between truncated-proxy and full-convergence targets.

    Reported so D-010's assumption (A-12) is measured rather than asserted.  A large bias here
    invalidates proxy-trained utility predictions and must appear in the results.
    """
    if not proxy or not full:
        return {}
    bias: dict[str, float] = {}
    for head in OUTCOME_HEADS:
        proxy_values = [e.targets[head] for e in proxy if head in e.targets]
        full_values = [e.targets[head] for e in full if head in e.targets]
        if proxy_values and full_values:
            bias[head] = (sum(proxy_values) / len(proxy_values)) - (sum(full_values) / len(full_values))
    return bias

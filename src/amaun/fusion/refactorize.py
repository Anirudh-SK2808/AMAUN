"""M8 -- how a fused adapter is stored (D-013).

idea.md S6.1 explicitly requires documenting whether the fused adapter is kept as a dense
effective update or re-factorised back to low rank, and does not choose.  All three options
are implemented, and the lossy one *measures* its error rather than assuming it is small:

- ``keep_components``: store components + alpha.  Exact, but storage grows linearly in ``m``,
  so it would undercut the growth-control claim if used for reported runs.
- ``dense_delta``: store the dense ``Delta_W``.  Exact, but ``O(d_out*d_in)`` per layer --
  far larger than the LoRA factors it replaces.
- ``refactorize_svd``: truncated SVD back to rank ``r``.  Keeps adapter size constant (which
  is what makes the storage claim honest) at the cost of an approximation, whose relative
  Frobenius error is recorded per layer in the fusion log.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Sequence

import torch

from amaun.adapters.delta import AdapterDelta, LayerDelta, combine_deltas
from amaun.errors import FusionError


class FusionStorage(str, enum.Enum):
    DENSE_DELTA = "dense_delta"
    REFACTORIZE_SVD = "refactorize_svd"
    KEEP_COMPONENTS = "keep_components"


@dataclass
class RefactorizationReport:
    """Per-layer approximation error, so the cost of D-013 is measured, not assumed."""

    storage: str
    target_rank: int | None
    per_layer_relative_error: dict[str, float] = field(default_factory=dict)

    @property
    def max_relative_error(self) -> float:
        return max(self.per_layer_relative_error.values(), default=0.0)

    @property
    def mean_relative_error(self) -> float:
        values = list(self.per_layer_relative_error.values())
        return sum(values) / len(values) if values else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "storage": self.storage,
            "target_rank": self.target_rank,
            "max_relative_error": self.max_relative_error,
            "mean_relative_error": self.mean_relative_error,
            "per_layer_relative_error": dict(self.per_layer_relative_error),
        }


def _svd_factorize(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Best rank-``r`` factorisation of ``delta`` and its relative Frobenius error.

    Splits the singular values as ``B = U sqrt(S)``, ``A = sqrt(S) V^T`` so the two factors
    have comparable scale, which keeps subsequent gradient magnitudes balanced.
    """
    delta32 = delta.to(torch.float32)
    u, s, vh = torch.linalg.svd(delta32, full_matrices=False)
    rank = min(rank, int(s.numel()))
    root = torch.sqrt(s[:rank])
    b = u[:, :rank] * root.unsqueeze(0)
    a = root.unsqueeze(1) * vh[:rank, :]
    reconstruction = b @ a
    denominator = torch.linalg.norm(delta32)
    error = float(torch.linalg.norm(delta32 - reconstruction) / denominator) if denominator > 0 else 0.0
    return a, b, error


def fuse_to_delta(
    components: Sequence[AdapterDelta],
    alphas: dict[str, torch.Tensor] | list[float],
    *,
    storage: FusionStorage | str,
    target_rank: int | None = None,
    adapter_id: str = "",
) -> tuple[AdapterDelta, RefactorizationReport]:
    """Collapse fused components into a single storable :class:`AdapterDelta`."""
    storage = FusionStorage(storage)
    exact = combine_deltas(list(components), alphas)
    exact.adapter_id = adapter_id or exact.adapter_id
    report = RefactorizationReport(storage=storage.value, target_rank=target_rank)

    if storage in (FusionStorage.KEEP_COMPONENTS, FusionStorage.DENSE_DELTA):
        # Both are exact; concatenated factors already represent the fused delta exactly,
        # and dense_delta differs only in the on-disk form, handled by the caller.
        return exact, report

    if target_rank is None:
        raise FusionError("refactorize_svd requires a target rank (fusion.refactorize_rank or r)")

    layers: dict[str, LayerDelta] = {}
    for key, layer in exact.items():
        a, b, error = _svd_factorize(layer.dense(), target_rank)
        layers[key] = LayerDelta(A=a, B=b, scaling=1.0)
        report.per_layer_relative_error[key] = error
    return AdapterDelta(layers=layers, adapter_id=exact.adapter_id), report

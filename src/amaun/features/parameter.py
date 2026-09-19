"""M6.2 -- parameter-structure features (idea.md S4.2).

Per layer ``l``, between the temporary adapter and candidate ``i``:

    d_F(l,i)    = || Delta_W_temp_l - Delta_W_i_l ||_F
    d_2(l,i)    = || vec(Delta_W_temp_l) - vec(Delta_W_i_l) ||_2
    v_l^i       = Var(vec(Delta_W_i_l))
    d_rank(l,i) = | rank(Delta_W_temp_l) - rank(Delta_W_i_l) |
    d_sv(l,i)   = || sigma(Delta_W_temp_l) - sigma(Delta_W_i_l) ||_2

All computed on the **effective update** ``Delta_W = scaling * B @ A``, never on the ``A``/``B``
factors separately.

Note that ``d_F`` and ``d_2`` are mathematically the same quantity (the Frobenius norm *is*
the L2 norm of the vectorisation).  idea.md S4.2 lists both, so both are emitted for
faithfulness; the utility network sees a perfectly collinear pair, which is harmless for a
regression target but is documented here so a reviewer does not mistake it for a bug.

Cost control: SVD on a 3584x18944 delta is expensive, so singular values are computed from
the much smaller ``r x r`` Gram factorisation -- exactly, not approximately.  ``Delta_W`` and
``(B A)`` share nonzero singular values with the ``r x r`` matrix ``R_B R_A`` obtained from
QR factorisations of ``B`` and ``A^T``, which costs ``O(r^2(d+r))`` instead of
``O(d_in d_out min(d_in,d_out))``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from amaun.adapters.delta import LayerDelta
from amaun.features.base import FeatureFamily


def effective_singular_values(layer: LayerDelta) -> torch.Tensor:
    """Exact nonzero singular values of ``scaling * B @ A`` without materialising it.

    ``B = Q_B R_B`` (``[d_out, r]``) and ``A^T = Q_A R_A`` (``[d_in, r]``) give
    ``B A = Q_B (R_B R_A^T) Q_A^T`` with ``Q_B``, ``Q_A`` orthonormal, so the singular values
    of ``B A`` are exactly those of the ``r x r`` matrix ``R_B R_A^T``.
    """
    b = layer.B.to(torch.float32)
    a = layer.A.to(torch.float32)
    _, r_b = torch.linalg.qr(b)              # [r, r]
    _, r_a = torch.linalg.qr(a.transpose(0, 1))   # [r, r]
    core = r_b @ r_a.transpose(0, 1)
    return torch.linalg.svdvals(core) * abs(layer.scaling)


def effective_frobenius_norm(layer: LayerDelta) -> float:
    """``||scaling * B A||_F`` computed via traces, at ``O(r^2 d)`` instead of ``O(d^2)``.

    ``||BA||_F^2 = tr(A^T B^T B A) = tr((B^T B)(A A^T))``, both factors being ``r x r``.
    """
    b = layer.B.to(torch.float32)
    a = layer.A.to(torch.float32)
    gram_b = b.transpose(0, 1) @ b       # [r, r]
    gram_a = a @ a.transpose(0, 1)       # [r, r]
    value = float(torch.sum(gram_b * gram_a.transpose(0, 1)))
    return abs(layer.scaling) * math.sqrt(max(0.0, value))


def effective_difference_norm(left: LayerDelta, right: LayerDelta) -> float:
    """``||Delta_left - Delta_right||_F`` without materialising either delta.

    The difference of two low-rank products is itself low-rank with the concatenated
    factors ``[s_l B_l | -s_r B_r] @ [A_l ; A_r]``, so the same Gram trick applies.
    """
    b = torch.cat([left.scaling * left.B.to(torch.float32), -right.scaling * right.B.to(torch.float32)], dim=1)
    a = torch.cat([left.A.to(torch.float32), right.A.to(torch.float32)], dim=0)
    gram_b = b.transpose(0, 1) @ b
    gram_a = a @ a.transpose(0, 1)
    return math.sqrt(max(0.0, float(torch.sum(gram_b * gram_a.transpose(0, 1)))))


def effective_variance(layer: LayerDelta) -> float:
    """``Var(vec(Delta_W))`` from norms only.

    ``Var = E[x^2] - (E[x])^2``; ``E[x^2] = ||Delta||_F^2 / n`` and
    ``E[x] = (1^T Delta 1) / n``, both computable from the factors.
    """
    b = layer.B.to(torch.float32)
    a = layer.A.to(torch.float32)
    d_out, d_in = layer.shape
    n = d_out * d_in
    frobenius_sq = (effective_frobenius_norm(layer)) ** 2
    total = float((b.sum(dim=0) * layer.scaling) @ a.sum(dim=1))
    mean = total / n
    return max(0.0, frobenius_sq / n - mean * mean)


def numerical_rank(singular_values: torch.Tensor, tolerance: float) -> int:
    if singular_values.numel() == 0:
        return 0
    cutoff = tolerance * float(singular_values.max())
    return int((singular_values > cutoff).sum())


class ParameterFeatures(FeatureFamily):
    name = "parameter"

    def _key_layout(self, aggregation: str, n_layers: int) -> list[str]:
        """The exact, data-independent key set this family emits."""
        keys: list[str] = []
        keys += self.aggregate_keys(aggregation, "d_F", n_layers=n_layers)
        keys += self.aggregate_keys("mean", "d_2", n_layers=n_layers)
        keys += self.aggregate_keys(aggregation, "var_cand", n_layers=n_layers)
        keys += self.aggregate_keys("mean", "d_rank", n_layers=n_layers)
        keys += self.aggregate_keys(aggregation, "d_sv", n_layers=n_layers)
        keys += ["norm_ratio", "n_shared_layers"]
        return keys

    def compute(self, ctx: Any) -> dict[str, float | None]:
        temp = ctx.temp_delta
        candidate = ctx.candidate_delta
        aggregation = str(self.config.get("aggregation", "mean_std"))
        tolerance = float(self.config.get("rank_tolerance", 1e-5))
        max_svd_layers = int(self.config.get("max_layers_for_svd", 32))
        declared_layers = int(self.config.get("n_layers_for_per_layer", 0))

        try:
            shared = temp.assert_alignable(candidate)
        except Exception:
            # Structurally incompatible adapters still get the full key layout, all unknown,
            # so the feature vector's width never depends on candidate compatibility.
            return {key: None for key in self._key_layout(aggregation, declared_layers)}

        frobenius, l2, variances, rank_gaps, sv_distances = [], [], [], [], []
        temp_norms, candidate_norms = [], []

        for index, key in enumerate(shared):
            temp_layer, candidate_layer = temp[key], candidate[key]
            difference = effective_difference_norm(temp_layer, candidate_layer)
            frobenius.append(difference)
            l2.append(difference)   # identical by definition; both listed in idea.md S4.2
            variances.append(effective_variance(candidate_layer))
            temp_norms.append(effective_frobenius_norm(temp_layer))
            candidate_norms.append(effective_frobenius_norm(candidate_layer))

            if index < max_svd_layers:
                temp_sv = effective_singular_values(temp_layer)
                candidate_sv = effective_singular_values(candidate_layer)
                rank_gaps.append(
                    abs(numerical_rank(temp_sv, tolerance) - numerical_rank(candidate_sv, tolerance))
                )
                width = min(temp_sv.numel(), candidate_sv.numel())
                sv_distances.append(
                    float(torch.linalg.norm(temp_sv[:width] - candidate_sv[:width]))
                )

        n_layers = declared_layers or len(shared)
        out: dict[str, float | None] = {}
        out.update(self.aggregate(frobenius, aggregation, "d_F", n_layers=n_layers))
        out.update(self.aggregate(l2, "mean", "d_2", n_layers=n_layers))
        out.update(self.aggregate(variances, aggregation, "var_cand", n_layers=n_layers))
        out.update(self.aggregate(rank_gaps, "mean", "d_rank", n_layers=n_layers))
        out.update(self.aggregate(sv_distances, aggregation, "d_sv", n_layers=n_layers))

        temp_total = math.sqrt(sum(v * v for v in temp_norms))
        candidate_total = math.sqrt(sum(v * v for v in candidate_norms))
        out["norm_ratio"] = self.finite(
            temp_total / candidate_total if candidate_total > 0 else None
        )
        out["n_shared_layers"] = float(len(shared))
        return out

"""M6.3 -- gradient/optimization features (idea.md S4.3).

    a_grad(i) = (g_t . g_i) / (||g_t|| ||g_i|| + eps)     alignment
    m_g(i)    = || g_t - g_i ||_2                         discrepancy
    v_g(i)    = Var(g_t - g_i)                            variance

The **same fixed reference minibatch** is used for both adapters of a pair (idea.md S4.3),
selected deterministically from ``(task_id, features.gradient.reference_seed)``.  Without
that, alignment values are not comparable across candidates and the feature is noise.

Note this is a *feature*, not a threshold (idea.md S4.3 is explicit: "gradient alignment
(feature, not a threshold)").  Cosine appears here only because alignment is definitionally
a cosine of gradients; it is one of ~25 inputs to the utility network.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from amaun.features.base import FeatureFamily


def flatten_gradients(named_gradients: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    """Flatten gradients in a fixed key order so two vectors are index-comparable."""
    chunks = [named_gradients[key].reshape(-1).to(torch.float32) for key in keys if key in named_gradients]
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


class GradientFeatures(FeatureFamily):
    name = "gradient"

    def compute(self, ctx: Any) -> dict[str, float | None]:
        temp_gradients = getattr(ctx, "temp_gradients", None)
        candidate_gradients = getattr(ctx, "candidate_gradients", None)
        if not temp_gradients or not candidate_gradients:
            return {
                "a_grad": None, "m_g": None, "v_g": None,
                "grad_norm_temp": None, "grad_norm_cand": None, "grad_norm_ratio": None,
            }

        shared_keys = sorted(set(temp_gradients) & set(candidate_gradients))
        g_t = flatten_gradients(temp_gradients, shared_keys)
        g_i = flatten_gradients(candidate_gradients, shared_keys)
        if g_t.numel() == 0 or g_t.numel() != g_i.numel():
            return {
                "a_grad": None, "m_g": None, "v_g": None,
                "grad_norm_temp": None, "grad_norm_cand": None, "grad_norm_ratio": None,
            }

        epsilon = 1e-8
        norm_t = float(torch.linalg.norm(g_t))
        norm_i = float(torch.linalg.norm(g_i))
        difference = g_t - g_i
        return {
            "a_grad": self.finite(float(torch.dot(g_t, g_i)) / (norm_t * norm_i + epsilon)),
            "m_g": self.finite(float(torch.linalg.norm(difference))),
            "v_g": self.finite(float(difference.var(unbiased=False)) if difference.numel() > 1 else 0.0),
            "grad_norm_temp": self.finite(norm_t),
            "grad_norm_cand": self.finite(norm_i),
            "grad_norm_ratio": self.finite(norm_t / norm_i if norm_i > epsilon else None),
        }

"""M6.1 -- semantic features (idea.md S4.1).

    d_task(i)   = ||e_t - e_i||_2^2                          Euclidean, NOT cosine
    d_hidden(i) = (1/N) sum_n ||h_t(x_n) - h_i(x_n)||_2^2     hidden-state discrepancy
    D_out(i)    = (1/N) sum_n KL( p_i(y|x_n) || p_temp(y|x_n) )   output divergence

idea.md S4.1 specifies Euclidean distance explicitly, and S16.2 forbids cosine similarity as
the decision rule.  Cosine may be included as *one additional feature* when
``features.semantic.include_cosine_as_feature`` is set -- it then enters the utility network
alongside 20+ others and cannot act as the rule.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from amaun.features.base import FeatureFamily


class SemanticFeatures(FeatureFamily):
    name = "semantic"

    def compute(self, ctx: Any) -> dict[str, float | None]:
        # Fixed key layout: the vector's width must not depend on whether an embedding happens
        # to be available for this particular candidate, or the normaliser fitted during
        # bootstrapping would reject vectors produced later in the run.
        include_cosine = bool(self.config.get("include_cosine_as_feature", False))
        out: dict[str, float | None] = {"d_task_sq": None, "d_task_l2": None}
        if include_cosine:
            out["cos_task"] = None

        task_embedding = getattr(ctx, "task_embedding", None)
        candidate_embedding = getattr(ctx, "candidate_task_embedding", None)
        if task_embedding is not None and candidate_embedding is not None:
            left = torch.as_tensor(task_embedding, dtype=torch.float32).flatten()
            right = torch.as_tensor(candidate_embedding, dtype=torch.float32).flatten()
            if left.numel() == right.numel() and left.numel() > 0:
                out["d_task_sq"] = self.finite(float(torch.sum((left - right) ** 2)))
                out["d_task_l2"] = self.finite(float(torch.linalg.norm(left - right)))
                if include_cosine:
                    denominator = float(torch.linalg.norm(left) * torch.linalg.norm(right))
                    out["cos_task"] = self.finite(
                        float(torch.dot(left, right)) / denominator if denominator > 0 else None
                    )

        out["d_hidden_sq"] = self.finite(getattr(ctx, "hidden_discrepancy", None))
        out["D_out_kl"] = self.finite(getattr(ctx, "output_kl", None))
        out["D_out_kl_reverse"] = self.finite(getattr(ctx, "output_kl_reverse", None))
        return out

"""M6.5 -- continual-learning features (idea.md S4.5).

    F_i  = (1/M) sum_j max(0, s_{j,pre} - s_{j,post})    average forgetting over prior tasks
    KR_i = 1 - F_i                                       knowledge retention
    P_i  = s_{new,post} - s_{new,pre}                     plasticity on the incoming task
    S_i  = 1 - F_i + rho * P_i                            stability-plasticity descriptor

All quantities use each task's ``primary_score`` (D-009), never "accuracy".  Scores come
from the repository's recorded history and from support-set probes -- **never from TEST**,
which is why this family takes precomputed scores rather than a dataset handle.
"""

from __future__ import annotations

from typing import Any

from amaun.features.base import FeatureFamily


class ContinualFeatures(FeatureFamily):
    name = "continual"

    def compute(self, ctx: Any) -> dict[str, float | None]:
        pre = getattr(ctx, "prior_scores_pre", None) or {}
        post = getattr(ctx, "prior_scores_post", None) or {}
        shared = sorted(set(pre) & set(post))

        if shared:
            drops = [max(0.0, float(pre[task]) - float(post[task])) for task in shared]
            forgetting = sum(drops) / len(drops)
            signed = [float(post[task]) - float(pre[task]) for task in shared]
            signed_transfer = sum(signed) / len(signed)
        else:
            forgetting = None
            signed_transfer = None

        new_pre = getattr(ctx, "new_task_score_pre", None)
        new_post = getattr(ctx, "new_task_score_post", None)
        plasticity = (
            float(new_post) - float(new_pre)
            if new_pre is not None and new_post is not None
            else None
        )

        rho = float(self.config.get("rho", 0.5))
        stability = (
            1.0 - forgetting + rho * plasticity
            if forgetting is not None and plasticity is not None
            else None
        )

        return {
            "F_i": self.finite(forgetting),
            "KR_i": self.finite(1.0 - forgetting if forgetting is not None else None),
            "P_i": self.finite(plasticity),
            "S_i": self.finite(stability),
            "signed_transfer": self.finite(signed_transfer),
            "n_prior_tasks": float(len(shared)),
        }

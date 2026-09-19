"""M6.4 -- performance-quality features (idea.md S4.4).

    primary_score     task-appropriate scalar in [0,1]   (D-009, replaces undefined "Acc")
    L_val_i           mean cross-entropy on held-out data
    C_i               mean max confidence
    H_i               mean predictive entropy
    ECE_i             expected calibration error, fixed bins

D-009: idea.md writes ``Acc_i``, but five of six tasks are open-ended generation where
accuracy is undefined.  Each task declares its own ``primary_score`` and that is what enters
the features, the utility targets, and every continual-learning metric.  Calibration
quantities (``C``, ``H``, ``ECE``) are computed over **next-token distributions on response
positions**, which is well defined for generative tasks, unlike class-label calibration.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from amaun.data.preprocess import response_mask
from amaun.features.base import FeatureFamily


def confidence_entropy_ece(
    logits: torch.Tensor, labels: torch.Tensor, *, bins: int
) -> dict[str, float]:
    """Token-level confidence, entropy, and ECE over response positions.

    ECE bins predictions by max-probability and compares bin accuracy (is the argmax token
    correct?) with bin confidence -- the standard definition, applied at token level.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = response_mask(labels)
    if mask.sum() == 0:
        return {"confidence": float("nan"), "entropy": float("nan"), "ece": float("nan"),
                "token_accuracy": float("nan")}

    probabilities = torch.softmax(shift_logits, dim=-1)
    max_prob, predicted = probabilities.max(dim=-1)
    log_probabilities = torch.log_softmax(shift_logits, dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1)

    flat_mask = mask.reshape(-1)
    confidence = max_prob.reshape(-1)[flat_mask]
    correct = (predicted.reshape(-1)[flat_mask] == shift_labels.reshape(-1)[flat_mask]).float()
    entropy_masked = entropy.reshape(-1)[flat_mask]

    total = confidence.numel()
    ece = 0.0
    edges = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        low, high = float(edges[index]), float(edges[index + 1])
        in_bin = (confidence > low) & (confidence <= high) if index > 0 else (confidence <= high)
        count = int(in_bin.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))

    return {
        "confidence": float(confidence.mean()),
        "entropy": float(entropy_masked.mean()),
        "ece": float(ece),
        "token_accuracy": float(correct.mean()),
    }


class PerformanceFeatures(FeatureFamily):
    name = "performance"

    def compute(self, ctx: Any) -> dict[str, float | None]:
        stats = getattr(ctx, "candidate_quality", None) or {}
        temp_stats = getattr(ctx, "temp_quality", None) or {}
        out: dict[str, float | None] = {
            "primary_score_cand": self.finite(stats.get("primary_score")),
            "L_val_cand": self.finite(stats.get("loss")),
            "C_cand": self.finite(stats.get("confidence")),
            "H_cand": self.finite(stats.get("entropy")),
            "ECE_cand": self.finite(stats.get("ece")),
            "token_acc_cand": self.finite(stats.get("token_accuracy")),
            "L_val_temp": self.finite(temp_stats.get("loss")),
            "C_temp": self.finite(temp_stats.get("confidence")),
            "H_temp": self.finite(temp_stats.get("entropy")),
            "token_acc_temp": self.finite(temp_stats.get("token_accuracy")),
        }
        # The candidate-minus-temp gap is what actually informs the decision: an adapter that
        # is merely good is not useful if the fresh adapter is much better on this task.
        if out["L_val_cand"] is not None and out["L_val_temp"] is not None:
            out["L_val_gap"] = out["L_val_cand"] - out["L_val_temp"]
        else:
            out["L_val_gap"] = None
        return out

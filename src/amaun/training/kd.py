"""M8 -- knowledge distillation (idea.md S6.4).

    q_old   = softmax(s_old / T)        teacher = the PRE-MERGE OLD ADAPTER
    q_merge = softmax(s_merge / T)      student = the fused adapter
    L_KD    = T^2 * (1/N_r) * sum_n KL( q_old(.|x_n) || q_merge(.|x_n) )
    L_total = L_new + lambda * L_KD

Three details are load-bearing:

**Teacher identity.**  idea.md S16.9 requires the teacher to be the pre-merge old adapter,
*not* the frozen backbone.  The preservation target is the behaviour that fusion would
otherwise overwrite; the backbone has no such behaviour to preserve.  ``TeacherCache``
records which adapter produced it, and the fusion loop refuses to proceed if the recorded
teacher is not the selected ``A_old``.

**Capture ordering.**  Teacher logits are captured *before* any fusion parameter update
(idea.md S6.7 step 5).  Capturing afterwards would distil the student against a
partially-merged model -- a silent, plausible-looking bug that would quietly weaken the KD
ablation.

**Masking.**  KD is computed on response positions only, with exactly the same shift and
mask as the task loss, so ``L_new`` and ``L_KD`` are defined over the same positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from amaun.data.preprocess import TokenizedBatch, response_mask
from amaun.errors import FusionError


@dataclass
class KDConfig:
    enabled: bool
    lam: float
    temperature: float
    teacher: str
    retention_examples: int
    top_p_truncation: float | None

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "KDConfig":
        required = ("enabled", "lambda", "temperature", "teacher", "retention_examples")
        missing = [key for key in required if key not in section]
        if missing:
            raise FusionError(f"kd config missing keys: {missing}")
        teacher = str(section["teacher"])
        if teacher != "pre_merge_old_adapter":
            raise FusionError(
                f"kd.teacher must be 'pre_merge_old_adapter' (idea.md S16.9); got '{teacher}'. "
                f"Distilling from the frozen backbone would preserve the wrong behaviour."
            )
        return cls(
            enabled=bool(section["enabled"]),
            lam=float(section["lambda"]),
            temperature=float(section["temperature"]),
            teacher=teacher,
            retention_examples=int(section["retention_examples"]),
            top_p_truncation=section.get("top_p_truncation"),
        )


@dataclass
class TeacherCache:
    """Cached teacher logits with the provenance needed to verify teacher identity."""

    teacher_adapter_id: str
    batches: list[TokenizedBatch]
    logits: list[torch.Tensor]
    temperature: float
    captured_before_fusion: bool
    top_p_truncation: float | None = None
    source_task_ids: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.batches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "teacher_adapter_id": self.teacher_adapter_id,
            "num_batches": len(self.batches),
            "temperature": self.temperature,
            "captured_before_fusion": self.captured_before_fusion,
            "top_p_truncation": self.top_p_truncation,
            "source_task_ids": list(self.source_task_ids),
        }


@torch.no_grad()
def capture_teacher(
    model: Any,
    batches: Sequence[TokenizedBatch],
    *,
    teacher_adapter_id: str,
    temperature: float,
    device: str,
    top_p_truncation: float | None = None,
    source_task_ids: Sequence[str] = (),
) -> TeacherCache:
    """Run the pre-merge old adapter over ``D_rep`` and cache its logits.

    Must be called while ``model`` still carries **only** the old adapter -- i.e. before
    :func:`amaun.fusion.layerwise.build_fused_model`.
    """
    model.eval()
    cached: list[torch.Tensor] = []
    kept: list[TokenizedBatch] = []
    for batch in batches:
        moved = batch.to(device)
        logits = model(input_ids=moved.input_ids, attention_mask=moved.attention_mask).logits
        # Only response positions are ever read by kd_loss, so only they are cached:
        # ``[n_response_positions, vocab]`` instead of ``[batch, seq, vocab]``.  At a 151 936-token
        # vocabulary the full fp32 cache was ~778 MB per 2-example batch of 640 tokens, almost all
        # of it prompt positions the loss masks out.
        mask = response_mask(moved.labels)
        cached.append(logits[:, :-1, :][mask].detach().to(torch.float32).cpu())
        del logits
        kept.append(batch)
    return TeacherCache(
        teacher_adapter_id=teacher_adapter_id,
        batches=kept,
        logits=cached,
        temperature=temperature,
        captured_before_fusion=True,
        top_p_truncation=top_p_truncation,
        source_task_ids=list(source_task_ids),
    )


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """``T^2 * mean_n KL(q_old || q_merge)`` over response positions only.

    The ``T^2`` factor (Hinton et al.) restores the gradient scale that dividing logits by
    ``T`` removes, so ``lambda`` means the same thing across temperatures.
    """
    if temperature <= 0:
        raise FusionError("KD temperature must be positive")
    shift_student = student_logits[:, :-1, :]
    mask = response_mask(labels).to(shift_student.device)
    if mask.sum() == 0:
        return torch.zeros((), device=shift_student.device, dtype=shift_student.dtype)

    # Both operands are reduced to the masked positions before the fp32 softmaxes, which is
    # numerically identical to masking afterwards and avoids three full-size fp32 copies.
    student = shift_student[mask]
    if teacher_logits.dim() == 2:
        # Compact cache from capture_teacher: already ``[n_response_positions, vocab]``.
        teacher = teacher_logits.to(student.device)
    else:
        teacher = teacher_logits[:, :-1, :].to(student.device)[mask]
    if teacher.shape != student.shape:
        raise FusionError(
            f"KD teacher/student position mismatch: teacher {tuple(teacher.shape)} vs student "
            f"{tuple(student.shape)}; the teacher must be captured on the same retention batch"
        )

    student_log_probs = torch.log_softmax(student.float() / temperature, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher.float() / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    # KL(teacher || student), summed over the vocabulary, averaged over masked positions.
    per_position = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return (temperature**2) * per_position.mean()


def verify_teacher(cache: TeacherCache, expected_adapter_id: str) -> None:
    """Invariants I8.6 and F8: teacher identity and capture ordering."""
    if cache.teacher_adapter_id != expected_adapter_id:
        raise FusionError(
            f"KD teacher mismatch: cache holds logits from '{cache.teacher_adapter_id}' but the "
            f"selected pre-merge adapter is '{expected_adapter_id}' (idea.md S16.9)"
        )
    if not cache.captured_before_fusion:
        raise FusionError(
            "KD teacher logits were not captured before fusion began; the student would be "
            "distilled against a partially-merged model (idea.md S6.7 step 5)"
        )

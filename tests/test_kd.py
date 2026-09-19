"""M8 -- knowledge distillation (criteria F8, F9)."""

import pytest
import torch

from amaun.data.preprocess import IGNORE_INDEX, TokenizedBatch
from amaun.errors import FusionError
from amaun.training.kd import KDConfig, TeacherCache, kd_loss, verify_teacher


def _batch(batch=2, seq=6, n_prompt=3):
    labels = torch.full((batch, seq), IGNORE_INDEX)
    labels[:, n_prompt:] = torch.randint(1, 50, (batch, seq - n_prompt))
    return TokenizedBatch(
        input_ids=torch.randint(1, 50, (batch, seq)),
        attention_mask=torch.ones(batch, seq, dtype=torch.long),
        labels=labels,
        prompt_lengths=[n_prompt] * batch,
    )


def test_kd_config_rejects_any_teacher_but_the_pre_merge_adapter():
    """idea.md S16.9: distilling from the frozen backbone preserves the wrong behaviour."""
    good = {"enabled": True, "lambda": 1.0, "temperature": 2.0,
            "teacher": "pre_merge_old_adapter", "retention_examples": 4}
    assert KDConfig.from_config(good).teacher == "pre_merge_old_adapter"
    for bad_teacher in ("frozen_backbone", "adapter_ensemble", "best_adapter"):
        with pytest.raises(FusionError):
            KDConfig.from_config({**good, "teacher": bad_teacher})


def test_teacher_identity_is_verified():
    cache = TeacherCache(
        teacher_adapter_id="A_old", batches=[], logits=[], temperature=2.0,
        captured_before_fusion=True,
    )
    verify_teacher(cache, "A_old")
    with pytest.raises(FusionError, match="teacher mismatch"):
        verify_teacher(cache, "A_other")


def test_teacher_must_be_captured_before_fusion():
    """Invariant I8.6: a teacher captured after fusion distils the student against itself."""
    cache = TeacherCache(
        teacher_adapter_id="A_old", batches=[], logits=[], temperature=2.0,
        captured_before_fusion=False,
    )
    with pytest.raises(FusionError, match="before fusion"):
        verify_teacher(cache, "A_old")


def test_kd_loss_is_zero_for_identical_distributions():
    batch = _batch()
    logits = torch.randn(2, 6, 40)
    assert float(kd_loss(logits, logits.clone(), batch.labels, temperature=2.0)) == pytest.approx(0.0, abs=1e-5)


def test_kd_loss_is_positive_for_different_distributions():
    batch = _batch()
    student = torch.randn(2, 6, 40)
    teacher = torch.randn(2, 6, 40) * 3
    assert float(kd_loss(student, teacher, batch.labels, temperature=2.0)) > 0


def test_kd_loss_ignores_prompt_positions():
    """L_new and L_KD must be defined over the same positions."""
    batch = _batch()
    student = torch.randn(2, 6, 40)
    teacher = torch.randn(2, 6, 40)
    reference = float(kd_loss(student, teacher, batch.labels, temperature=2.0))

    # Perturb the teacher only where labels are ignored (aligned to logits[:, :-1]).
    perturbed = teacher.clone()
    ignored = batch.labels[:, 1:] == IGNORE_INDEX
    padded = torch.zeros(teacher.shape[:2], dtype=torch.bool)
    padded[:, :-1] = ignored
    perturbed[..., 0] = perturbed[..., 0] + 50.0 * padded.to(perturbed.dtype)
    assert float(kd_loss(student, perturbed, batch.labels, temperature=2.0)) == pytest.approx(
        reference, abs=1e-4
    )


def test_temperature_squared_scaling_is_applied():
    """The T^2 factor keeps lambda's meaning stable across temperatures (Hinton et al.)."""
    batch = _batch()
    student = torch.randn(2, 6, 40)
    teacher = torch.randn(2, 6, 40)
    at_one = float(kd_loss(student, teacher, batch.labels, temperature=1.0))
    at_four = float(kd_loss(student, teacher, batch.labels, temperature=4.0))
    # With T^2 scaling the losses stay the same order of magnitude; without it the T=4 loss
    # would be ~16x smaller.
    assert at_four > at_one / 16.0


def test_zero_mask_returns_zero_rather_than_nan():
    labels = torch.full((2, 6), IGNORE_INDEX)
    value = kd_loss(torch.randn(2, 6, 40), torch.randn(2, 6, 40), labels, temperature=2.0)
    assert float(value) == 0.0


def test_nonpositive_temperature_is_rejected():
    batch = _batch()
    with pytest.raises(FusionError):
        kd_loss(torch.randn(2, 6, 40), torch.randn(2, 6, 40), batch.labels, temperature=0.0)

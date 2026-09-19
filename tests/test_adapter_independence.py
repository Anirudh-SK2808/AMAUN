"""Criterion C4 -- the six adapters are trained INDEPENDENTLY from the same theta_0.

Brief S1 forbids the chain ``Base -> A1 -> A2 -> ...``.  Reviewer falsification question 2
asks whether adapter *k* inherits state from adapter *k-1*.  The strongest available evidence
is that training the same adapter in two different orders produces bit-identical weights.
"""

import pytest

from amaun.data.splits import trainer_guard
from amaun.models.backbone import backbone_fingerprint
from amaun.training.lora_trainer import TrainingConfig, train_adapter


def _hash(cfg, backbone, task, splits, lora_spec, adapter_id):
    delta, _ = train_adapter(
        backbone=backbone, task=task, guard=trainer_guard(splits[task.task_id]),
        lora_spec=lora_spec, train_cfg=TrainingConfig.from_config(cfg.section("training")),
        resolved=cfg, adapter_id=adapter_id,
    )
    return delta.content_hash()


def test_training_order_does_not_affect_the_result(session_cfg, backbone, tasks, splits, lora_spec):
    first, second = tasks[0], tasks[1]

    # Order A: second, then first.
    hash_second_first = _hash(session_cfg, backbone, second, splits, lora_spec, "ind-B")
    hash_first_after = _hash(session_cfg, backbone, first, splits, lora_spec, "ind-A")

    # Order B: first, then second.
    hash_first_alone = _hash(session_cfg, backbone, first, splits, lora_spec, "ind-A")
    hash_second_after = _hash(session_cfg, backbone, second, splits, lora_spec, "ind-B")

    assert hash_first_after == hash_first_alone, "adapter A depends on training order"
    assert hash_second_first == hash_second_after, "adapter B depends on training order"


def test_different_tasks_give_different_adapters(session_cfg, backbone, tasks, splits, lora_spec):
    a = _hash(session_cfg, backbone, tasks[0], splits, lora_spec, "dist-A")
    b = _hash(session_cfg, backbone, tasks[1], splits, lora_spec, "dist-B")
    assert a != b


def test_backbone_is_the_same_theta_zero_for_every_adapter(
    session_cfg, backbone, tasks, splits, lora_spec
):
    """The fan-out requirement: every adapter starts from the identical frozen backbone."""
    fingerprints = []
    for task in tasks[:3]:
        fingerprints.append(backbone_fingerprint(backbone.model))
        _hash(session_cfg, backbone, task, splits, lora_spec, f"theta0-{task.task_id}")
        fingerprints.append(backbone_fingerprint(backbone.model))
    assert len(set(fingerprints)) == 1, "theta_0 differed between adapters"

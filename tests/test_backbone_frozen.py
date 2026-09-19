"""M1 -- the frozen-backbone invariant (criteria C1, C2, C7)."""

import pytest
import torch

from amaun.adapters.lora import ensure_lora
from amaun.errors import BackboneMutatedError, BackboneNotFrozenError
from amaun.models.backbone import (
    BackboneCache,
    adapter_parameter_names,
    assert_backbone_frozen,
    assert_fingerprint_unchanged,
    backbone_fingerprint,
)


def test_backbone_loads_fully_frozen(backbone):
    """No BACKBONE parameter may be trainable.

    Adapter parameters are excluded: the session-scoped backbone may already carry a LoRA
    injection from another test, and LoRA factors are legitimately trainable. The invariant is
    about theta, which is what ``assert_backbone_frozen`` checks.
    """
    from amaun.models.backbone import _is_adapter_parameter

    trainable = [
        name for name, param in backbone.model.named_parameters()
        if param.requires_grad and not _is_adapter_parameter(name)
    ]
    assert trainable == []
    assert_backbone_frozen(backbone.model)
    assert backbone.frozen_parameter_count > 0


def test_freeze_assertion_fires_on_a_trainable_backbone_param(backbone):
    param = next(iter(backbone.model.parameters()))
    param.requires_grad_(True)
    try:
        with pytest.raises(BackboneNotFrozenError):
            assert_backbone_frozen(backbone.model)
    finally:
        param.requires_grad_(False)


def test_fingerprint_stable_across_lora_injection(backbone, lora_spec):
    before = backbone_fingerprint(backbone.model)
    ensure_lora(backbone.model, lora_spec)
    assert backbone_fingerprint(backbone.model) == before


def test_fingerprint_detects_a_real_weight_change(backbone):
    before = backbone_fingerprint(backbone.model)
    target = next(p for n, p in backbone.model.named_parameters() if "lora" not in n.lower())
    # Keep an exact copy: add_(1.0) followed by sub_(1.0) is NOT lossless in floating point
    # for small weights, which would leave the session backbone subtly perturbed.
    original = target.detach().clone()
    with torch.no_grad():
        target.add_(1.0)
    try:
        assert backbone_fingerprint(backbone.model) != before
        with pytest.raises(BackboneMutatedError):
            assert_fingerprint_unchanged(backbone.model, before, phase="test")
    finally:
        with torch.no_grad():
            target.copy_(original)
    assert backbone_fingerprint(backbone.model) == before


def test_adapter_parameters_are_recognised(backbone, lora_spec):
    model = ensure_lora(backbone.model, lora_spec)
    names = adapter_parameter_names(model)
    assert names, "no adapter parameters detected"
    assert all("lora" in n.lower() or "fusion_logit" in n.lower() for n in names)
    # Everything trainable must be an adapter parameter and nothing else.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable <= names


def test_only_one_backbone_resident(session_cfg):
    """Invariant I1.4 / development rule 6: never duplicate the 7B model."""
    from amaun.models.backbone import load_backbone

    first = load_backbone(session_cfg)
    second = load_backbone(session_cfg)
    assert first is second
    assert BackboneCache.resident_count() == 1

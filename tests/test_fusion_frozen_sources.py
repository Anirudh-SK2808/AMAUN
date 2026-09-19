"""Criterion F6 / invariant I8.4 -- fusion must not mutate the source adapters."""

import pytest
import torch
import torch.nn as nn

from amaun.errors import FusionSourceMutatedError
from amaun.fusion.layerwise import (
    FusionConfig,
    assert_sources_unmutated,
    build_fused_model,
    build_optimizer,
    teardown_fused_model,
)
from tests.conftest import make_delta


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(8, 6, bias=False)
        self.l1 = nn.Linear(8, 6, bias=False)


def _cfg(**overrides):
    base = dict(
        parameterisation="softmax_k", granularity="layerwise", init_value=0.5,
        alpha_learning_rate=0.05, storage="keep_components", refactorize_rank=None,
        train_lora_params=True, freeze_source_adapters=True, steps=3, lora_learning_rate=0.01,
    )
    base.update(overrides)
    return FusionConfig(**base)


def _two_components():
    components = [make_delta(s, f"c{s}", rank=4, layers=2, d_in=8, d_out=6) for s in (1, 2)]
    for component in components:
        component.layers = {k: component.layers[k] for k in ("l0", "l1")}
    return components


def test_optimising_fusion_leaves_sources_byte_identical():
    model = _Tiny()
    components = _two_components()
    before = [component.content_hash() for component in components]

    state = build_fused_model(model, components, _cfg())
    try:
        optimizer = build_optimizer(state, _cfg())
        x = torch.randn(4, 8)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            model.l0(x).pow(2).mean().backward()
            optimizer.step()
    finally:
        teardown_fused_model(state)

    after = [component.content_hash() for component in components]
    assert before == after
    assert_sources_unmutated(state, components)   # must not raise


def test_source_mutation_is_detected():
    model = _Tiny()
    components = _two_components()
    state = build_fused_model(model, components, _cfg())
    teardown_fused_model(state)
    with torch.no_grad():
        components[0].layers["l0"].B.add_(1.0)
    with pytest.raises(FusionSourceMutatedError):
        assert_sources_unmutated(state, components)


def test_teardown_restores_the_original_modules():
    model = _Tiny()
    original_l0 = model.l0
    state = build_fused_model(model, _two_components(), _cfg())
    assert model.l0 is not original_l0
    teardown_fused_model(state)
    assert model.l0 is original_l0

"""Criterion F3 / invariant I8.1 -- ``alpha_l = 0.5`` exactly at initialisation.

idea.md S6.1 and S10 fix equal-contribution initialisation and state it is NOT tunable.
"""

import pytest
import torch
import torch.nn as nn

from amaun.adapters.delta import LayerDelta
from amaun.errors import FusionError
from amaun.fusion.fused_module import FusedLoRALinear
from amaun.fusion.layerwise import FusionConfig, build_fused_model, teardown_fused_model
from tests.conftest import make_delta


def _components(n):
    generator = torch.Generator().manual_seed(0)
    return [
        LayerDelta(A=torch.randn(4, 8, generator=generator),
                   B=torch.randn(6, 4, generator=generator), scaling=1.0)
        for _ in range(n)
    ]


@pytest.mark.parametrize("parameterisation", ["softmax_k", "sigmoid_pair"])
def test_pairwise_alpha_is_exactly_one_half(parameterisation):
    base = nn.Linear(8, 6, bias=False)
    fused = FusedLoRALinear(base, _components(2), parameterisation=parameterisation, init_value=0.5)
    alpha = fused.alpha_values()
    assert alpha == pytest.approx([0.5, 0.5], abs=1e-7)


@pytest.mark.parametrize("m", [2, 3, 4, 6])
def test_top_k_alpha_is_exactly_equal_contribution(m):
    base = nn.Linear(8, 6, bias=False)
    fused = FusedLoRALinear(base, _components(m), init_value=1.0 / m)
    assert fused.alpha_values() == pytest.approx([1.0 / m] * m, abs=1e-6)


def test_wrong_init_value_is_rejected():
    """A config that contradicts idea.md must fail loudly, not be reinterpreted."""
    base = nn.Linear(8, 6, bias=False)
    with pytest.raises(FusionError):
        FusedLoRALinear(base, _components(2), init_value=0.7)


def test_pairwise_config_must_declare_zero_point_five():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(8, 6, bias=False)

    components = [make_delta(s, f"c{s}", rank=4, layers=1, d_in=8, d_out=6) for s in (1, 2)]
    for component in components:
        component.layers = {"l0": component.layers["l0"]}
    cfg = FusionConfig(
        parameterisation="softmax_k", granularity="layerwise", init_value=0.4,
        alpha_learning_rate=0.01, storage="keep_components", refactorize_rank=None,
        train_lora_params=True, freeze_source_adapters=True, steps=1, lora_learning_rate=0.001,
    )
    with pytest.raises(FusionError, match="must be 0.5"):
        build_fused_model(Tiny(), components, cfg)


def test_alpha_at_init_makes_fusion_the_plain_average():
    """At alpha = 0.5 the fused delta must equal the arithmetic mean of the components."""
    base = nn.Linear(8, 6, bias=False)
    components = _components(2)
    fused = FusedLoRALinear(base, components, init_value=0.5)
    expected = 0.5 * components[0].dense() + 0.5 * components[1].dense()
    assert torch.allclose(fused.dense_delta(), expected, atol=1e-5)

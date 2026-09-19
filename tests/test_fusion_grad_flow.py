"""Criterion F5 / invariant I8.5 -- ``a_l`` really receives gradient.

A detached ``alpha`` is the single most plausible silent bug in this module: everything would
run, the loss would fall, ``alpha`` would stay at 0.5 forever, and the "learnable layer-wise
fusion" contribution (H2) would be silently false while the ablation A3 showed no difference.
"""

import torch
import torch.nn as nn

from amaun.adapters.delta import LayerDelta
from amaun.fusion.fused_module import FusedLoRALinear
from amaun.fusion.layerwise import FusionConfig, build_fused_model, build_optimizer, teardown_fused_model
from tests.conftest import make_delta


def _components(n=2):
    generator = torch.Generator().manual_seed(0)
    return [
        LayerDelta(A=torch.randn(4, 8, generator=generator),
                   B=torch.randn(6, 4, generator=generator), scaling=1.0)
        for _ in range(n)
    ]


def test_alpha_receives_nonzero_gradient():
    base = nn.Linear(8, 6, bias=False)
    fused = FusedLoRALinear(base, _components(2), init_value=0.5)
    loss = fused(torch.randn(4, 8)).pow(2).mean()
    loss.backward()
    assert fused.fusion_logit.grad is not None, "alpha is detached from the graph"
    assert float(fused.fusion_logit.grad.abs().sum()) > 0


def test_alpha_moves_under_optimisation():
    base = nn.Linear(8, 6, bias=False)
    fused = FusedLoRALinear(base, _components(2), init_value=0.5)
    optimizer = torch.optim.AdamW([fused.fusion_logit], lr=0.1)
    x = torch.randn(4, 8)
    target = torch.randn(4, 6)
    before = fused.alpha_values()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(fused(x), target).backward()
        optimizer.step()
    after = fused.alpha_values()
    assert any(abs(a - b) > 1e-4 for a, b in zip(before, after)), "alpha did not move"
    assert abs(sum(after) - 1.0) < 1e-5


def test_lora_and_alpha_are_in_separate_optimiser_groups():
    """idea.md S6.5 optimises phi_merge and a jointly but with distinct learning rates."""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(8, 6, bias=False)

    components = [make_delta(s, f"c{s}", rank=4, layers=1, d_in=8, d_out=6) for s in (1, 2)]
    for component in components:
        component.layers = {"l0": component.layers["l0"]}
    cfg = FusionConfig(
        parameterisation="softmax_k", granularity="layerwise", init_value=0.5,
        alpha_learning_rate=0.05, storage="keep_components", refactorize_rank=None,
        train_lora_params=True, freeze_source_adapters=True, steps=1, lora_learning_rate=0.001,
    )
    state = build_fused_model(Tiny(), components, cfg)
    try:
        optimizer = build_optimizer(state, cfg)
        names = {group.get("name"): group["lr"] for group in optimizer.param_groups}
        assert names["lora"] == 0.001
        assert names["alpha"] == 0.05
    finally:
        teardown_fused_model(state)


def test_only_the_candidate_clone_is_trainable():
    """Invariant I8.4: the repository's adapter must not be an optimisation target."""
    base = nn.Linear(8, 6, bias=False)
    fused = FusedLoRALinear(base, _components(2), trainable_component_indices=(1,), init_value=0.5)
    assert isinstance(fused.component_A(1), nn.Parameter)
    assert not isinstance(fused.component_A(0), nn.Parameter)   # the OLD adapter is a buffer
    for param in fused.base.parameters():
        assert not param.requires_grad

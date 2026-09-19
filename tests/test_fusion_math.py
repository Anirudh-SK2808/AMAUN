"""M8 -- fusion correctness (criteria F1, F2, F4, F7, F10)."""

import pytest
import torch
import torch.nn as nn

from amaun.adapters.delta import LayerDelta
from amaun.errors import FusionError
from amaun.fusion.fused_module import AlphaParameterisation, FusedLoRALinear
from amaun.fusion.layerwise import FusionConfig, build_fused_model, teardown_fused_model
from amaun.fusion.refactorize import FusionStorage, fuse_to_delta
from tests.conftest import make_delta


def _components(n: int = 2, rank: int = 4, d_in: int = 8, d_out: int = 6):
    generator = torch.Generator().manual_seed(0)
    return [
        LayerDelta(
            A=torch.randn(rank, d_in, generator=generator),
            B=torch.randn(d_out, rank, generator=generator),
            scaling=1.0 + index,
        )
        for index in range(n)
    ]


def test_delta_level_fusion_equivalence():
    """Criterion F2 / invariant I8.3: the factored forward equals the dense definition."""
    base = nn.Linear(8, 6, bias=False)
    for n in (2, 3, 4):
        fused = FusedLoRALinear(base, _components(n), init_value=1.0 / n)
        x = torch.randn(5, 8)
        expected = base(x) + x @ fused.dense_delta().T
        assert torch.allclose(fused(x), expected, atol=1e-5), f"m={n} mismatch"


def test_coefficients_are_normalised_and_bounded():
    """Invariant I8.2, at initialisation and after arbitrary logit values."""
    base = nn.Linear(8, 6, bias=False)
    for n in (2, 3, 5):
        fused = FusedLoRALinear(base, _components(n), init_value=1.0 / n)
        # Moderate logits: alpha must be strictly interior, as the (0,1) parameterisation claims.
        for logits in (torch.zeros(n), torch.randn(n)):
            with torch.no_grad():
                fused.fusion_logit.copy_(logits)
            alpha = fused.alpha().detach()
            assert pytest.approx(float(alpha.sum()), abs=1e-5) == 1.0
            assert bool((alpha > 0).all()) and bool((alpha < 1).all())

        # Extreme logits: float32 softmax SATURATES to exactly 1.0/0.0, so the bound is [0,1]
        # numerically even though the parameterisation is (0,1) mathematically. Normalisation --
        # the property fusion actually depends on -- still holds exactly. Documented in
        # fusion/fused_module.py so a saturated coefficient is not mistaken for a bug.
        with torch.no_grad():
            fused.fusion_logit.copy_(torch.tensor([60.0] + [-60.0] * (n - 1)))
        alpha = fused.alpha().detach()
        assert pytest.approx(float(alpha.sum()), abs=1e-5) == 1.0
        assert bool((alpha >= 0).all()) and bool((alpha <= 1).all())


def test_sigmoid_pair_equals_softmax_at_m_two():
    """D-004: the generalisation reduces exactly to idea.md's pairwise form."""
    base = nn.Linear(8, 6, bias=False)
    components = _components(2)
    pair = FusedLoRALinear(base, components, parameterisation="sigmoid_pair", init_value=0.5)
    softmax = FusedLoRALinear(base, components, parameterisation="softmax_k", init_value=0.5)
    for value in (0.0, 0.7, -1.3, 4.0):
        with torch.no_grad():
            pair.fusion_logit.copy_(torch.tensor([value]))
            softmax.fusion_logit.copy_(torch.tensor([value, 0.0]))
        assert torch.allclose(pair.alpha(), softmax.alpha(), atol=1e-6)


def test_sigmoid_pair_rejects_more_than_two_components():
    base = nn.Linear(8, 6, bias=False)
    with pytest.raises(FusionError):
        FusedLoRALinear(base, _components(3), parameterisation="sigmoid_pair", init_value=1 / 3)


def test_global_alpha_shares_one_coefficient_across_layers():
    """Criterion F7 / ablation A8."""
    generator = torch.Generator().manual_seed(1)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(8, 6, bias=False)
            self.l1 = nn.Linear(8, 6, bias=False)

    for granularity, expect_shared in (("global", True), ("layerwise", False)):
        model = Tiny()
        components = [
            make_delta(seed, f"c{seed}", rank=4, layers=2, d_in=8, d_out=6) for seed in (1, 2)
        ]
        for component in components:
            component.layers = {"l0": component.layers["l0"], "l1": component.layers["l1"]}
        cfg = FusionConfig(
            parameterisation="softmax_k", granularity=granularity, init_value=0.5,
            alpha_learning_rate=0.01, storage="keep_components", refactorize_rank=None,
            train_lora_params=True, freeze_source_adapters=True, steps=1, lora_learning_rate=0.001,
        )
        state = build_fused_model(model, components, cfg)
        try:
            unique = {id(m.fusion_logit) for m in state.modules.values()}
            assert (len(unique) == 1) is expect_shared
            assert len(state.alpha_parameters()) == (1 if expect_shared else 2)
        finally:
            teardown_fused_model(state)


def test_fusion_parameter_cost_is_reported():
    """Criterion F10 / invariant I8.7: the extra trainable parameters must be disclosed."""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(8, 6, bias=False)

    components = [make_delta(s, f"c{s}", rank=4, layers=1, d_in=8, d_out=6) for s in (1, 2)]
    for component in components:
        component.layers = {"l0": component.layers["l0"]}
    cfg = FusionConfig(
        parameterisation="softmax_k", granularity="layerwise", init_value=0.5,
        alpha_learning_rate=0.01, storage="keep_components", refactorize_rank=None,
        train_lora_params=True, freeze_source_adapters=True, steps=1, lora_learning_rate=0.001,
    )
    state = build_fused_model(Tiny(), components, cfg)
    try:
        counts = state.trainable_parameter_count()
        assert counts["alpha_parameters"] == 2       # one layer x m=2 logits
        assert counts["num_fused_layers"] == 1
        assert counts["total"] == counts["alpha_parameters"] + counts["lora_parameters"]
    finally:
        teardown_fused_model(state)


def test_refactorization_is_exact_at_full_rank_and_reports_error_when_truncated():
    """D-013: the lossy storage option must MEASURE its loss."""
    a, b = make_delta(1, "a", rank=4), make_delta(2, "b", rank=4)
    alphas = {key: torch.tensor([0.6, 0.4]) for key in a.keys()}

    exact, report = fuse_to_delta([a, b], alphas, storage=FusionStorage.KEEP_COMPONENTS)
    for key in a.keys():
        expected = 0.6 * a[key].dense() + 0.4 * b[key].dense()
        assert torch.allclose(exact[key].dense(), expected, atol=1e-5)
    assert report.max_relative_error == 0.0

    full, full_report = fuse_to_delta(
        [a, b], alphas, storage=FusionStorage.REFACTORIZE_SVD, target_rank=8
    )
    assert full_report.max_relative_error < 1e-4      # rank 8 captures the rank-8 sum exactly

    truncated, trunc_report = fuse_to_delta(
        [a, b], alphas, storage=FusionStorage.REFACTORIZE_SVD, target_rank=2
    )
    assert truncated.rank == 2
    assert trunc_report.max_relative_error > 0.0      # the cost is real and recorded
    assert set(trunc_report.per_layer_relative_error) == set(a.keys())

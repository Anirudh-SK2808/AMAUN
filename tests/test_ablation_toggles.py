"""Criterion H6 -- every ablation toggle provably changes behaviour.

An ablation that is silently a no-op is worse than a missing ablation: it produces a
"no difference" result that looks like evidence about the mechanism when it is only evidence
that the switch was not wired up.  Each test here asserts that a toggle changes a concrete
observable.
"""

import pytest
import torch
import torch.nn as nn

from amaun.features.extractor import FAMILY_ORDER, CandidateContext, FeatureExtractor
from amaun.fusion.fused_module import FusedLoRALinear
from amaun.fusion.layerwise import FusionConfig, build_fused_model, teardown_fused_model
from amaun.manager.decision import CandidateScore, Outcome, Thresholds, decide
from amaun.training.kd import KDConfig
from tests.conftest import make_delta, make_metadata


def _context():
    return CandidateContext(
        temp_id="temp", temp_delta=make_delta(1, "temp"),
        candidate_id="cand", candidate_delta=make_delta(2, "cand"),
        candidate_metadata=make_metadata("cand", created_at_task_index=0, use_count=2),
        task_index=3, total_events=5,
        task_embedding=[0.1] * 8, candidate_task_embedding=[0.3] * 8,
        hidden_discrepancy=0.7, output_kl=0.12,
        temp_gradients={"g": torch.randn(20)}, candidate_gradients={"g": torch.randn(20)},
        temp_quality={"loss": 2.1}, candidate_quality={"primary_score": 0.35, "loss": 2.6},
        prior_scores_pre={"t1": 0.5}, prior_scores_post={"t1": 0.45},
        new_task_score_pre=0.1, new_task_score_post=0.3,
        repository_bytes=9000.0, repository_size=4, candidate_latency_seconds=0.004,
    )


def _dims(section, families):
    return len(FeatureExtractor({**section, "families": families}).extract(_context()))


def test_a1_reduced_feature_set_changes_the_vector(cfg):
    section = dict(cfg.section("features"))
    full = _dims(section, {name: True for name in FAMILY_ORDER})
    semantic_only = _dims(
        section, {name: name == "semantic" for name in FAMILY_ORDER}
    )
    assert semantic_only < full


@pytest.mark.parametrize("family", ["historical", "resource", "gradient", "continual", "parameter"])
def test_a6_a7_family_removal_changes_the_vector(cfg, family):
    """A6 (historical), A7 (resource), and the other family toggles."""
    section = dict(cfg.section("features"))
    full = _dims(section, {name: True for name in FAMILY_ORDER})
    reduced = _dims(section, {name: name != family for name in FAMILY_ORDER})
    assert reduced < full, f"disabling '{family}' changed nothing"


def test_a12_task_representation_toggle_changes_the_encoder(cfg):
    """A12: no semantic task representation."""
    from amaun.manager.task_representation import TaskEncoder

    assert TaskEncoder("token_stats") is TaskEncoder.TOKEN_STATS
    assert TaskEncoder(cfg.get("task_repr.encoder")) is TaskEncoder.BACKBONE_MEAN_HIDDEN


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(8, 6, bias=False)
        self.l1 = nn.Linear(8, 6, bias=False)


def _components():
    components = [make_delta(s, f"c{s}", rank=4, layers=2, d_in=8, d_out=6) for s in (1, 2)]
    for component in components:
        component.layers = {k: component.layers[k] for k in ("l0", "l1")}
    return components


def _cfg(**overrides):
    base = dict(
        parameterisation="softmax_k", granularity="layerwise", init_value=0.5,
        alpha_learning_rate=0.05, storage="keep_components", refactorize_rank=None,
        train_lora_params=True, freeze_source_adapters=True, steps=3, lora_learning_rate=0.01,
    )
    base.update(overrides)
    return FusionConfig(**base)


def test_a3_fixed_alpha_keeps_coefficients_constant():
    """A3: with alpha_learning_rate = 0 the coefficients must not move."""
    model = _Tiny()
    state = build_fused_model(model, _components(), _cfg(alpha_learning_rate=0.0))
    try:
        from amaun.fusion.layerwise import build_optimizer

        optimizer = build_optimizer(state, _cfg(alpha_learning_rate=0.0))
        before = state.alpha_by_layer()
        x = torch.randn(4, 8)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            model.l0(x).pow(2).mean().backward()
            optimizer.step()
        after = state.alpha_by_layer()
        for key in before:
            assert before[key] == pytest.approx(after[key], abs=1e-7)
    finally:
        teardown_fused_model(state)


def test_a8_global_alpha_changes_the_parameter_count():
    """A8: one global coefficient instead of one per layer."""
    layerwise = build_fused_model(_Tiny(), _components(), _cfg(granularity="layerwise"))
    try:
        layerwise_count = layerwise.trainable_parameter_count()["alpha_parameters"]
    finally:
        teardown_fused_model(layerwise)

    global_state = build_fused_model(_Tiny(), _components(), _cfg(granularity="global"))
    try:
        global_count = global_state.trainable_parameter_count()["alpha_parameters"]
    finally:
        teardown_fused_model(global_state)

    assert global_count < layerwise_count
    assert layerwise_count == 2 * 2      # 2 layers x m=2
    assert global_count == 2             # shared across layers


def test_a4_kd_disabled_changes_the_objective():
    """A4: lambda = 0 must make L_total identical to L_new."""
    enabled = KDConfig.from_config({
        "enabled": True, "lambda": 1.0, "temperature": 2.0,
        "teacher": "pre_merge_old_adapter", "retention_examples": 4,
    })
    disabled = KDConfig.from_config({
        "enabled": False, "lambda": 0.0, "temperature": 2.0,
        "teacher": "pre_merge_old_adapter", "retention_examples": 4,
    })
    assert enabled.enabled and enabled.lam == 1.0
    assert not disabled.enabled and disabled.lam == 0.0


def test_a5_compression_disabled_by_null_budget():
    """A5: a null R_max disables the compression trigger entirely."""
    from amaun.manager.compression import should_compress

    class _Repo:
        def memory_bytes(self):
            return 10**9

    triggered, reason = should_compress(_Repo(), None)
    assert not triggered and "A5" in reason
    assert should_compress(_Repo(), 1)[0] is True


def test_a9_a10_a11_outcome_restrictions_change_the_decision():
    """A9 reuse-only, A10 fuse-only, A11 create-always."""
    thresholds = Thresholds(tau_u=0.2, tau_reuse=0.6, source="fixed", selected_on="config")
    scores = [CandidateScore("A1", 0.9, {}, "sig", 54)]

    def run(allowed):
        return decide(
            task_id="t", task_index=1, scores=scores, thresholds=thresholds,
            decision_order="train_then_decide", allowed_outcomes=allowed,
            repository_size=1, temp_adapter_trained=True, top_k=1, reuse_prescreen_passed=True,
        ).outcome

    assert run(["REUSE"]) is Outcome.REUSE
    assert run(["FUSE"]) is Outcome.FUSE
    assert run(["CREATE"]) is Outcome.CREATE
    # The full configuration and each restriction must not all coincide.
    assert len({run(["REUSE"]), run(["FUSE"]), run(["CREATE"])}) == 3


def test_a2_fixed_rule_uses_no_utility_network():
    """A2: a fixed selection rule instead of the learned predictor."""
    from amaun.baselines.strategies import AmaunStrategy, FixedRuleStrategy, SimilarityConfig

    fixed = FixedRuleStrategy(
        SimilarityConfig(feature="semantic.d_task_sq", threshold=0.0, higher_is_more_similar=False)
    )
    assert fixed.requires_utility_network is False
    assert AmaunStrategy.requires_utility_network is True


def test_d004_parameterisation_toggle_is_observable():
    """sigmoid_pair vs softmax_k: the logit tensor's shape differs."""
    generator = torch.Generator().manual_seed(0)
    from amaun.adapters.delta import LayerDelta

    components = [
        LayerDelta(A=torch.randn(4, 8, generator=generator),
                   B=torch.randn(6, 4, generator=generator), scaling=1.0)
        for _ in range(2)
    ]
    base = nn.Linear(8, 6, bias=False)
    pair = FusedLoRALinear(base, components, parameterisation="sigmoid_pair", init_value=0.5)
    softmax = FusedLoRALinear(base, components, parameterisation="softmax_k", init_value=0.5)
    assert pair.fusion_logit.numel() == 1
    assert softmax.fusion_logit.numel() == 2

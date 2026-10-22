"""The fusion memory fix is value-preserving (docs/l4_decision_study.md S6.2).

Two changes made a batch-4 fusion + KD step fit on a 22 GB L4:

1. ``FusedLoRALinear`` scales the rank-r intermediate by ``alpha_k * s_k`` instead of the
   d_out-wide term, so backward no longer saves an fp32 ``[.., d_out]`` tensor per component.
2. ``backward_fusion_losses`` runs one backward pass per loss term, so the task graph is freed
   before the KD retention forward.

Each test compares against the previous formulation, written out here as the reference.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from amaun.adapters.delta import LayerDelta
from amaun.data.preprocess import IGNORE_INDEX, TokenizedBatch, masked_cross_entropy
from amaun.fusion.fused_module import FusedLoRALinear
from amaun.fusion.optimize import backward_fusion_losses
from amaun.training.kd import KDConfig, kd_loss

D_IN, D_OUT, RANK = 16, 256, 4


def _components(n: int = 2, d_in: int = D_IN, d_out: int = D_OUT):
    generator = torch.Generator().manual_seed(0)
    return [
        LayerDelta(A=torch.randn(RANK, d_in, generator=generator) * 0.3,
                   B=torch.randn(d_out, RANK, generator=generator) * 0.3, scaling=1.0 + index)
        for index in range(n)
    ]


def _previous_forward(module: FusedLoRALinear, x: torch.Tensor) -> torch.Tensor:
    """The formulation before the fix: scale each d_out-wide term after applying B."""
    out = module.base(x)
    coefficients = module.alpha()
    x_work = x.to(module.component_A(0).dtype)
    delta = None
    for index in range(module.num_components):
        inner = torch.nn.functional.linear(x_work, module.component_A(index))
        term = torch.nn.functional.linear(inner, module.component_B(index))
        term = term * (coefficients[index] * module.scalings[index])
        delta = term if delta is None else delta + term
    return out + delta.to(out.dtype)


def _fused(seed: int = 0, n: int = 2) -> FusedLoRALinear:
    torch.manual_seed(seed)
    module = FusedLoRALinear(nn.Linear(D_IN, D_OUT, bias=False), _components(n),
                             trainable_component_indices=(n - 1,), init_value=1.0 / n)
    with torch.no_grad():
        module.fusion_logit.copy_(torch.randn(n))      # away from init, so alpha's gradient is generic
    return module


def _grads(module: FusedLoRALinear, x: torch.Tensor, forward) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    module.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    out = forward(module, x)
    out.pow(2).mean().backward()
    grads = {name: p.grad.clone() for name, p in module.named_parameters() if p.grad is not None}
    grads["x"] = x.grad.clone()
    return out.detach(), grads


def test_scaled_inner_forward_and_gradients_match_the_previous_formulation():
    for n in (2, 3):
        module = _fused(n=n)
        x = torch.randn(3, 5, D_IN)
        new_out, new_grads = _grads(module, x, lambda m, v: m(v))
        old_out, old_grads = _grads(module, x, _previous_forward)
        assert torch.allclose(new_out, old_out, atol=1e-6, rtol=1e-5)
        assert set(new_grads) == set(old_grads) >= {"fusion_logit", f"lora_A_{n - 1}", f"lora_B_{n - 1}", "x"}
        for name in old_grads:
            assert torch.allclose(new_grads[name], old_grads[name], atol=1e-6, rtol=1e-5), name


def _saved_shapes(module: FusedLoRALinear, x: torch.Tensor, forward) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []

    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    x = x.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        forward(module, x).sum().backward()
    return shapes


def test_no_d_out_wide_activation_is_saved_for_backward():
    module = _fused()
    x = torch.randn(3, 5, D_IN)
    wide = (3, 5, D_OUT)
    # The previous formulation saved one per component; this proves the check can fail.
    assert _saved_shapes(module, x, _previous_forward).count(wide) == module.num_components
    assert wide not in _saved_shapes(module, x, lambda m, v: m(v))


class _TinyLM(nn.Module):
    """Embedding -> FusedLoRALinear -> vocab head, with the ``.logits`` interface of a causal LM."""

    def __init__(self, vocab: int = 32):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, D_IN)
        self.hidden = FusedLoRALinear(nn.Linear(D_IN, D_IN, bias=False), _components(2, D_IN, D_IN),
                                      trainable_component_indices=(1,), init_value=0.5)
        self.head = nn.Linear(D_IN, vocab, bias=False)
        for param in (*self.embed.parameters(), *self.head.parameters()):
            param.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.head(torch.tanh(self.hidden(self.embed(input_ids)))))


def _batch(seed: int, batch: int = 2, seq: int = 7, vocab: int = 32) -> TokenizedBatch:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, seq), generator=generator)
    labels = ids.clone()
    labels[:, :3] = IGNORE_INDEX              # prompt positions
    return TokenizedBatch(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                          prompt_lengths=[3] * batch)


def test_separate_backward_passes_give_the_joint_gradient():
    model = _TinyLM()
    kd_cfg = KDConfig(enabled=True, lam=0.7, temperature=2.0, teacher="pre_merge_old_adapter",
                      retention_examples=2, top_p_truncation=None)
    task, retention = _batch(1), _batch(2)
    teacher = torch.randn(2, 7, 32, generator=torch.Generator().manual_seed(3))
    trainable = [p for p in model.parameters() if p.requires_grad]

    # Reference: the previous loop -- both forwards, one backward of the summed loss.
    model.zero_grad(set_to_none=True)
    joint_new = masked_cross_entropy(model(task.input_ids, task.attention_mask).logits, task.labels)
    joint_kd = kd_loss(model(retention.input_ids, retention.attention_mask).logits, teacher,
                       retention.labels, temperature=kd_cfg.temperature)
    (joint_new + kd_cfg.lam * joint_kd).backward()
    reference = [p.grad.clone() for p in trainable]

    model.zero_grad(set_to_none=True)
    loss_new, loss_kd = backward_fusion_losses(model, task, retention, teacher, kd_cfg=kd_cfg)
    assert float(loss_kd) > 0
    assert torch.allclose(loss_new, joint_new.detach()) and torch.allclose(loss_kd, joint_kd.detach())
    for param, expected in zip(trainable, reference):
        assert torch.allclose(param.grad, expected, atol=1e-6, rtol=1e-5)

    # Without retention data, only the task loss contributes.
    model.zero_grad(set_to_none=True)
    task_only = masked_cross_entropy(model(task.input_ids, task.attention_mask).logits, task.labels)
    task_only.backward()
    expected = [p.grad.clone() for p in trainable]
    model.zero_grad(set_to_none=True)
    _, zero = backward_fusion_losses(model, task, None, None, kd_cfg=kd_cfg)
    assert float(zero) == 0.0
    for param, grad in zip(trainable, expected):
        assert torch.allclose(param.grad, grad, atol=1e-7)

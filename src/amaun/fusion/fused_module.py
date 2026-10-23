"""M8 -- ``FusedLoRALinear``: fusion at the effective-update level.

Implements idea.md S6.1 generalised to ``m`` components (D-004):

    Delta_W_merge_l = sum_k alpha_{l,k} * Delta_W_{l,k},    sum_k alpha_{l,k} = 1

and applies it to activations **without materialising the dense delta**:

    x @ (sum_k a_k s_k B_k A_k)^T  ==  sum_k a_k s_k ((x @ A_k^T) @ B_k^T)

That identity is exact, not an approximation, and it is what makes the method runnable at
7B: a dense ``Delta_W`` for one 3584x18944 MLP projection is ~68M floats, and there are
seven such modules per layer across 28 layers.  ``tests/test_fusion_math.py`` verifies the
module's output against an explicitly materialised dense reference on a small model.

Two parameterisations are supported, and they agree exactly at ``m = 2``:

- ``softmax_k``: ``alpha_l = softmax(a_l)``, logits initialised to 0 so ``alpha = 1/m``.
- ``sigmoid_pair``: ``alpha = [sigmoid(a), 1 - sigmoid(a)]``, ``a = 0`` so ``alpha = 0.5``
  -- literally idea.md S6.2.

Both satisfy idea.md S6.1's mandated initialisation ``alpha_l = 0.5`` when ``m = 2``
(invariant I8.1) and keep ``alpha`` bounded in ``(0,1)`` and normalised at every step (I8.2).

A third, ``free``, is the l4_full_study deviation (docs/l4_next_run_prompt.md item 3): one
independent, unnormalised coefficient per component, initialised to ``init_value`` (1.0), so a
fused pair can keep both adapters at full strength.  Normalisation forces each component to about
half strength, which run 1 identified as one reason fusion lost to either source adapter.

One numerical caveat, verified by ``tests/test_fusion_math.py``: at extreme logits (|a| beyond
roughly 40 in float32) softmax **saturates** to exactly 1.0 and 0.0, so the realised bound is
``[0,1]`` rather than the open interval.  Normalisation -- the property fusion actually depends
on -- holds exactly regardless.  A saturated coefficient means the optimiser has driven one
component to dominate that layer completely, which is a legitimate outcome worth reading in the
alpha log, not a bug.
"""

from __future__ import annotations

import enum
from typing import Sequence

import torch
import torch.nn as nn

from amaun.adapters.delta import LayerDelta
from amaun.errors import FusionError


class AlphaParameterisation(str, enum.Enum):
    SOFTMAX_K = "softmax_k"
    SIGMOID_PAIR = "sigmoid_pair"
    FREE = "free"

    @property
    def normalised(self) -> bool:
        return self is not AlphaParameterisation.FREE


class FusedLoRALinear(nn.Module):
    """A frozen base ``Linear`` plus ``m`` fused LoRA components with learnable coefficients.

    The coefficient parameter is named ``fusion_logit`` so
    :func:`amaun.models.backbone.adapter_parameter_names` recognises it as a legitimately
    trainable parameter and the frozen-backbone assertion does not flag it.
    """

    def __init__(
        self,
        base: nn.Module,
        components: Sequence[LayerDelta],
        *,
        parameterisation: AlphaParameterisation = AlphaParameterisation.SOFTMAX_K,
        trainable_component_indices: Sequence[int] = (),
        shared_logit: nn.Parameter | None = None,
        init_value: float = 0.5,
    ) -> None:
        super().__init__()
        if not components:
            raise FusionError("FusedLoRALinear needs at least one component")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)   # the base layer stays frozen, always

        self.parameterisation = AlphaParameterisation(parameterisation)
        self.num_components = len(components)
        if self.parameterisation is AlphaParameterisation.SIGMOID_PAIR and self.num_components != 2:
            raise FusionError(
                f"sigmoid_pair is defined for exactly 2 components (idea.md S6.2), got "
                f"{self.num_components}; use softmax_k for top-k fusion"
            )
        if self.parameterisation.normalised and abs(init_value - 1.0 / self.num_components) > 1e-9:
            raise FusionError(
                f"init_value={init_value} is inconsistent with equal initial contribution for "
                f"m={self.num_components} components (expected {1.0/self.num_components}). "
                f"idea.md S6.1 fixes equal-contribution initialisation; it is not tunable."
            )

        device = next(base.parameters()).device
        dtype = torch.float32   # coefficients and factors in fp32 for stable alpha gradients

        self.trainable_indices = tuple(sorted(set(trainable_component_indices)))
        self.scalings: list[float] = []
        for index, component in enumerate(components):
            a = component.A.detach().clone().to(device=device, dtype=dtype)
            b = component.B.detach().clone().to(device=device, dtype=dtype)
            self.scalings.append(component.scaling)
            if index in self.trainable_indices:
                # A *clone* is trainable, so the source adapter in R_c is never mutated
                # (invariant I8.4).
                self.register_parameter(f"lora_A_{index}", nn.Parameter(a))
                self.register_parameter(f"lora_B_{index}", nn.Parameter(b))
            else:
                self.register_buffer(f"lora_A_{index}", a, persistent=False)
                self.register_buffer(f"lora_B_{index}", b, persistent=False)

        if shared_logit is not None:
            # Global-alpha variant (ablation A8): one coefficient vector shared by all layers.
            self.fusion_logit = shared_logit
            self._owns_logit = False
        else:
            size = 1 if self.parameterisation is AlphaParameterisation.SIGMOID_PAIR else self.num_components
            # Normalised forms start from zero logits (alpha = 1/m); ``free`` holds the
            # coefficients themselves, starting at ``init_value``.
            start = 0.0 if self.parameterisation.normalised else float(init_value)
            self.fusion_logit = nn.Parameter(torch.full((size,), start, device=device, dtype=dtype))
            self._owns_logit = True

    # -- coefficients ------------------------------------------------------------
    def alpha(self) -> torch.Tensor:
        """Current coefficients, shape ``[m]``: normalised for softmax / sigmoid, raw for ``free``."""
        if self.parameterisation is AlphaParameterisation.FREE:
            return self.fusion_logit
        if self.parameterisation is AlphaParameterisation.SIGMOID_PAIR:
            first = torch.sigmoid(self.fusion_logit[0])
            return torch.stack([first, 1.0 - first])
        return torch.softmax(self.fusion_logit, dim=0)

    def alpha_values(self) -> list[float]:
        with torch.no_grad():
            return [float(v) for v in self.alpha()]

    def component_A(self, index: int) -> torch.Tensor:
        return getattr(self, f"lora_A_{index}")

    def component_B(self, index: int) -> torch.Tensor:
        return getattr(self, f"lora_B_{index}")

    # -- forward -----------------------------------------------------------------
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = self.base(x, *args, **kwargs)
        coefficients = self.alpha()
        work_dtype = self.component_A(0).dtype
        x_work = x.to(work_dtype)
        delta = None
        for index in range(self.num_components):
            inner = torch.nn.functional.linear(x_work, self.component_A(index))
            # The coefficient scales the rank-r intermediate, not the d_out-wide term: by
            # linearity (c * h) @ B^T == c * (h @ B^T), but the multiply's backward then saves
            # an [.., r] tensor instead of an [.., d_out] fp32 one per component.  At batch 4 x
            # 640 tokens that difference is what made a fusion step exceed 22 GB on an L4.
            scaled = inner * (coefficients[index] * self.scalings[index])
            term = torch.nn.functional.linear(scaled, self.component_B(index))
            delta = term if delta is None else delta + term
        return out + delta.to(out.dtype)

    # -- reference implementation (tests only) -----------------------------------
    def dense_delta(self) -> torch.Tensor:
        """Materialise ``sum_k alpha_k Delta_W_k``.

        Used only by tests to check the factored forward against the dense definition
        (invariant I8.3), and by ``fusion.storage: dense_delta``.  Never called in a 7B run.
        """
        coefficients = self.alpha()
        total = None
        for index in range(self.num_components):
            block = (
                coefficients[index]
                * self.scalings[index]
                * (self.component_B(index) @ self.component_A(index))
            )
            total = block if total is None else total + block
        return total

    def extra_repr(self) -> str:
        return (
            f"m={self.num_components}, parameterisation={self.parameterisation.value}, "
            f"trainable_components={self.trainable_indices}, alpha={self.alpha_values()}"
        )

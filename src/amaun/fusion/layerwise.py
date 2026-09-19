"""M8 -- building, optimising, and tearing down a layer-wise fused adapter.

Implements Algorithm 2 (idea.md S6.7):

    1. compute Delta_W_old_l and Delta_W_temp_l for every layer
    2. initialise alpha_l = 0.5
    3. Delta_W_merge_l = alpha_l Delta_W_old_l + (1-alpha_l) Delta_W_temp_l
    4. build the merged model
    5. run A_old on D_rep, store teacher distributions q_old      <- BEFORE any update
    6-11. optimise L_total = L_new + lambda L_KD, updating LoRA params AND every alpha
    12. return the optimised merged adapter and [alpha_1 .. alpha_L]

The teacher-capture ordering of step 5 is enforced by a flag, not by comment: attempting to
step the optimiser before the teacher is captured raises (invariant I8.6).  A teacher
captured *after* fusion begins would be distilling the student against itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from amaun.adapters.delta import AdapterDelta
from amaun.errors import FusionError, FusionSourceMutatedError
from amaun.fusion.fused_module import AlphaParameterisation, FusedLoRALinear
from amaun.fusion.refactorize import FusionStorage, RefactorizationReport, fuse_to_delta
from amaun.utils.logging import get_logger

log = get_logger("fusion.layerwise")


@dataclass
class FusionConfig:
    parameterisation: str
    granularity: str            # "layerwise" | "global"
    init_value: float
    alpha_learning_rate: float
    storage: str
    refactorize_rank: int | None
    train_lora_params: bool
    freeze_source_adapters: bool
    steps: int
    lora_learning_rate: float

    @classmethod
    def from_config(cls, section: dict[str, Any], *, lora_learning_rate: float) -> "FusionConfig":
        coefficient = section.get("coefficient") or {}
        required = ("parameterisation", "granularity", "init_value", "learning_rate")
        missing = [key for key in required if key not in coefficient]
        if missing:
            raise FusionError(f"fusion.coefficient missing keys: {missing}")
        return cls(
            parameterisation=str(coefficient["parameterisation"]),
            granularity=str(coefficient["granularity"]),
            init_value=float(coefficient["init_value"]),
            alpha_learning_rate=float(coefficient["learning_rate"]),
            storage=str(section["storage"]),
            refactorize_rank=section.get("refactorize_rank"),
            train_lora_params=bool(section.get("train_lora_params", True)),
            freeze_source_adapters=bool(section.get("freeze_source_adapters", True)),
            steps=int(section["steps"]),
            lora_learning_rate=lora_learning_rate,
        )


@dataclass
class FusionState:
    """A live fused model plus the bookkeeping needed to audit it."""

    model: Any
    modules: dict[str, FusedLoRALinear]
    component_ids: list[str]
    global_logit: nn.Parameter | None
    source_hashes: dict[str, str]
    replaced: dict[str, nn.Module] = field(default_factory=dict)
    teacher_captured: bool = False

    # -- coefficients ------------------------------------------------------------
    def alpha_by_layer(self) -> dict[str, list[float]]:
        return {key: module.alpha_values() for key, module in self.modules.items()}

    def alpha_parameters(self) -> list[nn.Parameter]:
        if self.global_logit is not None:
            return [self.global_logit]
        # dict de-duplicates shared parameters while preserving order
        unique = {id(m.fusion_logit): m.fusion_logit for m in self.modules.values()}
        return list(unique.values())

    def lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self.modules.values():
            for index in module.trainable_indices:
                params.append(getattr(module, f"lora_A_{index}"))
                params.append(getattr(module, f"lora_B_{index}"))
        return params

    def trainable_parameter_count(self) -> dict[str, int]:
        """Reported so fusion's parameter cost is disclosed (invariant I8.7)."""
        alpha_params = sum(p.numel() for p in self.alpha_parameters())
        lora_params = sum(p.numel() for p in self.lora_parameters())
        return {
            "alpha_parameters": alpha_params,
            "lora_parameters": lora_params,
            "total": alpha_params + lora_params,
            "num_fused_layers": len(self.modules),
            "num_components": len(self.component_ids),
        }


def _get_module(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def build_fused_model(
    base_model: nn.Module,
    components: Sequence[AdapterDelta],
    cfg: FusionConfig,
) -> FusionState:
    """Replace each adapted base ``Linear`` with a :class:`FusedLoRALinear`.

    ``components[0]`` is by convention ``A_old`` (the selected repository adapter) and the
    last is ``A_temp``, matching idea.md S6.1's ``alpha*old + (1-alpha)*temp``.
    """
    if len(components) < 2:
        raise FusionError("fusion needs at least two components (an existing adapter and a candidate)")

    shared_keys = set(components[0].layers)
    for other in components[1:]:
        shared_keys &= set(other.layers)
    if not shared_keys:
        raise FusionError("components share no layer keys; they cannot be fused")

    parameterisation = AlphaParameterisation(cfg.parameterisation)

    # idea.md S6.1/S10 fix the initialisation at equal contribution and declare it NOT tunable.
    # For the pairwise case that means alpha = 0.5; generalised to m components (D-004) it means
    # alpha = 1/m, which the softmax gives for all-zero logits.  The configured value is checked
    # against the pairwise case so a config that contradicts idea.md is rejected rather than
    # silently reinterpreted.
    equal_contribution = 1.0 / len(components)
    if len(components) == 2 and abs(cfg.init_value - 0.5) > 1e-9:
        raise FusionError(
            f"fusion.coefficient.init_value must be 0.5 for pairwise fusion (idea.md S6.1 fixes "
            f"equal-contribution initialisation and states it is not tunable); got {cfg.init_value}"
        )

    global_logit: nn.Parameter | None = None
    if cfg.granularity == "global":
        # Ablation A8: a single coefficient vector for the whole network.
        device = next(base_model.parameters()).device
        size = 1 if parameterisation is AlphaParameterisation.SIGMOID_PAIR else len(components)
        global_logit = nn.Parameter(torch.zeros(size, device=device, dtype=torch.float32))
    elif cfg.granularity != "layerwise":
        raise FusionError(f"unknown fusion granularity '{cfg.granularity}'")

    # Which components are trainable.  Source repository adapters are frozen (I8.4); only a
    # *clone* of the candidate is optimised, so R_c is provably untouched.
    trainable = (len(components) - 1,) if cfg.train_lora_params else ()

    modules: dict[str, FusedLoRALinear] = {}
    replaced: dict[str, nn.Module] = {}
    for key in sorted(shared_keys):
        parent, attribute = _get_module(base_model, key)
        current = getattr(parent, attribute)
        # ``key`` is a *base* module path, but on a PEFT-wrapped backbone that path now holds
        # PEFT's LoRA wrapper.  The fused module must wrap the wrapper's ``base_layer`` -- the
        # true frozen Linear -- for two reasons:
        #   1. wrapping the LoRA wrapper would add the currently-installed adapter's delta on
        #      top of the fused components, double-counting it;
        #   2. FusedLoRALinear freezes its base module's parameters, which on the LoRA wrapper
        #      would permanently disable the LoRA factors' gradients.
        # The LoRA wrapper itself is what gets restored on teardown.
        base_layer = getattr(current, "base_layer", current)
        fused = FusedLoRALinear(
            base_layer,
            [component.layers[key] for component in components],
            parameterisation=parameterisation,
            trainable_component_indices=trainable,
            shared_logit=global_logit,
            init_value=equal_contribution,
        )
        setattr(parent, attribute, fused)
        modules[key] = fused
        replaced[key] = current

    state = FusionState(
        model=base_model,
        modules=modules,
        component_ids=[c.adapter_id for c in components],
        global_logit=global_logit,
        source_hashes={c.adapter_id: c.content_hash() for c in components},
        replaced=replaced,
    )
    counts = state.trainable_parameter_count()
    log.info(
        "fused %d layers over %d components (%s, %s): %d alpha + %d lora trainable params",
        counts["num_fused_layers"], counts["num_components"], cfg.parameterisation,
        cfg.granularity, counts["alpha_parameters"], counts["lora_parameters"],
    )
    return state


def teardown_fused_model(state: FusionState) -> None:
    """Restore the original base ``Linear`` modules, leaving the backbone exactly as found."""
    for key, original in state.replaced.items():
        parent, attribute = _get_module(state.model, key)
        setattr(parent, attribute, original)
    state.replaced.clear()


def assert_sources_unmutated(state: FusionState, components: Sequence[AdapterDelta]) -> None:
    """Invariant I8.4: source adapters must be byte-identical after fusion."""
    for component in components:
        expected = state.source_hashes.get(component.adapter_id)
        if expected is None:
            continue
        actual = component.content_hash()
        if actual != expected:
            raise FusionSourceMutatedError(
                f"source adapter '{component.adapter_id}' changed during fusion "
                f"({expected[:12]} -> {actual[:12]}). Repository adapters must stay frozen "
                f"during fusion experiments (docs/system_specification.md I8.4)."
            )


def build_optimizer(state: FusionState, cfg: FusionConfig) -> torch.optim.Optimizer:
    """Joint optimiser with separate LR groups for ``phi_merge`` and ``a`` (idea.md S6.5)."""
    groups = []
    lora_params = state.lora_parameters()
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.lora_learning_rate, "name": "lora"})
    alpha_params = state.alpha_parameters()
    if not alpha_params:
        raise FusionError("no fusion coefficients to optimise")
    groups.append({"params": alpha_params, "lr": cfg.alpha_learning_rate, "name": "alpha"})
    return torch.optim.AdamW(groups)


def collapse(
    state: FusionState,
    components: Sequence[AdapterDelta],
    cfg: FusionConfig,
    *,
    adapter_id: str,
    default_rank: int | None = None,
) -> tuple[AdapterDelta, RefactorizationReport]:
    """Produce the storable fused adapter from the current live coefficients and factors.

    The *trained* factors are read back out of the fused modules, so post-fusion
    optimisation of ``phi_merge`` is reflected in what gets stored -- reading the original
    component tensors instead would silently discard the optimisation.
    """
    from amaun.adapters.delta import LayerDelta

    live_components: list[AdapterDelta] = []
    for index in range(len(components)):
        layers = {}
        for key, module in state.modules.items():
            layers[key] = LayerDelta(
                A=module.component_A(index).detach().cpu().clone(),
                B=module.component_B(index).detach().cpu().clone(),
                scaling=module.scalings[index],
            )
        live_components.append(AdapterDelta(layers=layers, adapter_id=components[index].adapter_id))

    alphas = {
        key: module.alpha().detach().cpu() for key, module in state.modules.items()
    }
    rank = cfg.refactorize_rank or default_rank
    return fuse_to_delta(
        live_components, alphas,
        storage=FusionStorage(cfg.storage),
        target_rank=rank,
        adapter_id=adapter_id,
    )

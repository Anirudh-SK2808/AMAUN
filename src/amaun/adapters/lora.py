"""M3 -- LoRA injection, extraction, and switching.

Thin, explicit layer over PEFT.  Two responsibilities matter scientifically:

1. **Injection must not unfreeze the backbone.**  ``prepare_model_for_kbit_training`` and
   gradient checkpointing both re-enable ``requires_grad`` on parts of the base model; we
   re-assert the frozen invariant after every such call (invariant I1.3).
2. **Extraction must yield an** :class:`~amaun.adapters.delta.AdapterDelta` keyed by the
   *base* module path, so deltas from different adapters -- and from different processes --
   are comparable by key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

from amaun.adapters.delta import AdapterDelta, LayerDelta
from amaun.errors import AdapterError
from amaun.models.backbone import assert_backbone_frozen
from amaun.utils.logging import get_logger

log = get_logger("adapters.lora")

# Qwen2 attention + MLP projections.  Kept here as the *documented* default set; the
# experiment-affecting value lives in configs/model/*.yaml and is what actually gets used.
QWEN2_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class LoRASpec:
    """LoRA hyperparameters.  Every field comes from config, never from a code default."""

    r: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "LoRASpec":
        missing = [k for k in ("r", "alpha", "dropout", "target_modules") if k not in section]
        if missing:
            raise AdapterError(f"lora config missing required keys: {missing}")
        return cls(
            r=int(section["r"]),
            alpha=float(section["alpha"]),
            dropout=float(section["dropout"]),
            target_modules=tuple(section["target_modules"]),
            bias=str(section.get("bias", "none")),
            task_type=str(section.get("task_type", "CAUSAL_LM")),
        )

    @property
    def scaling(self) -> float:
        """PEFT's LoRA scaling ``alpha / r``."""
        return self.alpha / self.r

    def as_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
            "task_type": self.task_type,
            "scaling": self.scaling,
        }


def _normalise_key(module_name: str) -> str:
    """Strip PEFT wrapper prefixes so keys are stable base-module paths.

    PEFT wraps the model as ``base_model.model.<original path>``; we remove that so a delta
    extracted from a PEFT model has the same keys as one loaded from disk or built by hand.
    """
    for prefix in ("base_model.model.", "base_model."):
        if module_name.startswith(prefix):
            module_name = module_name[len(prefix) :]
    return module_name


def inject_lora(model: Any, spec: LoRASpec, *, adapter_name: str = "default") -> Any:
    """Attach a LoRA parameterisation to a frozen backbone.

    PEFT rewrites the base model's modules **in place**, so a backbone may carry only one
    injection at a time; a second ``get_peft_model`` (or an ``unload()``) would invalidate every
    handle the rest of the pipeline holds.  AMAUN therefore injects **once per backbone** (see
    :func:`ensure_lora`) and achieves per-adapter independence by re-initialising the LoRA
    parameters instead (see :func:`reset_lora_parameters`).  That is also what keeps a single
    7B backbone resident (brief S20, development rule 6).
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias=spec.bias,
        task_type=spec.task_type,
    )
    peft_model = get_peft_model(model, config, adapter_name=adapter_name)
    # PEFT is supposed to freeze the base model; verify rather than trust (I1.3).
    assert_backbone_frozen(peft_model)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    log.info(
        "LoRA injected (r=%d, alpha=%s, %d target module types): %d trainable / %d total (%.4f%%)",
        spec.r, spec.alpha, len(spec.target_modules), trainable, total, 100.0 * trainable / max(total, 1),
    )
    return peft_model


def iter_lora_modules(model: Any, adapter_name: str = "default") -> Iterable[tuple[str, Any]]:
    """Yield ``(normalised_base_key, lora_layer_module)`` for each adapted layer."""
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        if adapter_name not in lora_a or adapter_name not in lora_b:
            continue
        yield _normalise_key(name), module


def extract_delta(model: Any, *, adapter_name: str = "default", adapter_id: str = "") -> AdapterDelta:
    """Extract an :class:`AdapterDelta` from a live PEFT model."""
    layers: dict[str, LayerDelta] = {}
    for key, module in iter_lora_modules(model, adapter_name):
        a_weight = module.lora_A[adapter_name].weight
        b_weight = module.lora_B[adapter_name].weight
        scaling = float(module.scaling[adapter_name]) if hasattr(module, "scaling") else 1.0
        layers[key] = LayerDelta(
            A=a_weight.detach().to(torch.float32).cpu().clone(),
            B=b_weight.detach().to(torch.float32).cpu().clone(),
            scaling=scaling,
        )
    if not layers:
        raise AdapterError(
            f"no LoRA layers found for adapter '{adapter_name}'; was inject_lora called?"
        )
    return AdapterDelta(layers=layers, adapter_id=adapter_id)


def load_delta_into(model: Any, delta: AdapterDelta, *, adapter_name: str = "default") -> int:
    """Write a delta's factors back into a live PEFT model.  Returns layers written.

    Used for adapter *switching* (I3.5): the repository holds deltas, and a delta is
    installed into the single resident backbone on demand rather than instantiating a new
    model per adapter.
    """
    written = 0
    with torch.no_grad():
        for key, module in iter_lora_modules(model, adapter_name):
            if key not in delta:
                continue
            layer = delta[key]
            a_param = module.lora_A[adapter_name].weight
            b_param = module.lora_B[adapter_name].weight
            if a_param.shape != layer.A.shape or b_param.shape != layer.B.shape:
                raise AdapterError(
                    f"shape mismatch installing '{key}': model expects A{tuple(a_param.shape)} "
                    f"B{tuple(b_param.shape)}, delta has A{tuple(layer.A.shape)} B{tuple(layer.B.shape)}"
                )
            a_param.copy_(layer.A.to(a_param.dtype).to(a_param.device))
            b_param.copy_(layer.B.to(b_param.dtype).to(b_param.device))
            written += 1
    if written == 0:
        raise AdapterError("no layers were written; delta keys do not match the model")
    return written


def zero_delta_(model: Any, *, adapter_name: str = "default") -> None:
    """Zero the LoRA ``B`` factors, making the adapter a no-op.

    Zeroing ``B`` (not ``A``) reproduces LoRA's initialisation condition, where the adapted
    model is exactly the base model.  This is how we verify that adapter switching leaves no
    residue (I3.5) without tearing down and rebuilding the PEFT wrapper.
    """
    with torch.no_grad():
        for _key, module in iter_lora_modules(model, adapter_name):
            module.lora_B[adapter_name].weight.zero_()


def trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def trainable_parameter_count(model: Any) -> int:
    return sum(p.numel() for p in trainable_parameters(model))


# --------------------------------------------------------------------------------------
# Single-injection management
# --------------------------------------------------------------------------------------

# One injection per base model object.  Keyed by id() because PEFT mutates the model in place,
# so "the same model" is object identity, not equality.
_INJECTIONS: dict[int, tuple[Any, LoRASpec]] = {}


def ensure_lora(model: Any, spec: LoRASpec, *, adapter_name: str = "default") -> Any:
    """Return the PEFT wrapper for ``model``, injecting it the first time only.

    Re-injecting would rewrite modules that existing handles already point at, and PEFT's
    ``unload()`` would remove the LoRA layers from *every* holder of the shared backbone --
    the cause of a class of "delta keys do not match the model" failures.  One injection,
    reused, removes that whole failure mode.
    """
    key = id(model)
    if key in _INJECTIONS:
        wrapper, existing = _INJECTIONS[key]
        if existing != spec:
            raise AdapterError(
                f"backbone already carries a LoRA injection with spec {existing.as_dict()}; "
                f"refusing to re-inject with {spec.as_dict()}. Changing the LoRA spec mid-run "
                f"would make adapters mutually incompatible."
            )
        return wrapper
    wrapper = inject_lora(model, spec, adapter_name=adapter_name)
    _INJECTIONS[key] = (wrapper, spec)
    return wrapper


def forget_injection(model: Any) -> None:
    """Drop the cached injection record (used when a backbone is released)."""
    _INJECTIONS.pop(id(model), None)


def reset_lora_parameters(model: Any, *, seed: int, adapter_name: str = "default") -> int:
    """Re-initialise the LoRA factors to LoRA's initial condition, under ``seed``.

    LoRA's initialisation sets ``A`` to a Kaiming-uniform draw and ``B`` to zero, so the adapted
    model starts exactly equal to the base model.  Resetting to that state before each adapter's
    training -- with a seed derived from the adapter id -- gives the same independence guarantee
    a fresh injection would, without touching the shared module graph:

      * no parameter values survive from the previous adapter,
      * the RNG stream is a function of the adapter id alone, not of call order.

    Returns the number of layers reset.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    count = 0
    with torch.no_grad():
        for _key, module in iter_lora_modules(model, adapter_name):
            a_weight = module.lora_A[adapter_name].weight
            b_weight = module.lora_B[adapter_name].weight
            fan_in = a_weight.shape[1]
            bound = math.sqrt(6.0 / max(1, fan_in))   # Kaiming-uniform with a=sqrt(5)
            sample = torch.empty(a_weight.shape, dtype=torch.float32).uniform_(
                -bound, bound, generator=generator
            )
            a_weight.copy_(sample.to(a_weight.dtype).to(a_weight.device))
            b_weight.zero_()
            count += 1
    if count == 0:
        raise AdapterError("reset_lora_parameters found no LoRA layers")
    return count

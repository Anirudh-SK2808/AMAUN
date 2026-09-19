"""M0 -- configuration, run-mode resolution, and the single backbone registry.

This module is the *only* place in the codebase where a model checkpoint id or an
experiment-affecting numeric default may appear, and even here the numerics live in YAML
under ``configs/``.  See ``docs/system_specification.md`` M0.

Key contract (idea.md S0.1): ``run_mode`` is required and has **no default**.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from amaun.errors import (
    ConfigError,
    MissingRunModeError,
    QuantizationUnavailableError,
)

# --------------------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------------------


class RunMode(str, enum.Enum):
    """Backbone tier.  See docs/decision_register.md D-001.

    ``dummy`` is a *test fixture* tier, not a research tier: it builds a tiny randomly
    initialised Qwen2 locally so the full pipeline can be exercised offline in seconds.
    It is hard-blocked from producing any reportable metric.
    """

    DUMMY = "dummy"
    LOCAL_SMOKE = "local_smoke"
    ACTUAL = "actual"

    @property
    def is_reportable(self) -> bool:
        """Only ``actual`` runs may populate result tables (idea.md S0.1.1)."""
        return self is RunMode.ACTUAL


class Quantization(str, enum.Enum):
    NONE = "none"
    INT8 = "int8"
    NF4 = "nf4"

    @property
    def requires_cuda(self) -> bool:
        return self is not Quantization.NONE


@dataclass(frozen=True)
class BackboneSpec:
    """Everything backbone-specific, resolved from ``run_mode`` and nothing else."""

    model_id: str | None  # None => built locally (dummy tier)
    dtype: str
    max_seq_len: int
    attn_implementation: str
    quantization: Quantization
    device_preference: tuple[str, ...]
    is_reportable: bool
    notes: str = ""


# The one and only checkpoint table (idea.md S0.1 BACKBONE_REGISTRY).
BACKBONE_REGISTRY: dict[RunMode, BackboneSpec] = {
    RunMode.DUMMY: BackboneSpec(
        model_id=None,
        dtype="float32",
        max_seq_len=128,
        attn_implementation="eager",
        quantization=Quantization.NONE,
        device_preference=("cpu",),
        is_reportable=False,
        notes="Tiny randomly-initialised Qwen2 built in-process. Offline. Never reportable.",
    ),
    RunMode.LOCAL_SMOKE: BackboneSpec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        dtype="bfloat16",
        max_seq_len=1024,
        attn_implementation="sdpa",
        quantization=Quantization.NONE,
        device_preference=("mps", "cuda", "cpu"),
        is_reportable=False,
        notes="Pipeline validation only (idea.md S0.1). No quantisation: bitsandbytes has no MPS backend.",
    ),
    RunMode.ACTUAL: BackboneSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        max_seq_len=2048,
        attn_implementation="sdpa",
        quantization=Quantization.NF4,
        device_preference=("cuda",),
        is_reportable=True,
        notes="The only reportable tier. QLoRA requires CUDA (D-008).",
    ),
}


# --------------------------------------------------------------------------------------
# Config tree
# --------------------------------------------------------------------------------------


class _Missing:
    """Sentinel distinguishing "no default supplied" from ``default=None``."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<required>"


MISSING = _Missing()


@dataclass
class ResolvedConfig:
    """A fully resolved, validated, hashable configuration.

    ``raw`` keeps the merged YAML exactly as loaded so ``config_used.json`` can reproduce
    the run without reference to any code default.
    """

    run_mode: RunMode
    backbone: BackboneSpec
    experiment_id: str
    seed: int
    raw: dict[str, Any]
    device: str = "cpu"

    # ---- convenience accessors (never invent defaults for experiment-affecting keys) --
    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if value is None:
            raise ConfigError(f"missing required config section '{name}'")
        if not isinstance(value, dict):
            raise ConfigError(f"config section '{name}' must be a mapping, got {type(value)}")
        return value

    def get(self, dotted: str, default: Any = MISSING) -> Any:
        """Read ``a.b.c``.  Omitting ``default`` makes the key mandatory.

        Mandatory-by-default is deliberate: an experiment-affecting value must never be
        silently substituted by a code default (master_prompt constraint 14).
        """
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node or node[part] is None:
                if isinstance(default, _Missing):
                    raise ConfigError(
                        f"missing required config key '{dotted}'; AMAUN provides no default "
                        f"for experiment-affecting values"
                    )
                return default
            node = node[part]
        return node

    @property
    def config_hash(self) -> str:
        """Stable sha256 over the canonical form of the resolved configuration."""
        payload = {
            "run_mode": self.run_mode.value,
            "backbone": {**asdict(self.backbone), "quantization": self.backbone.quantization.value},
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "raw": self.raw,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_config_used(self) -> dict[str, Any]:
        """The payload written to ``config_used.json`` (idea.md S0.1.1)."""
        from amaun.utils.provenance import collect_provenance

        return {
            "schema_version": 1,
            "run_mode": self.run_mode.value,
            "is_reportable": self.run_mode.is_reportable,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "device": self.device,
            "config_hash": self.config_hash,
            "backbone": {
                **asdict(self.backbone),
                "quantization": self.backbone.quantization.value,
            },
            "config": self.raw,
            "provenance": collect_provenance(),
        }



# --------------------------------------------------------------------------------------
# Loading and resolution
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at top level")
    return data


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``includes:`` list relative to ``config_root``.

    ``includes`` are merged in order, then the including file's own keys win, then
    ``overrides`` win.  This keeps experiment files small without hiding values.
    """
    path = Path(path)
    root = Path(config_root) if config_root is not None else path.parent.parent
    data = load_yaml(path)
    merged: dict[str, Any] = {}
    for include in data.pop("includes", []) or []:
        merged = _deep_merge(merged, load_config(root / include, config_root=root))
    merged = _deep_merge(merged, data)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def _select_device(spec: BackboneSpec, requested: str | None) -> str:
    """Pick a device, honouring ``device_preference`` and availability."""
    import torch

    def available(name: str) -> bool:
        if name == "cpu":
            return True
        if name == "cuda":
            return torch.cuda.is_available()
        if name == "mps":
            return torch.backends.mps.is_available()
        return False

    if requested is not None and requested != "auto":
        if not available(requested):
            raise ConfigError(f"requested device '{requested}' is not available")
        return requested
    for candidate in spec.device_preference:
        if available(candidate):
            return candidate
    return "cpu"


def resolve(
    raw: Mapping[str, Any],
    *,
    run_mode: str | RunMode | None = None,
    experiment_id: str | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> ResolvedConfig:
    """Validate a raw config tree and resolve it against the backbone registry.

    ``run_mode`` may come from the config or from the caller (CLI), but must come from
    somewhere: there is no default (idea.md S0.1).
    """
    raw = dict(raw)

    mode_value = run_mode if run_mode is not None else raw.get("run_mode")
    if mode_value is None:
        raise MissingRunModeError(
            "run_mode is required and has no default; set it in the config or pass --run-mode. "
            "Allowed: " + ", ".join(m.value for m in RunMode)
        )
    mode = RunMode(mode_value) if not isinstance(mode_value, RunMode) else mode_value
    raw["run_mode"] = mode.value

    spec = BACKBONE_REGISTRY[mode]

    # Per-run-mode overlay: allows local_smoke to use smaller volumes without duplicating
    # the whole tree, while keeping every value visible in config_used.json.
    overlays = raw.get("run_mode_overrides") or {}
    if mode.value in overlays:
        raw = _deep_merge(raw, overlays[mode.value])

    exp_id = experiment_id or raw.get("experiment_id")
    if not exp_id:
        raise ConfigError("experiment_id is required (it names the results directory)")

    seeds = raw.get("experiment", {}).get("seeds")
    if seed is None:
        if not seeds:
            raise ConfigError(
                "experiment.seeds must list at least one seed (D-012); no default is provided"
            )
        seed = int(seeds[0])

    # Quantisation feasibility -- never silently downgrade (D-008).
    quant_override = raw.get("model", {}).get("quantization")
    quantization = Quantization(quant_override) if quant_override else spec.quantization
    resolved_device = _select_device(spec, device or raw.get("model", {}).get("device"))
    if quantization.requires_cuda and resolved_device != "cuda":
        raise QuantizationUnavailableError(
            f"quantization={quantization.value} requires CUDA but the resolved device is "
            f"'{resolved_device}'. AMAUN refuses to silently downgrade precision because that "
            f"would misreport the training recipe (docs/decision_register.md D-008). "
            f"Either run on a CUDA host or set model.quantization: none explicitly."
        )
    spec = BackboneSpec(**{**asdict(spec), "quantization": quantization})

    # Optional explicit model override, recorded loudly.
    model_override = raw.get("model", {}).get("model_id")
    if model_override:
        spec = BackboneSpec(**{**asdict(spec), "model_id": model_override,
                               "notes": spec.notes + f" [OVERRIDDEN to {model_override}]"})

    if raw.get("model", {}).get("max_seq_len"):
        spec = BackboneSpec(**{**asdict(spec), "max_seq_len": int(raw["model"]["max_seq_len"])})

    return ResolvedConfig(
        run_mode=mode,
        backbone=spec,
        experiment_id=str(exp_id),
        seed=int(seed),
        raw=raw,
        device=resolved_device,
    )

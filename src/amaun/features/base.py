"""M6 -- the feature-family contract.

idea.md S4 requires seven heterogeneous families, and S16.2 forbids any single metric from
being the decision rule.  Two structural choices enforce that:

1. A family returns a **named, ordered** dict of scalars.  The order is stable, and the
   normaliser stores the name order alongside mu/sigma, so a feature vector can never be
   silently permuted between fitting and inference.
2. Families are toggled by config, and disabling one **changes the vector's dimension**,
   which is recorded in the run log.  That is what makes ablations A1/A6/A7/A12 provably
   real rather than silent no-ops (invariant I6.4).
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class FeatureVector:
    """A named feature vector with per-family provenance."""

    names: list[str]
    values: list[float]
    family_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)   # counted, per invariant I6.6

    def __len__(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))

    def family(self, name: str) -> dict[str, float]:
        start, end = self.family_slices[name]
        return dict(zip(self.names[start:end], self.values[start:end]))

    def to_list(self) -> list[float]:
        return list(self.values)

    def signature(self) -> str:
        """Hash of the *name order*, so a normaliser can refuse a mismatched vector."""
        import hashlib

        return hashlib.sha256("|".join(self.names).encode()).hexdigest()[:16]


class FeatureFamily(abc.ABC):
    """One of the seven families of idea.md S4."""

    name: str = "unnamed"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    @abc.abstractmethod
    def compute(self, ctx: Any) -> dict[str, float | None]:
        """Return ``{feature_name: value}``; ``None`` marks a genuinely unknown value.

        Returning ``None`` rather than 0.0 matters: a brand-new adapter has *unknown*
        historical performance, and encoding that as zero would bias the utility network
        against new adapters.  The extractor substitutes the configured sentinel and counts
        the substitution so the bias is visible in the logs.
        """

    # -- helpers shared by families ---------------------------------------------
    @staticmethod
    def finite(value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    @staticmethod
    def aggregate_keys(mode: str, prefix: str, *, n_layers: int = 0) -> list[str]:
        """The key set a mode emits.  Fixed for a given mode, independent of the data.

        This is what keeps the feature vector's width constant across task events.  Emitting
        fewer keys when a value happens to be unavailable would change the vector's dimension
        mid-run and invalidate the fitted normaliser -- and, worse, would silently misalign
        feature indices if the normaliser were refitted.
        """
        if mode == "mean":
            return [f"{prefix}_mean"]
        if mode == "max":
            return [f"{prefix}_max"]
        if mode == "mean_std":
            return [f"{prefix}_mean", f"{prefix}_std"]
        if mode == "per_layer":
            return [f"{prefix}_l{i}" for i in range(n_layers)]
        raise ValueError(f"unknown aggregation mode '{mode}'")

    @staticmethod
    def aggregate(
        values: Iterable[float], mode: str, prefix: str, *, n_layers: int = 0
    ) -> dict[str, float | None]:
        """Aggregate per-layer values into named scalars (idea.md S4.2 'layer -> adapter').

        Always returns exactly the keys of :meth:`aggregate_keys`, using ``None`` for values
        that could not be computed, so the feature layout is stable.
        """
        keys = FeatureFamily.aggregate_keys(mode, prefix, n_layers=n_layers)
        collected = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if not collected:
            return {key: None for key in keys}
        count = len(collected)
        mean = sum(collected) / count
        if mode == "mean":
            return {keys[0]: mean}
        if mode == "max":
            return {keys[0]: max(collected)}
        if mode == "mean_std":
            variance = sum((v - mean) ** 2 for v in collected) / count
            return {keys[0]: mean, keys[1]: math.sqrt(variance)}
        # per_layer: pad or truncate to the declared width so the layout is fixed.
        padded: dict[str, float | None] = {}
        for index, key in enumerate(keys):
            padded[key] = collected[index] if index < len(collected) else None
        return padded

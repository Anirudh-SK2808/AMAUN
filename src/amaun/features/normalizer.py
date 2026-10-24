"""M6.8 -- feature normalisation (idea.md S4.7).  Leakage-critical.

    z'_{i,j} = (z_{i,j} - mu_j) / (sigma_j + eps)

``mu, sigma`` must come **only** from the utility-learning training split and be reused
unchanged for validation and test (idea.md S16.5).  They must also never cross run modes,
since a different backbone produces a different feature distribution (idea.md S4.7).

The normaliser therefore records *what it was fitted on* and refuses to transform under a
different run mode or to be fitted on a forbidden split.  Getting this wrong is invisible in
results -- the numbers still look reasonable -- which is exactly why it is enforced in code
rather than by discipline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from amaun.errors import NormalizerLeakageError
from amaun.features.base import FeatureVector

# Splits that may legitimately contribute to normalisation statistics.
_ALLOWED_FIT_SPLITS = {"utility_train", "train", "validation", "support"}
_FORBIDDEN_FIT_SPLITS = {"test"}


#: A feature whose spread on the fit split is at or below this is constant (D-029).
CONSTANT_STD = 1e-12


@dataclass
class FeatureNormalizer:
    """Standardises feature vectors, with provenance strong enough to audit."""

    feature_names: list[str] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    epsilon: float = 1e-8
    fitted_on_split: str | None = None
    fitted_run_mode: str | None = None
    fitted_n_samples: int = 0
    signature: str | None = None
    #: D-029: per feature, whether it varied on the fit split.  ``None`` = every feature is used
    #: (the behaviour before D-029, and of normalisers saved before it).  A feature that never
    #: varied carries no information, and dividing a later different value by its zero spread
    #: would blow it up to ~1e8 (e.g. repository size in a growing stream), so it is left out.
    active: list[bool] | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self.mean) and bool(self.std)

    def fit(
        self,
        vectors: Sequence[FeatureVector],
        *,
        split: str,
        run_mode: str,
        drop_constant: bool = False,
    ) -> "FeatureNormalizer":
        if split in _FORBIDDEN_FIT_SPLITS:
            raise NormalizerLeakageError(
                f"refusing to fit normalisation statistics on split '{split}'. "
                f"idea.md S16.5: never estimate normalisation statistics from the test sequence."
            )
        if split not in _ALLOWED_FIT_SPLITS:
            raise NormalizerLeakageError(
                f"unknown fit split '{split}'; allowed: {sorted(_ALLOWED_FIT_SPLITS)}"
            )
        if not vectors:
            raise NormalizerLeakageError("cannot fit a normaliser on zero feature vectors")

        names = vectors[0].names
        for vector in vectors:
            if vector.names != names:
                raise NormalizerLeakageError(
                    "feature vectors disagree on feature names/order; a normaliser fitted on "
                    "inconsistent vectors would silently permute features"
                )

        count = len(vectors)
        dimension = len(names)
        means = [0.0] * dimension
        for vector in vectors:
            for index, value in enumerate(vector.values):
                means[index] += value
        means = [total / count for total in means]

        variances = [0.0] * dimension
        for vector in vectors:
            for index, value in enumerate(vector.values):
                variances[index] += (value - means[index]) ** 2
        stds = [math.sqrt(total / count) for total in variances]

        self.feature_names = list(names)
        self.mean = means
        self.std = stds
        self.fitted_on_split = split
        self.fitted_run_mode = run_mode
        self.fitted_n_samples = count
        self.signature = vectors[0].signature()
        self.active = [sigma > CONSTANT_STD for sigma in stds] if drop_constant else None
        if self.active is not None and not any(self.active):
            raise NormalizerLeakageError("every feature is constant on the fit split; nothing to learn from")
        return self

    @property
    def n_active(self) -> int:
        return len(self.feature_names) if self.active is None else sum(self.active)

    @property
    def active_names(self) -> list[str]:
        if self.active is None:
            return list(self.feature_names)
        return [name for name, keep in zip(self.feature_names, self.active) if keep]

    def transform(self, vector: FeatureVector, *, run_mode: str) -> list[float]:
        if not self.is_fitted:
            raise NormalizerLeakageError("normaliser has not been fitted")
        if run_mode != self.fitted_run_mode:
            raise NormalizerLeakageError(
                f"normaliser was fitted under run_mode='{self.fitted_run_mode}' but is being "
                f"applied under '{run_mode}'. Feature distributions differ between backbones, "
                f"so statistics must never be shared across run modes (idea.md S4.7)."
            )
        if vector.names != self.feature_names:
            raise NormalizerLeakageError(
                f"feature name/order mismatch: normaliser expects {len(self.feature_names)} "
                f"features with signature {self.signature}, got {len(vector.names)} with "
                f"{vector.signature()}. This usually means a feature family was toggled "
                f"without refitting the normaliser."
            )
        scaled = [
            (value - mu) / (sigma + self.epsilon)
            for value, mu, sigma in zip(vector.values, self.mean, self.std)
        ]
        if self.active is None:
            return scaled
        return [value for value, keep in zip(scaled, self.active) if keep]

    # -- persistence -------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "mean": self.mean,
                    "std": self.std,
                    "epsilon": self.epsilon,
                    "fitted_on_split": self.fitted_on_split,
                    "fitted_run_mode": self.fitted_run_mode,
                    "fitted_n_samples": self.fitted_n_samples,
                    "active": self.active,
                    "signature": self.signature,
                },
                handle, indent=2,
            )
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: Path | str, *, expected_run_mode: str) -> "FeatureNormalizer":
        with Path(path).open() as handle:
            data = json.load(handle)
        if data.get("fitted_run_mode") != expected_run_mode:
            raise NormalizerLeakageError(
                f"refusing to load a normaliser fitted under run_mode="
                f"'{data.get('fitted_run_mode')}' into a '{expected_run_mode}' run "
                f"(idea.md S16.12)"
            )
        return cls(**data)

"""M12 -- statistics for comparing arms (D-012, falsification question 12).

Neither source document specifies a seed count or a significance test, yet a "gain" reported
from one seed is not evidence.  This module makes that mechanical:

- :func:`aggregate_seeds` returns mean +/- std and refuses to report a std from one seed.
- :func:`compare_arms` raises :class:`InsufficientSeedsError` below the configured minimum,
  so a single-seed difference cannot be presented as a result.
- The test is a **paired bootstrap over per-task scores**, pairing by ``(task, seed)``.
  Pairing matters: tasks differ enormously in difficulty, so an unpaired test would be
  dominated by between-task variance and would have almost no power.
- An effect size (Cohen's d on the paired differences) is always returned next to any
  p-value, because a p-value alone says nothing about magnitude.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from amaun.errors import InsufficientSeedsError

# D-012 recommendation: >= 3 seeds for any reported arm.
MIN_SEEDS_FOR_COMPARISON = 3


@dataclass
class SeedAggregate:
    mean: float
    std: float | None
    n: int
    values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"mean": self.mean, "std": self.std, "n": self.n, "values": list(self.values)}

    def format(self) -> str:
        if self.std is None:
            return f"{self.mean:.4f} (n=1, no dispersion estimate)"
        return f"{self.mean:.4f} +/- {self.std:.4f} (n={self.n})"


def aggregate_seeds(values: Sequence[float]) -> SeedAggregate:
    if not values:
        raise InsufficientSeedsError("cannot aggregate zero values")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return SeedAggregate(mean=mean, std=None, n=1, values=list(values))
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)   # sample std
    return SeedAggregate(mean=mean, std=math.sqrt(variance), n=n, values=list(values))


@dataclass
class ComparisonResult:
    arm_a: str
    arm_b: str
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size_d: float
    n_pairs: int
    n_seeds: int
    bootstrap_samples: int
    significant: bool
    interpretation: str

    def as_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return asdict(self)


def _paired_values(
    a: Mapping[tuple[str, int], float], b: Mapping[tuple[str, int], float]
) -> tuple[list[float], set[int]]:
    keys = sorted(set(a) & set(b))
    return [a[key] - b[key] for key in keys], {key[1] for key in keys}


def compare_arms(
    arm_a: str,
    arm_b: str,
    scores_a: Mapping[tuple[str, int], float],
    scores_b: Mapping[tuple[str, int], float],
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    min_seeds: int = MIN_SEEDS_FOR_COMPARISON,
) -> ComparisonResult:
    """Paired bootstrap comparison. Keys are ``(task_id, seed)``.

    Returns the mean difference ``a - b``, a percentile CI, a two-sided bootstrap p-value,
    and Cohen's d on the paired differences.
    """
    differences, seeds = _paired_values(scores_a, scores_b)
    if not differences:
        raise InsufficientSeedsError(f"no (task, seed) pairs shared by '{arm_a}' and '{arm_b}'")
    if len(seeds) < min_seeds:
        raise InsufficientSeedsError(
            f"refusing to compare '{arm_a}' with '{arm_b}': only {len(seeds)} seed(s) present, "
            f"minimum is {min_seeds}. A difference from fewer seeds cannot be distinguished "
            f"from run-to-run noise (docs/decision_register.md D-012)."
        )

    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    n = len(differences)
    resampled_means = []
    for _ in range(n_bootstrap):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(sample) / n)
    resampled_means.sort()

    low_index = int((alpha / 2) * n_bootstrap)
    high_index = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))
    ci_low, ci_high = resampled_means[low_index], resampled_means[high_index]

    # Two-sided p-value: proportion of centred resamples at least as extreme as observed.
    centred = [m - observed for m in resampled_means]
    extreme = sum(1 for value in centred if abs(value) >= abs(observed))
    p_value = (extreme + 1) / (n_bootstrap + 1)

    mean_difference = observed
    if n > 1:
        variance = sum((d - mean_difference) ** 2 for d in differences) / (n - 1)
        std = math.sqrt(variance)
        effect = mean_difference / std if std > 0 else 0.0
    else:
        effect = 0.0

    significant = not (ci_low <= 0.0 <= ci_high)
    if significant:
        direction = "higher" if mean_difference > 0 else "lower"
        interpretation = (
            f"'{arm_a}' scores {abs(mean_difference):.4f} {direction} than '{arm_b}' "
            f"(95% CI [{ci_low:.4f}, {ci_high:.4f}], d={effect:.3f}, {len(seeds)} seeds)"
        )
    else:
        interpretation = (
            f"no difference distinguishable from noise: CI [{ci_low:.4f}, {ci_high:.4f}] "
            f"contains zero ({len(seeds)} seeds, {n} paired observations)"
        )

    return ComparisonResult(
        arm_a=arm_a,
        arm_b=arm_b,
        mean_difference=mean_difference,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        effect_size_d=effect,
        n_pairs=n,
        n_seeds=len(seeds),
        bootstrap_samples=n_bootstrap,
        significant=significant,
        interpretation=interpretation,
    )

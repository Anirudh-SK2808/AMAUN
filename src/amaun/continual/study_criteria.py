"""The l4_full_study's judgement rules, as pure functions of measured numbers.

Everything that turns measurements into a verdict lives here -- acceptability, the oracle,
decision quality, the adapter cap, fusion-config selection, task eligibility, the probe rule and
the simple policies' candidate choices, and the paired bootstrap.  None of it touches a model, so
each rule is unit-tested on hand-built numbers (``tests/test_study_criteria.py``) and the study
can only apply it, not reinterpret it.

Definitions are docs/l4_next_run_prompt.md items 4-8 and 15-18; the choices that prompt leaves
open are docs/decision_register.md D-018 (cap fallback) and D-020 (``amaun_train_first``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: New-task equivalence: an option is as good as CREATE when it keeps >= 98 % of CREATE's score.
EQUIVALENCE = 0.98
#: Forgetting tolerance: every old task may lose at most 1 % of its standalone score.
FORGETTING = 0.01
#: Eligibility: a dedicated adapter must beat zero-shot by >= 5 % relative to discriminate options.
ELIGIBILITY = 0.05
#: D-018: what a CREATE becomes when the repository is at its cap.
CAP_FALLBACK = "BACKBONE"
#: Absolute slack on the threshold comparisons above.  A score exactly on a boundary can compute
#: just past it in floating point (0.21 vs 0.20 gives a gain of 0.0499..., 1.0 -> 0.99 a drop of
#: 0.0100...01), which would fail a ">= 5 %" or "<= 1 %" the spec states.  Far below any real gap.
TOLERANCE = 1e-9

BACKBONE = "BACKBONE"
CREATE = "CREATE"

#: D-027: metrics on an interval scale -- no meaningful zero, so "x % of the score" is not defined.
#: Rescaled BERTScore-F1 subtracts the encoder's baseline and is mostly negative on ELI5 (backbone
#: mean -0.11 on 16 TRAIN rows), where every ratio rule above would flip direction.  For these
#: metrics the equivalence bar is a share of CREATE's gain over zero-shot (:func:`equivalence_bar`),
#: and forgetting is a share of the adapter's gain lost (:func:`gain_drop`).
INTERVAL_SCALE_METRICS = frozenset({"bertscore_f1"})


# --------------------------------------------------------------------------------------
# Relative scores
# --------------------------------------------------------------------------------------


def at_least(value: float, bound: float) -> bool:
    """``value >= bound``, counting a value on the boundary up to float error as meeting it."""
    return value >= bound - TOLERANCE


def at_most(value: float, bound: float) -> bool:
    """``value <= bound``, counting a value on the boundary up to float error as meeting it."""
    return value <= bound + TOLERANCE


def equivalence_bar(reference: float, *, equivalence: float = EQUIVALENCE,
                    zero_shot: float | None = None) -> float:
    """The score an option must reach to count as equal to ``reference`` (D-027).

    Ratio-scale metrics: ``equivalence * reference`` (written ``reference - (1 - eq) * |reference|``,
    which is the same for every non-negative reference and keeps the margin below a negative one).
    Interval-scale metrics (``zero_shot`` given): ``zero_shot + eq * (reference - zero_shot)`` --
    the option keeps ``eq`` of the reference's gain over the backbone, whatever the scale's zero.
    """
    if zero_shot is not None:
        return zero_shot + equivalence * (reference - zero_shot)
    return reference - (1.0 - equivalence) * abs(reference)


def gain_drop(standalone: float, post: float, zero_shot: float) -> float:
    """Interval-scale forgetting: share of the adapter's gain over zero-shot that was lost (D-027).

    With no gain to lose (``standalone <= zero_shot``) the drop is measured against ``|standalone|``.
    """
    gain = standalone - zero_shot
    if gain > 0:
        return max(0.0, (standalone - post) / gain)
    return max(0.0, (standalone - post) / abs(standalone)) if standalone else 0.0


def relative_score(post: float, reference: float) -> float:
    """``post / reference``; a zero reference counts as matched by any non-negative score."""
    if reference > 0:
        return post / reference
    return 1.0 if post >= reference else 0.0


def relative_drop(standalone: float, post: float) -> float:
    """Relative loss of an old task's score, ``max(0, (standalone - post) / standalone)``."""
    if standalone <= 0:
        return 0.0
    return max(0.0, (standalone - post) / standalone)


def is_beneficial(relative_new: float, max_old_drop: float, *, equivalence: float = EQUIVALENCE,
                  forgetting: float = FORGETTING) -> bool:
    """The measured supervision label (D-019): new task kept within 2 %, no old task down > 1 %."""
    return at_least(relative_new, equivalence) and at_most(max_old_drop, forgetting)


# --------------------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------------------


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], *, weight_b: float = 1.0,
                        resamples: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Percentile CI of ``mean(a - weight_b * b)`` over examples resampled with replacement.

    ``a`` and ``b`` are per-example scores on the same examples in the same order.  One seed, so
    the interval reflects example sampling only (docs/experiment_protocol.md S10).
    """
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}")
    if not a:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "resamples": resamples}
    diffs = [x - weight_b * y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low = means[int((alpha / 2) * resamples)]
    high = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return {"mean": sum(diffs) / n, "ci_low": low, "ci_high": high, "n": n, "resamples": resamples,
            "contains_zero": low <= 0.0 <= high}


# --------------------------------------------------------------------------------------
# Options, acceptability, oracle, decision quality
# --------------------------------------------------------------------------------------


@dataclass
class OptionMeasure:
    """One measured option for one evaluation task (point estimates plus per-example scores)."""

    key: str                                   # BACKBONE | CREATE | REUSE:<id> | FUSE:<id>
    new_score: float
    old_drops: dict[str, float] = field(default_factory=dict)   # task -> relative drop
    adapters_after: int = 0
    training_seconds: float = 0.0
    stored_bytes: int = 0
    per_example: list[float] | None = None

    @property
    def outcome(self) -> str:
        return self.key.partition(":")[0]

    @property
    def forgetting(self) -> float:
        return max(self.old_drops.values(), default=0.0)

    def cost(self) -> tuple[int, float]:
        return (self.adapters_after, self.training_seconds)


def acceptability(option: OptionMeasure, create: OptionMeasure, *, equivalence: float = EQUIVALENCE,
                  forgetting: float = FORGETTING, resamples: int = 1000, seed: int = 0,
                  zero_shot: OptionMeasure | None = None) -> dict[str, Any]:
    """Acceptable on point estimates; inconclusive when the CI of (option - bar) holds 0.

    ``zero_shot`` (the BACKBONE option) switches to the interval-scale bar of D-027; its per-example
    scores make the per-example bar ``zs_i + eq * (create_i - zs_i)`` for the paired CI.
    """
    zs = None if zero_shot is None else zero_shot.new_score
    bar = equivalence_bar(create.new_score, equivalence=equivalence, zero_shot=zs)
    acceptable = at_least(option.new_score, bar) and at_most(option.forgetting, forgetting)
    ci = None
    inconclusive = None
    if option.per_example is not None and create.per_example is not None and option.key != create.key:
        if zero_shot is not None and zero_shot.per_example is not None:
            bars = [z + equivalence * (c - z) for c, z in zip(create.per_example, zero_shot.per_example)]
            ci = paired_bootstrap_ci(option.per_example, bars, resamples=resamples, seed=seed)
        else:
            ci = paired_bootstrap_ci(option.per_example, create.per_example, weight_b=equivalence,
                                     resamples=resamples, seed=seed)
        inconclusive = ci["contains_zero"]
    out = {"acceptable": acceptable, "inconclusive": inconclusive,
           "ci_vs_equivalence_of_create": ci, "equivalence_bar": bar,
           "new_vs_create_ratio": relative_score(option.new_score, create.new_score),
           "forgetting": option.forgetting}
    if zs is not None:
        gain = create.new_score - zs
        out["scale"] = "interval (gain retention, D-027)"
        out["gain_retained_vs_create"] = (option.new_score - zs) / gain if gain else None
    return out


def feasible(option: OptionMeasure, *, at_cap: bool) -> bool:
    """At the cap only CREATE is infeasible: every other option keeps the adapter count."""
    return not (at_cap and option.outcome == CREATE)


def executed_option(chosen_key: str, *, at_cap: bool, fallback: str = CAP_FALLBACK) -> str:
    """What actually happens when a policy chooses ``chosen_key`` (D-018)."""
    if at_cap and chosen_key.partition(":")[0] == CREATE:
        return fallback
    return chosen_key


def choose_oracle(options: Mapping[str, OptionMeasure], *, at_cap: bool = False,
                  equivalence: float = EQUIVALENCE, forgetting: float = FORGETTING,
                  zero_shot: float | None = None) -> dict[str, Any]:
    """Cheapest acceptable option: fewest adapters, then least training, then higher score.

    Uncapped, CREATE is always acceptable against itself, so the oracle always exists.  Capped,
    if no feasible option is acceptable the oracle is the feasible option with the highest score
    among those within the forgetting tolerance (D-018); BACKBONE always qualifies.
    """
    create = options[CREATE]
    pool = [o for o in options.values() if feasible(o, at_cap=at_cap)]
    bar = equivalence_bar(create.new_score, equivalence=equivalence, zero_shot=zero_shot)
    acceptable = [o for o in pool if at_least(o.new_score, bar) and at_most(o.forgetting, forgetting)]
    if acceptable:
        best = min(acceptable, key=lambda o: (o.cost(), -o.new_score, o.key))
        return {"option": best.key, "acceptable": True, "rule": "cheapest acceptable"}
    safe = [o for o in pool if at_most(o.forgetting, forgetting)] or pool
    best = max(safe, key=lambda o: (o.new_score, -o.adapters_after, -o.training_seconds))
    return {"option": best.key, "acceptable": False,
            "rule": "cap: no feasible acceptable option; best feasible score within the forgetting tolerance"}


def decision_quality(chosen_key: str, options: Mapping[str, OptionMeasure], oracle_key: str, *,
                     equivalence: float = EQUIVALENCE, forgetting: float = FORGETTING,
                     zero_shot: float | None = None) -> dict[str, Any]:
    """Item 7: decision correct, quality shortfall, cost excess, forgetting."""
    chosen, oracle, create = options[chosen_key], options[oracle_key], options[CREATE]
    bar = equivalence_bar(create.new_score, equivalence=equivalence, zero_shot=zero_shot)
    acceptable = at_least(chosen.new_score, bar) and at_most(chosen.forgetting, forgetting)
    correct = chosen_key == oracle_key or (acceptable and chosen.cost() <= oracle.cost())
    return {
        "decision_correct": bool(correct),
        "acceptable": bool(acceptable),
        "quality_shortfall": max(0.0, bar - chosen.new_score),
        "cost_excess": {"adapters": chosen.adapters_after - oracle.adapters_after,
                        "training_seconds": chosen.training_seconds - oracle.training_seconds},
        "forgetting": chosen.forgetting,
    }


# --------------------------------------------------------------------------------------
# Fusion selection (Phase 2)
# --------------------------------------------------------------------------------------


def select_fusion_config(table: Mapping[str, Sequence[Mapping[str, float]]], *,
                         ineligible: Sequence[str] = (), forgetting: float = FORGETTING,
                         equivalence: float = EQUIVALENCE) -> dict[str, Any]:
    """Item 4.  ``table[config] = [{"relative_new": r, "old_drop": d}, ...]`` over the pairs.

    Highest mean relative new-task score among configs with mean old drop <= 1 %; else the lowest
    mean forgetting.  Ties keep the table order.  ``fusion_viable``: the chosen config has
    ``r >= 0.98 and d <= 0.01`` on at least 2 pairs.
    """
    summary = {}
    for name, rows in table.items():
        if not rows:
            continue
        summary[name] = {
            "mean_relative_new": sum(r["relative_new"] for r in rows) / len(rows),
            "mean_old_drop": sum(r["old_drop"] for r in rows) / len(rows),
            "pairs_within_both_limits": sum(1 for r in rows if at_least(r["relative_new"], equivalence)
                                            and at_most(r["old_drop"], forgetting)),
            "n_pairs": len(rows),
            "eligible": name not in ineligible,
        }
    eligible = [n for n, s in summary.items() if s["eligible"]]
    if not eligible:
        raise ValueError("no eligible fusion config was measured")
    within = [n for n in eligible if at_most(summary[n]["mean_old_drop"], forgetting)]
    if within:
        chosen = max(within, key=lambda n: (summary[n]["mean_relative_new"], -eligible.index(n)))
        rule = "highest mean relative new-task score among configs with mean old drop <= 1%"
    else:
        chosen = min(eligible, key=lambda n: (summary[n]["mean_old_drop"], eligible.index(n)))
        rule = "no config within the forgetting limit: lowest mean forgetting"
    viable = summary[chosen]["pairs_within_both_limits"] >= 2
    return {"chosen": chosen, "rule": rule, "fusion_viable": viable, "summary": summary}


# --------------------------------------------------------------------------------------
# Eligibility (Phase 5)
# --------------------------------------------------------------------------------------


def eligibility(adapter_score: float, zero_shot_score: float, *, threshold: float = ELIGIBILITY) -> dict[str, Any]:
    """Item 25: the dedicated adapter must beat zero-shot by >= 5 % relative on VALIDATION."""
    if zero_shot_score != 0:
        # |zero-shot|: the same as before for every positive score, and the right direction for a
        # negative one (rescaled BERTScore, D-027).
        gain = (adapter_score - zero_shot_score) / abs(zero_shot_score)
        eligible = at_least(gain, threshold)
    else:
        gain = None
        eligible = adapter_score > 0
    return {"adapter_score": adapter_score, "zero_shot_score": zero_shot_score,
            "relative_gain": gain, "threshold": threshold, "eligible": eligible,
            "status": "eligible" if eligible else "non_discriminating"}


# --------------------------------------------------------------------------------------
# Probe-based policies (items 15-17)
# --------------------------------------------------------------------------------------


def best_by_probe_score(candidates: Mapping[str, Mapping[str, float]]) -> str | None:
    """Highest probe score; ties broken by lower probe loss, then id."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: (-candidates[c]["score"], candidates[c]["loss"], c))


def best_by_loss_gain(candidates: Mapping[str, Mapping[str, float]], backbone_loss: float) -> str | None:
    """Largest probe loss gain over the backbone; ties broken by id."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: (-(backbone_loss - candidates[c]["loss"]), c))


def top_k_by_loss_gain(candidates: Mapping[str, Mapping[str, float]], backbone_loss: float, k: int) -> list[str]:
    ordered = sorted(candidates, key=lambda c: (-(backbone_loss - candidates[c]["loss"]), c))
    return ordered[: max(0, k)]


def backbone_suffices(backbone: Mapping[str, float], candidates: Mapping[str, Mapping[str, float]], *,
                      equivalence: float = EQUIVALENCE) -> bool:
    best = best_by_probe_score(candidates)
    if best is None:
        return True
    return at_least(backbone["score"], equivalence_bar(candidates[best]["score"], equivalence=equivalence))


def probe_rule(backbone: Mapping[str, float], candidates: Mapping[str, Mapping[str, float]], *,
               equivalence: float = EQUIVALENCE) -> dict[str, Any]:
    """Item 16, on SUPPORT probes ``{"loss", "score"}`` for the backbone and each candidate."""
    best = best_by_probe_score(candidates)
    if best is not None:
        cand = candidates[best]
        if cand["loss"] < backbone["loss"] and at_least(cand["score"], equivalence_bar(backbone["score"],
                                                                                    equivalence=equivalence)):
            return {"option": f"REUSE:{best}",
                    "reason": f"'{best}' lowers SUPPORT loss vs the backbone and keeps >= "
                              f"{equivalence:.0%} of its probe score"}
    if backbone_suffices(backbone, candidates, equivalence=equivalence):
        return {"option": BACKBONE,
                "reason": f"backbone probe score >= {equivalence:.0%} of the best candidate's"}
    return {"option": CREATE, "reason": "no candidate or the backbone is good enough on SUPPORT"}


def amaun_order(*, reuse_choice: str | None, backbone: Mapping[str, float],
                candidates: Mapping[str, Mapping[str, float]], utilities: Mapping[str, float] | None,
                tau_u: float | None, equivalence: float = EQUIVALENCE) -> dict[str, Any]:
    """Item 15: REUSE (gate) -> BACKBONE (probe) -> FUSE (U >= tau_u) -> CREATE."""
    if reuse_choice is not None:
        return {"option": f"REUSE:{reuse_choice}", "step": 1, "reason": "reuse evidence fired"}
    if backbone_suffices(backbone, candidates, equivalence=equivalence):
        return {"option": BACKBONE, "step": 1,
                "reason": f"backbone probe score >= {equivalence:.0%} of the best candidate's"}
    if utilities and tau_u is not None:
        best = max(utilities, key=lambda c: (utilities[c], c))
        if utilities[best] >= tau_u:
            return {"option": f"FUSE:{best}", "step": 2,
                    "reason": f"U={utilities[best]:.4f} >= tau_u={tau_u:.4f}"}
        return {"option": CREATE, "step": 3, "reason": f"max U={utilities[best]:.4f} < tau_u={tau_u:.4f}"}
    return {"option": CREATE, "step": 3, "reason": "no utilities"}

"""M13 -- the baseline registry, including the citation gate.

Brief S14 names LoRM, FedALoRA, and FedZKD.  I have not verified those references, and
reconstructing a published method from its name would fabricate a baseline -- prohibited by
brief development rule 1 and idea.md S16.1.  Each therefore has an entry that raises
:class:`BaselineRequiresCitationError` stating exactly what is needed to implement it fairly.

This is deliberately a loud failure rather than a silent omission: a missing baseline should
block a claim of completeness, not quietly disappear from the results table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from amaun.baselines.strategies import (
    AmaunStrategy,
    FixedFusionStrategy,
    FixedRuleStrategy,
    ManagementStrategy,
    NaiveAverageStrategy,
    SeparateLoRAStrategy,
    SimilarityConfig,
    SimilaritySelectionStrategy,
)
from amaun.errors import BaselineRequiresCitationError


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    description: str
    implemented: bool
    source: str
    requirement: str | None = None


# Fully specified by idea.md S12.4 and brief S14, hence implementable faithfully.
IMPLEMENTED: dict[str, BaselineSpec] = {
    "B1_sequential_full_ft": BaselineSpec(
        "B1_sequential_full_ft",
        "Sequential full fine-tuning of the backbone (upper bound on plasticity, worst forgetting)",
        True,
        "brief S14 item 1",
    ),
    "B2_separate_lora": BaselineSpec(
        "B2_separate_lora", "One LoRA adapter per task, never merged", True, "brief S14 item 2"
    ),
    "B3_naive_average": BaselineSpec(
        "B3_naive_average", "Uniform-coefficient delta averaging", True, "brief S14 item 3"
    ),
    "B4_similarity_selection": BaselineSpec(
        "B4_similarity_selection", "Single-metric similarity selection", True, "brief S14 item 4 / idea.md S12.4"
    ),
    "B5_fixed_fusion": BaselineSpec(
        "B5_fixed_fusion", "Fusion with fixed, non-learnable coefficients", True, "idea.md S12.4"
    ),
    "B6_heuristic_compression": BaselineSpec(
        "B6_heuristic_compression",
        "Repository compression by age/usage heuristic instead of utility",
        True,
        "H5 comparison (idea.md S16.7 forbids this mechanism for AMAUN itself)",
    ),
}

# Named in brief S14 but NOT implementable without their references (D-007).
REQUIRES_CITATION: dict[str, BaselineSpec] = {
    "LoRM": BaselineSpec(
        "LoRM", "Named in brief S14 item 5", False, "brief S14 item 5",
        requirement=(
            "Provide: (1) the paper (title, authors, venue, year); (2) official code or exact "
            "algorithm; (3) the hyperparameters used; (4) its training budget, so budget parity "
            "can be enforced. Without these, any 'LoRM' row would be a method I invented and "
            "labelled with someone else's name."
        ),
    ),
    "FedALoRA": BaselineSpec(
        "FedALoRA", "Named in brief S14 item 6", False, "brief S14 item 6",
        requirement=(
            "Provide the reference and its aggregation rule. 'Federated adaptive LoRA' describes "
            "several distinct published methods; guessing which one is meant would misattribute "
            "results."
        ),
    ),
    "FedZKD": BaselineSpec(
        "FedZKD", "Named in brief S14 item 7", False, "brief S14 item 7",
        requirement=(
            "Provide the reference. Zero-shot/data-free KD in federated settings has several "
            "incompatible formulations (generator-based, ensemble-logit, synthetic-data), and "
            "they are not interchangeable."
        ),
    ),
}

ALL_SPECS: dict[str, BaselineSpec] = {**IMPLEMENTED, **REQUIRES_CITATION}


def describe(baseline_id: str) -> BaselineSpec:
    if baseline_id not in ALL_SPECS:
        raise KeyError(f"unknown baseline '{baseline_id}'; known: {sorted(ALL_SPECS)}")
    return ALL_SPECS[baseline_id]


def build_strategy(baseline_id: str, config: Mapping[str, Any]) -> ManagementStrategy:
    """Construct a baseline strategy, or refuse with the exact information needed."""
    if baseline_id in REQUIRES_CITATION:
        spec = REQUIRES_CITATION[baseline_id]
        raise BaselineRequiresCitationError(
            f"baseline '{baseline_id}' is registered but NOT implemented.\n"
            f"  Source: {spec.source}\n"
            f"  Why not: reconstructing a published method from its name alone would fabricate a "
            f"baseline (brief development rule 1, idea.md S16.1).\n"
            f"  To enable it: {spec.requirement}\n"
            f"  See docs/decision_register.md D-007."
        )

    if baseline_id == "B2_separate_lora":
        return SeparateLoRAStrategy()
    if baseline_id == "B3_naive_average":
        return NaiveAverageStrategy()
    if baseline_id in {"B4_similarity_selection", "B5_fixed_fusion", "A2_fixed_rule"}:
        section = config.get("similarity") or {}
        similarity = SimilarityConfig(
            feature=str(section.get("feature", "semantic.d_task_sq")),
            threshold=float(section.get("threshold", 0.0)),
            higher_is_more_similar=bool(section.get("higher_is_more_similar", False)),
        )
        if baseline_id == "B4_similarity_selection":
            return SimilaritySelectionStrategy(similarity)
        if baseline_id == "B5_fixed_fusion":
            return FixedFusionStrategy(similarity)
        return FixedRuleStrategy(similarity)
    if baseline_id == "B1_sequential_full_ft":
        raise BaselineRequiresCitationError(
            "B1_sequential_full_ft is not a management strategy: it trains the backbone itself "
            "and therefore runs through scripts/run_baseline.py --baseline B1, which requires "
            "the explicit backbone_trainable flag. It is the one arm where the frozen-backbone "
            "invariant is intentionally suspended, and that suspension must be visible in the "
            "config rather than hidden inside a strategy object."
        )
    if baseline_id == "B6_heuristic_compression":
        # B6 differs only in the compression mechanism, so it reuses AMAUN's selection with
        # repository.compression.mechanism = heuristic in its config.
        raise BaselineRequiresCitationError(
            "B6_heuristic_compression is configured, not constructed: set "
            "repository.compression.mechanism='heuristic' in configs/experiments/"
            "B6_heuristic_compression.yaml. It shares AMAUN's selection path so the ONLY "
            "difference is the compression rule (the H5 comparison)."
        )
    raise KeyError(f"no constructor for baseline '{baseline_id}'")


def implemented_ids() -> list[str]:
    return sorted(IMPLEMENTED)


def unimplemented_report() -> list[dict[str, Any]]:
    """Machine-readable record of what is missing and why, for the results directory."""
    return [
        {
            "baseline_id": spec.baseline_id,
            "source": spec.source,
            "implemented": False,
            "requirement": spec.requirement,
            "decision_register": "D-007",
        }
        for spec in REQUIRES_CITATION.values()
    ]

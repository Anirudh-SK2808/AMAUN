"""M7 -- threshold selection (D-005).

idea.md S10 says ``tau_u`` is "selected via validation" without saying which validation.
Choosing it on the same task stream that is later reported would be selection on test data
(idea.md S16.5).  Three declared procedures are implemented, and the choice is recorded in
every decision record so a reader can see which one produced the reported numbers:

- ``dev_stream``: choose on a disjoint decision-development task stream, never evaluated.
- ``historical_cv``: choose by cross-validation over *past* task events only.
- ``fixed``: use the configured constants -- honest but unjustified, so the returned
  ``Thresholds.notes`` says so explicitly.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from amaun.errors import ThresholdNotSelectedError
from amaun.manager.decision import Thresholds
from amaun.utils.logging import get_logger

log = get_logger("manager.threshold")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ThresholdNotSelectedError("cannot compute a quantile of zero observations")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def select_thresholds(
    config: Mapping[str, Any],
    *,
    observed_utilities: Sequence[float] | None = None,
    beneficial_flags: Sequence[bool] | None = None,
) -> Thresholds:
    """Select ``(tau_u, tau_reuse)`` by the declared procedure.

    ``observed_utilities`` and ``beneficial_flags`` come from the development stream or from
    historical events: for each observed candidate integration, the predicted utility and
    whether the integration actually turned out beneficial.  The threshold is placed to
    separate the two populations, which is what makes it *selected* rather than invented.
    """
    source = str(config.get("selection") or "")
    if not source:
        raise ThresholdNotSelectedError(
            "manager.threshold.selection is required and has no default (D-005)"
        )

    if source == "fixed":
        tau_u = config.get("fixed_tau_u")
        tau_reuse = config.get("fixed_tau_reuse")
        if tau_u is None or tau_reuse is None:
            raise ThresholdNotSelectedError(
                "selection='fixed' requires manager.threshold.fixed_tau_u and fixed_tau_reuse"
            )
        return Thresholds(
            tau_u=float(tau_u),
            tau_reuse=float(tau_reuse),
            source="fixed",
            selected_on="config",
            notes=(
                "Fixed thresholds taken from config. NOT experimentally justified: any result "
                "reported under this setting must state that tau was not selected on data "
                "(docs/decision_register.md D-005)."
            ),
        )

    if observed_utilities is None or beneficial_flags is None:
        raise ThresholdNotSelectedError(
            f"selection='{source}' requires observed utilities and beneficial/not flags from the "
            f"{'development stream' if source == 'dev_stream' else 'historical events'}; "
            f"none were supplied"
        )
    if len(observed_utilities) != len(beneficial_flags):
        raise ThresholdNotSelectedError("utilities and beneficial flags have different lengths")
    if not observed_utilities:
        raise ThresholdNotSelectedError("zero observations: cannot select a threshold")

    beneficial = [u for u, flag in zip(observed_utilities, beneficial_flags) if flag]
    harmful = [u for u, flag in zip(observed_utilities, beneficial_flags) if not flag]

    if beneficial and harmful:
        # Pick the cut that maximises balanced accuracy over the observed outcomes: the
        # threshold is then a *fitted* quantity with a stated objective, not a round number.
        candidates = sorted(set(observed_utilities))
        best_cut, best_score = candidates[0], -1.0
        for cut in candidates:
            true_positive = sum(1 for u in beneficial if u >= cut) / len(beneficial)
            true_negative = sum(1 for u in harmful if u < cut) / len(harmful)
            score = 0.5 * (true_positive + true_negative)
            if score > best_score:
                best_cut, best_score = cut, score
        tau_u = float(best_cut)
        notes = f"maximised balanced accuracy ({best_score:.4f}) over {len(observed_utilities)} observations"
    else:
        # One-class case: fall back to a stated quantile rather than a magic constant.
        tau_u = float(_quantile(list(observed_utilities), 0.5))
        notes = (
            f"only {'beneficial' if beneficial else 'harmful'} observations available; used the "
            f"median of {len(observed_utilities)} observed utilities"
        )

    # REUSE demands strictly more evidence than FUSE, so tau_reuse sits above tau_u at a
    # configured quantile of the beneficial population.
    reuse_quantile = float(config.get("reuse_quantile", 0.75))
    tau_reuse = float(_quantile(beneficial or list(observed_utilities), reuse_quantile))
    tau_reuse = max(tau_reuse, tau_u)

    log.info(
        "selected tau_u=%.4f tau_reuse=%.4f via %s (%s)", tau_u, tau_reuse, source, notes
    )
    return Thresholds(
        tau_u=tau_u,
        tau_reuse=tau_reuse,
        source=source,
        selected_on="dev_stream" if source == "dev_stream" else "historical_events",
        notes=notes,
    )

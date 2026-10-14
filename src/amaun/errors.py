"""Typed exceptions for AMAUN.

Every invariant in ``docs/system_specification.md`` has a dedicated exception so that a
violation is unambiguous in logs and testable in ``tests/``.  Nothing in this project
signals an invariant violation with a bare ``ValueError``.
"""

from __future__ import annotations


class AmaunError(Exception):
    """Base class for all AMAUN errors."""


# -- M0 config ---------------------------------------------------------------------
class ConfigError(AmaunError):
    """Malformed or inconsistent configuration."""


class MissingRunModeError(ConfigError):
    """``run_mode`` was not supplied.  It has no default by design (idea.md S0.1)."""


class RunModeArtifactError(AmaunError):
    """An artefact write would cross run-mode namespaces (idea.md S0.1.1)."""


class QuantizationUnavailableError(ConfigError):
    """Quantised training was requested on a device that cannot provide it (D-008).

    Raised rather than silently downgrading precision, because a silent downgrade would
    misreport the training recipe.
    """


class UnresolvedDecisionError(ConfigError):
    """A run depends on a decision still marked OPEN in docs/decision_register.md."""


# -- M1 backbone -------------------------------------------------------------------
class BackboneNotFrozenError(AmaunError):
    """A backbone parameter has ``requires_grad=True`` (master_prompt constraint 4)."""


class BackboneMutatedError(AmaunError):
    """Backbone weights changed; the fingerprint no longer matches."""


class BackboneDuplicationError(AmaunError):
    """A second full-precision backbone was requested while one is resident."""


# -- M2 data -----------------------------------------------------------------------
class DataError(AmaunError):
    """Dataset construction or access failure."""


class UnpinnedDatasetError(DataError):
    """A dataset has no pinned identity/revision (D-006).  Guessing one is forbidden."""


class UnresolvedRowAdapterError(DataError):
    """A task's row->example rule is unset or unknown (D-015).

    Upstream rows and the task's prompt contract do not always line up (DailyDialog has
    ``dialog``, not ``history``; FinQA has no ``context``; ELI5's target is a *list* of answers).
    Which rule maps a row to an example changes **what the adapter learns**, so it is named per
    task in ``configs/datasets/`` with no code default: an unset or misspelled name raises here
    rather than being guessed at.
    """


class SplitAccessViolation(DataError):
    """A component read a split it is not permitted to read (leakage guard)."""


class LeakageError(DataError):
    """Two splits that must be disjoint share content."""


# -- M3/M4 adapters ----------------------------------------------------------------
class AdapterError(AmaunError):
    """Adapter construction, IO, or lifecycle failure."""


class UnauthorisedRegistrationError(AdapterError):
    """A candidate adapter was registered without an authorising Decision (constraint 10)."""


class IncompatibleAdapterError(AdapterError):
    """Adapter metadata is incompatible with the live backbone or with a peer adapter."""


# -- M7 manager --------------------------------------------------------------------
class ManagerError(AmaunError):
    """Retrieval, utility prediction, or decision failure."""


class ThresholdNotSelectedError(ManagerError):
    """A decision was attempted before thresholds were selected from a declared source."""


class NormalizerLeakageError(ManagerError):
    """Normalisation statistics were fitted on, or reused across, a forbidden split/run mode."""


# -- M8 fusion ---------------------------------------------------------------------
class FusionError(AmaunError):
    """Fusion construction or optimisation failure."""


class FusionSourceMutatedError(FusionError):
    """A source adapter's delta changed during fusion (invariant I8.4)."""


# -- M11 federated -----------------------------------------------------------------
class FederatedError(AmaunError):
    """Client/server protocol failure."""


class PayloadContainsBackboneError(FederatedError):
    """A federated payload contains backbone tensors (constraint 4)."""


# -- M13 baselines -----------------------------------------------------------------
class BaselineRequiresCitationError(AmaunError):
    """A named literature baseline cannot be implemented without its reference (D-007).

    Reconstructing a published method from its name alone would fabricate a baseline,
    which master_prompt constraint 1 forbids.
    """


class BudgetParityError(AmaunError):
    """Two arms were compared under materially different budgets (invariant I13.1)."""


# -- M12 evaluation ----------------------------------------------------------------
class EvaluationError(AmaunError):
    """Evaluation failure."""


class InsufficientSeedsError(EvaluationError):
    """A difference was claimed from too few seeds (D-012)."""


class PreconditionError(AmaunError):
    """A reportable run was attempted before its preconditions were met."""

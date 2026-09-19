"""The only module permitted to construct result paths.

idea.md S0.1.1 mandates ``results/<run_mode>/<experiment_id>/`` and *exactly* that
top-level split.  Centralising path construction here means the separation cannot be
bypassed by a stray ``open()`` elsewhere in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from amaun.config import ResolvedConfig, RunMode
from amaun.errors import RunModeArtifactError

RESULTS_ROOT = Path("results")
ADAPTERS_ROOT = Path("adapters")
DATA_ROOT = Path("data")

# idea.md S0.1.1: "The top-level split is exactly results/local_smoke/ vs results/actual/"
# -- plus the dummy test tier (D-001), which is likewise namespaced.
_ALLOWED_TOP_LEVEL = {m.value for m in RunMode}


@dataclass(frozen=True)
class RunPaths:
    """Every artefact location for one run.  Created by :func:`run_paths`."""

    root: Path
    run_mode: RunMode
    experiment_id: str

    # -- files -------------------------------------------------------------------
    @property
    def config_used(self) -> Path:
        return self.root / "config_used.json"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions.jsonl"

    @property
    def repository_events(self) -> Path:
        return self.root / "repository_events.jsonl"

    @property
    def budget_ledger(self) -> Path:
        return self.root / "budget_ledger.json"

    @property
    def performance_matrix(self) -> Path:
        return self.root / "performance_matrix.json"

    # -- directories -------------------------------------------------------------
    @property
    def fusion_logs(self) -> Path:
        return self.root / "fusion_logs"

    @property
    def federated_rounds(self) -> Path:
        return self.root / "federated_rounds"

    @property
    def utility_network(self) -> Path:
        return self.root / "utility_network"

    @property
    def repository_snapshots(self) -> Path:
        return self.root / "repository_snapshots"

    @property
    def adapters(self) -> Path:
        """Adapter store, namespaced by run mode so checkpoints can never cross tiers.

        idea.md S16.12 forbids loading 1.5B checkpoints into an ``actual`` run and vice
        versa; separate roots make that a filesystem-level guarantee rather than a
        convention.
        """
        return ADAPTERS_ROOT / self.run_mode.value / self.experiment_id

    @property
    def candidates(self) -> Path:
        """Scratch store for temporary adapters (idea.md S3: never in the repository)."""
        return ADAPTERS_ROOT / self.run_mode.value / self.experiment_id / "_candidates"

    def ensure(self) -> "RunPaths":
        for directory in (
            self.root,
            self.fusion_logs,
            self.federated_rounds,
            self.utility_network,
            self.repository_snapshots,
            self.adapters,
            self.candidates,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def run_paths(cfg: ResolvedConfig, *, base: Path | str = RESULTS_ROOT) -> RunPaths:
    """Build the run's paths.  ``run_mode`` is a required argument by construction."""
    base = Path(base)
    root = base / cfg.run_mode.value / cfg.experiment_id
    return RunPaths(root=root, run_mode=cfg.run_mode, experiment_id=cfg.experiment_id)


def assert_write_allowed(path: Path, run_mode: RunMode, *, base: Path | str = RESULTS_ROOT) -> None:
    """Refuse a write that would cross run-mode namespaces (idea.md S0.1.1).

    Called by the tracking layer before every artefact write.  A ``local_smoke`` run that
    tries to write into ``results/actual/`` is a methodology error, not a typo.
    """
    base = Path(base)
    try:
        relative = Path(path).resolve().relative_to(base.resolve())
    except ValueError:
        return  # not under the results root; not our concern
    parts = relative.parts
    if not parts:
        return
    top = parts[0]
    if top not in _ALLOWED_TOP_LEVEL:
        raise RunModeArtifactError(
            f"results/ may only contain run-mode directories {sorted(_ALLOWED_TOP_LEVEL)}, "
            f"got '{top}' (idea.md S0.1.1)"
        )
    if top != run_mode.value:
        raise RunModeArtifactError(
            f"run_mode={run_mode.value} attempted to write under results/{top}/. "
            f"Smoke and actual artefacts must never mix (idea.md S0.1.1)."
        )


def smoke_label(run_mode: RunMode) -> str:
    """The mandatory visible label for non-reportable output (idea.md S0.1.1)."""
    if run_mode.is_reportable:
        return ""
    return f"[SMOKE TEST ({run_mode.value}) -- not a reported result]"

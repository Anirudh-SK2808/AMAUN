"""Checkpoint and resume for a long single run (l4_full_study item 28).

One JSON file, ``decision_study/checkpoint/state.json``, rewritten atomically after every unit of
work (a trained adapter, a fusion config x pair, a supervision pair, an evaluation task, a stream
run) and every phase.  Adapters themselves are already on disk (repository / candidate store);
the checkpoint records which units finished, their results, the measured-score caches (so R1's
"computed once" survives a restart), and the wall time spent before each resume.

A checkpoint is bound to the config hash it was written under: resuming with a different
configuration would mix two experiments in one result, so it is refused.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amaun.errors import PreconditionError


class StudyCheckpoint:
    def __init__(self, path: Path | str, *, config_hash: str, experiment_id: str, resume: bool) -> None:
        self.path = Path(path)
        self.resumed = False
        if self.path.exists():
            if not resume:
                raise PreconditionError(
                    f"a checkpoint already exists at {self.path}; pass --resume to continue that run, "
                    f"or use a new experiment id (results are never silently overwritten)")
            self.state = json.loads(self.path.read_text())
            if self.state.get("config_hash") != config_hash:
                raise PreconditionError(
                    f"refusing to resume: the checkpoint was written under config hash "
                    f"{str(self.state.get('config_hash'))[:12]} but this run resolves to {config_hash[:12]}")
            self.resumed = True
            self.state.setdefault("resumes", []).append({
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_phases": [p for p, v in self.state.get("phases", {}).items() if v == "done"],
                "completed_units": len(self.state.get("units", {})),
                "elapsed_seconds_before": self.state.get("elapsed_seconds", 0.0),
            })
        else:
            self.state = {"config_hash": config_hash, "experiment_id": experiment_id, "phases": {},
                          "units": {}, "caches": {}, "resumes": [], "elapsed_seconds": 0.0,
                          "created_utc": datetime.now(timezone.utc).isoformat()}
        self.elapsed_before = float(self.state.get("elapsed_seconds", 0.0))
        self.save()

    # -- phases and units ----------------------------------------------------------
    def phase_done(self, phase: str) -> bool:
        return self.state["phases"].get(phase) == "done"

    def mark_phase(self, phase: str, *, elapsed: float | None = None) -> None:
        self.state["phases"][phase] = "done"
        self.save(elapsed=elapsed)

    def unit(self, key: str) -> Any | None:
        return self.state["units"].get(key)

    def has_unit(self, key: str) -> bool:
        return key in self.state["units"]

    def save_unit(self, key: str, result: Any, *, elapsed: float | None = None) -> None:
        self.state["units"][key] = result
        self.save(elapsed=elapsed)

    # -- caches ---------------------------------------------------------------------
    def cache(self, name: str) -> dict[str, Any]:
        return self.state["caches"].setdefault(name, {})

    # -- persistence ----------------------------------------------------------------
    def save(self, *, elapsed: float | None = None) -> None:
        if elapsed is not None:
            self.state["elapsed_seconds"] = self.elapsed_before + elapsed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        os.replace(tmp, self.path)       # atomic: a crash mid-write never leaves a torn checkpoint

    @property
    def resumes(self) -> list[dict[str, Any]]:
        return list(self.state.get("resumes", []))

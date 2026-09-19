"""Structured experiment tracking (brief S21).

All records are JSONL with a ``schema_version``, a ``run_mode`` tag, and a UTC timestamp,
so results aggregate mechanically into tables and plots and never need parsing of prose
logs.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from amaun.config import ResolvedConfig
from amaun.utils.paths import RunPaths, assert_write_allowed, run_paths

SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and hasattr(value, "shape") and getattr(value, "ndim", 1) == 0:
        return value.item()  # 0-d numpy/torch scalar
    if hasattr(value, "tolist"):
        return value.tolist()  # arrays
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):  # enums
        return value.value
    return str(value)


class ExperimentTracker:
    """Append-only, thread-safe JSONL writer bound to one run.

    Binding to a ``ResolvedConfig`` is what makes ``assert_write_allowed`` enforceable:
    the tracker knows its run mode, so a cross-mode write is caught at the write site.
    """

    def __init__(self, cfg: ResolvedConfig, *, base: Path | str = "results") -> None:
        self.cfg = cfg
        self.paths: RunPaths = run_paths(cfg, base=base).ensure()
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> "ExperimentTracker":
        """Write ``config_used.json`` and ``provenance.json`` (idea.md S0.1.1)."""
        from amaun.utils.provenance import collect_provenance

        self._write_json(self.paths.config_used, self.cfg.to_config_used())
        self._write_json(self.paths.provenance, collect_provenance())
        return self

    # -- record writers ----------------------------------------------------------
    def log(self, stream: str, record: dict[str, Any]) -> dict[str, Any]:
        """Append one record to a named JSONL stream."""
        target = {
            "metrics": self.paths.metrics,
            "decisions": self.paths.decisions,
            "repository_events": self.paths.repository_events,
        }.get(stream, self.paths.root / f"{stream}.jsonl")
        with self._lock:
            self._counters[stream] = self._counters.get(stream, 0) + 1
            enriched = {
                "schema_version": SCHEMA_VERSION,
                "stream": stream,
                "seq": self._counters[stream],
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "run_mode": self.cfg.run_mode.value,
                "experiment_id": self.cfg.experiment_id,
                "seed": self.cfg.seed,
                "config_hash": self.cfg.config_hash,
                **_jsonable(record),
            }
            assert_write_allowed(target, self.cfg.run_mode)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as handle:
                handle.write(json.dumps(enriched, sort_keys=False) + "\n")
        return enriched

    def log_metric(self, **record: Any) -> dict[str, Any]:
        return self.log("metrics", record)

    def log_decision(self, **record: Any) -> dict[str, Any]:
        return self.log("decisions", record)

    def log_repository_event(self, **record: Any) -> dict[str, Any]:
        return self.log("repository_events", record)

    def write_json(self, relative: str | Path, payload: Any) -> Path:
        target = self.paths.root / relative
        self._write_json(target, payload)
        return target

    def _write_json(self, target: Path, payload: Any) -> None:
        assert_write_allowed(target, self.cfg.run_mode)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=False)
            handle.write("\n")

    # -- reading back ------------------------------------------------------------
    def read(self, stream: str) -> Iterator[dict[str, Any]]:
        target = self.paths.root / f"{stream}.jsonl"
        if stream == "metrics":
            target = self.paths.metrics
        elif stream == "decisions":
            target = self.paths.decisions
        elif stream == "repository_events":
            target = self.paths.repository_events
        if not target.exists():
            return iter(())
        return (json.loads(line) for line in target.read_text().splitlines() if line.strip())

"""Projected wall time of a whole study, before any long phase starts (l4_full_study item 28).

A projection is ``sum over work items of count x unit cost``.  The study lists its work items
(:class:`WorkItem`: phase, kind, task, count) from its own configuration -- the same loops the
phases run -- and this module prices them.

Unit costs per task and kind (seconds):

- ``train_step``      one optimiser step (micro-batches x accumulation), checkpointing as configured
- ``gen_example``     one evaluated example: teacher-forced loss + greedy generation at this run's
                      per-task cap and expected output length
- ``loss_example``    one teacher-forced example (validation losses, probes)
- ``fusion_step``     one fusion (+ KD) step
- ``probe_candidate`` feature extraction for one (task, candidate) pair

Two sources, both reported: run 1's measured timings, scaled to this run's per-task generation
caps; and a timing probe on this host (preflight).  The budget check uses, per (task, kind), the
larger of the two (conservative); a cost known for neither falls back to the mean over tasks of
that kind and is listed as such.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

KINDS = ("train_step", "gen_example", "loss_example", "fusion_step", "probe_candidate")


@dataclass(frozen=True)
class WorkItem:
    phase: str
    kind: str
    task: str
    count: float
    note: str = ""


def run1_unit_costs(run_dir: str | Path, *, generation_caps_run1: Mapping[str, int],
                    generation_caps_now: Mapping[str, int]) -> dict[str, dict[str, float]]:
    """Per-task unit costs measured by a finished decision-study run.

    Evaluation cost per example comes from ``metrics.jsonl`` (``wall_seconds / n_examples``, TEST
    and VALIDATION evaluations), scaled by ``cap_now / cap_run1`` for the task (a conservative
    over-estimate when the cap grows: not every generation reaches the cap).  Training cost per
    step comes from the adapters' records, fusion cost per step from ``fusion_logs``.
    """
    root = Path(run_dir)
    costs: dict[str, dict[str, list[float]]] = {}

    def add(task: str, kind: str, value: float) -> None:
        costs.setdefault(task, {}).setdefault(kind, []).append(value)

    metrics = root / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            phase = str(record.get("phase", ""))
            if not phase.startswith("decision_study:") or not record.get("n_examples"):
                continue
            task = record.get("task_id")
            seconds = float(record.get("wall_seconds") or 0.0) / int(record["n_examples"])
            scale = generation_caps_now.get(task, 1) / max(1, generation_caps_run1.get(task, 1))
            add(task, "gen_example", seconds * max(1.0, scale))
    adapters = root / "decision_study" / "initial_adapters.json"
    if adapters.exists():
        for item in json.loads(adapters.read_text()):
            if item.get("steps"):
                add(item["task_id"], "train_step", float(item["wall_seconds"]) / int(item["steps"]))
    for path in sorted((root / "decision_study").glob("evaluation_*.json")):
        payload = json.loads(path.read_text())
        temp = payload.get("temporary_adapter") or {}
        if temp.get("steps"):
            add(payload["task_id"], "train_step", float(temp["wall_seconds"]) / int(temp["steps"]))
    for path in sorted((root / "fusion_logs").glob("*.json")):
        payload = json.loads(path.read_text())
        steps = int(payload.get("steps") or 0)
        task = _fusion_task(path.stem, payload)
        if steps and task:
            add(task, "fusion_step", float(payload.get("wall_seconds") or 0.0) / steps)
    return {task: {kind: sum(v) / len(v) for kind, v in kinds.items()} for task, kinds in costs.items()}


def _fusion_task(stem: str, payload: Mapping[str, Any]) -> str | None:
    # run-1 ids: sup-seed-<new>-into-seed-<old>  /  eval-<task>-fuse-<candidate>
    if stem.startswith("eval-"):
        return stem[len("eval-"):].split("-fuse-")[0]
    if stem.startswith("sup-seed-"):
        return stem[len("sup-seed-"):].split("-into-")[0]
    return None


def merge_costs(*sources: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Per (task, kind), the largest cost any source measured."""
    out: dict[str, dict[str, float]] = {}
    for source in sources:
        for task, kinds in source.items():
            for kind, value in kinds.items():
                if value is None:
                    continue
                current = out.setdefault(task, {}).get(kind)
                out[task][kind] = value if current is None else max(current, value)
    return out


def project(items: Iterable[WorkItem], costs: Mapping[str, Mapping[str, float]], *,
            budget_hours: float) -> dict[str, Any]:
    """Price every work item; per-phase seconds, total, and whether it fits the budget."""
    means = {}
    for kind in KINDS:
        values = [c[kind] for c in costs.values() if kind in c]
        means[kind] = sum(values) / len(values) if values else None
    phases: dict[str, float] = {}
    fallbacks: set[str] = set()
    unpriced: list[dict[str, Any]] = []
    for item in items:
        unit = (costs.get(item.task) or {}).get(item.kind)
        if unit is None:
            unit = means.get(item.kind)
            if unit is None:
                unpriced.append(asdict(item))
                continue
            fallbacks.add(f"{item.task}:{item.kind}")
        phases[item.phase] = phases.get(item.phase, 0.0) + item.count * unit
    total = sum(phases.values())
    return {
        "per_phase_hours": {p: s / 3600.0 for p, s in phases.items()},
        "total_hours": total / 3600.0,
        "budget_hours": budget_hours,
        "fits": total / 3600.0 <= budget_hours and not unpriced,
        "unit_costs_seconds": {t: dict(k) for t, k in costs.items()},
        "fallback_mean_costs_used_for": sorted(fallbacks),
        "unpriced_items": unpriced,
    }


def summarise_items(items: Sequence[WorkItem]) -> dict[str, dict[str, float]]:
    """Counts per phase and kind (the workload half of the projection, for the report)."""
    out: dict[str, dict[str, float]] = {}
    for item in items:
        bucket = out.setdefault(item.phase, {})
        bucket[item.kind] = bucket.get(item.kind, 0.0) + item.count
    return out

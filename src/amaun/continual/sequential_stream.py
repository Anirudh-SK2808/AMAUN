"""Phase 6 of the l4_full_study: tasks arriving one after another into a repository that each
decision changes (``main_sequential`` semantics, frozen Phase-4 settings).

Starts empty.  At each arrival the policy decides against the *current* repository, the decision
is executed, and TEST scores are recorded:

- REUSE j   -> the task is served by j; the repository is unchanged;
- BACKBONE  -> the task is served by the frozen backbone;
- FUSE j    -> j and the task's adapter are fused with the frozen config (trained fresh in each
               stream run); the fused adapter replaces j and serves j's tasks and the new one;
- CREATE    -> the task's adapter joins the repository -- or, at ``max_adapters``, the task is
               served by the backbone (D-018).

Feasibility rules (D-023):

- **R2** the "temporary adapter" of task t is the adapter trained for t in Phase 1 (seed tasks) or
  Phase 5 (evaluation tasks): same task, rows and adapter id, hence the same derived seed;
- **R3** after a step only the new task and the tasks whose serving adapter changed are evaluated;
  every task is evaluated again on the final repository.  Evaluations are cached by adapter
  content, so this equals evaluating everything every step (``reevaluate_all_each_step`` runs the
  naive loop; the dummy-tier test compares the two).

Each step is checkpointed (fused adapters are written to disk), so a restart resumes mid-stream.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.metadata import AdapterOrigin
from amaun.adapters.repository import AdapterEntry
from amaun.continual import study_criteria as criteria
from amaun.data.splits import SplitName
from amaun.utils.logging import get_logger
from amaun.utils.resources import cleanup

log = get_logger("continual.sequential_stream")


class _RepositoryView:
    """What feature extraction reads from a repository (size, bytes, entries), for the stream's own
    repository rather than the study's seed repository."""

    def __init__(self, entries: list[AdapterEntry]) -> None:
        self._entries = entries

    def entries(self) -> list[AdapterEntry]:
        return list(self._entries)

    def memory_bytes(self) -> int:
        return sum(e.delta.storage_bytes() for e in self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def task_adapter(study: Any, task_id: str) -> tuple[AdapterDelta, AdapterEntry]:
    """R2: the adapter already trained for ``task_id`` (Phase 1 seed or Phase 5 temporary)."""
    seed_id = study.seed_by_task.get(task_id)
    if seed_id is not None:
        entry = study.repository.retrieve(seed_id)
        return entry.delta, entry
    temp = study._temp(task_id)
    return temp["delta"], temp["entry"]


def _materialise(study: Any, item: dict[str, Any]) -> AdapterEntry:
    source = item["source"]
    if source.startswith("file:"):
        delta = AdapterDelta.load(source[len("file:"):])
        base_task = item["metadata_from"]
        _, base = task_adapter(study, base_task)
    else:
        delta, base = task_adapter(study, source[len("task:"):])
    metadata = copy.deepcopy(base.metadata)
    metadata.adapter_id = item["adapter_id"]
    metadata.tasks_served = list(item["tasks_served"])
    if source.startswith("file:"):
        metadata.origin = AdapterOrigin.FUSED.value
    return AdapterEntry(delta=delta, metadata=metadata)


def run_stream(study: Any, *, policy: str, cap: int | None, name: str) -> dict[str, Any]:
    cfg = study.full.stream
    order = list(cfg["order"])
    naive = bool(cfg.get("reevaluate_all_each_step", False))
    state_key = f"P6:{name}:state"
    state = study.ckpt.unit(state_key) or {"step": 0, "entries": [], "serving": {}, "trajectory": [],
                                           "arrival": {}, "decisions": {}}
    stream_dir = Path(study.tracker.paths.adapters) / "_stream" / name

    def serving_delta(task_id: str, entries: list[AdapterEntry]) -> AdapterDelta | None:
        adapter_id = state["serving"].get(task_id)
        if adapter_id is None:
            return None
        return next(e.delta for e in entries if e.adapter_id == adapter_id)

    def evaluate(task_id: str, entries: list[AdapterEntry]) -> float:
        return study._measure(serving_delta(task_id, entries), task_id, split=SplitName.TEST,
                              n=study._examples("test", task_id),
                              label=f"stream:{name}:{task_id}:{state['serving'].get(task_id) or 'backbone'}")["primary_score"]

    for index in range(state["step"], len(order)):
        task_id = order[index]
        study.runner.ensure_representation(task_id)
        entries = [_materialise(study, item) for item in state["entries"]]
        own_delta, own_entry = task_adapter(study, task_id)
        if policy == "always_separate" or not entries:
            decision = {"option": criteria.CREATE,
                        "reason": "repository empty" if not entries else "policy: always create"}
        else:
            seeds = study.runner.repository
            study.runner.repository = _RepositoryView(entries)
            try:
                decided = study.decide(task_id=task_id, task_index=index, temp_delta=own_delta,
                                       candidates=entries, policies=[policy], log_as=f"stream:{name}")
            finally:
                study.runner.repository = seeds
            decision = decided["decisions"][policy]
        at_cap = cap is not None and len(entries) >= cap
        executed = criteria.executed_option(decision["option"], at_cap=at_cap)
        outcome, _, target = executed.partition(":")
        changed: list[str] = []
        if outcome == criteria.CREATE:
            state["entries"].append({"adapter_id": own_entry.adapter_id, "source": f"task:{task_id}",
                                     "tasks_served": [task_id]})
            state["serving"][task_id] = own_entry.adapter_id
        elif outcome == criteria.BACKBONE:
            state["serving"][task_id] = None
        elif outcome == "REUSE":
            item = next(i for i in state["entries"] if i["adapter_id"] == target)
            item["tasks_served"].append(task_id)
            state["serving"][task_id] = target
        else:   # FUSE
            item = next(i for i in state["entries"] if i["adapter_id"] == target)
            old = next(e for e in entries if e.adapter_id == target)
            fused_id = f"stream-{name}-fused-{index:02d}-{task_id}"
            fused, fusion = study._fuse_with(study.selected, old, own_delta, task_id, fused_id)
            path = stream_dir / fused_id / "delta.pt"
            fused.save(path)
            changed = [t for t, a in state["serving"].items() if a == target]
            position = state["entries"].index(item)
            base_task = item.get("metadata_from") or item["source"].partition(":")[2]
            state["entries"][position] = {"adapter_id": fused_id, "source": f"file:{path}",
                                          "metadata_from": base_task,
                                          "tasks_served": item["tasks_served"] + [task_id],
                                          "fusion": {k: fusion[k] for k in ("wall_seconds", "stored_bytes",
                                                                             "alpha_summary")}}
            for t in changed + [task_id]:
                state["serving"][t] = fused_id
            del fused
            cleanup(study.backbone.device)
        entries = [_materialise(study, item) for item in state["entries"]]
        to_evaluate = [t for t in order[: index + 1]] if naive else [task_id] + changed
        scores_now = {t: evaluate(t, entries) for t in to_evaluate}
        state["arrival"][task_id] = scores_now[task_id]
        state["decisions"][task_id] = {"chosen": decision["option"], "executed": executed,
                                       "reason": decision.get("reason"), "fallback": decision.get("fallback")}
        state["trajectory"].append({
            "step": index, "task_id": task_id, "chosen": decision["option"], "executed": executed,
            "repository_size": len(entries), "repository_bytes": sum(e.delta.storage_bytes() for e in entries),
            "reevaluated": to_evaluate, "scores": scores_now,
        })
        state["step"] = index + 1
        study.ckpt.save_unit(state_key, state, elapsed=study._elapsed())

    entries = [_materialise(study, item) for item in state["entries"]]
    final = {t: evaluate(t, entries) for t in order}
    forgetting = {t: criteria.relative_drop(state["arrival"][t], final[t]) for t in order}
    counts: dict[str, int] = {}
    for d in state["decisions"].values():
        outcome = d["executed"].partition(":")[0]
        counts[outcome] = counts.get(outcome, 0) + 1
    return {
        "policy": policy, "cap": cap, "order": order,
        "fallback": next((d["fallback"] for d in state["decisions"].values() if d.get("fallback")), None),
        "trajectory": state["trajectory"], "decisions": state["decisions"],
        "arrival_scores": state["arrival"], "final_scores": final, "forgetting": forgetting,
        "mean_final_score": sum(final.values()) / len(final),
        "decision_counts": counts,
        "final_repository": [{k: v for k, v in item.items() if k != "source"} for item in state["entries"]],
        "final_repository_size": len(entries),
        "final_repository_bytes": sum(e.delta.storage_bytes() for e in entries),
        "reevaluation": "naive (every task every step)" if naive else "R3 (new + changed tasks, then all at the end)",
    }

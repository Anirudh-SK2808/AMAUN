#!/usr/bin/env python
"""One-off checkpoint surgery for D-029: redo Phase 4 and Phase 5's decisions under a changed config.

    .venv/bin/python scripts/checkpoint_surgery_d029.py --config configs/experiments/l4_full_study.yaml \
        --run-mode local_smoke             # dry run: prints what would change
    ... --apply                            # then: run_full_study.py ... --resume

A checkpoint is bound to its config hash so that a resume can never mix two experiments.  D-029
changes the config on purpose (revised utility models, policy ``amaun_v0``), so this script
overrides the guard deliberately and on the record:

1. refuses while a study process is running, or if the surgery was already applied;
2. compares the new resolved config with the run's ``config_used.json`` and refuses unless every
   changed key is under :data:`ALLOWED` (nothing phases 0-3 read);
3. backs up ``state.json`` and renames (never deletes) what phases 4+ wrote -- the utility models,
   decision logs, evaluation / policy / stream reports -- with the suffix ``.pre_d029``;
4. removes phase 4+ completion and the per-task decision units, and KEEPS every measurement: the
   Phase 5 adapters (``P5:train:*``), every measured option (``P5:<task>:option:*``) and the
   evaluation / loss caches, none of which depend on the utility model;
5. writes the new config hash and a ``surgeries`` record into the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.experiment import add_common_arguments, resolve_from_args  # noqa: E402

DECISION = "D-029"
SUFFIX = ".pre_d029"
#: Config keys D-029 may change.  Phases 0-3 read none of them.
ALLOWED = ("full_study.policies", "full_study.utility_models")
#: Phases whose results depend on the utility model, and are therefore redone.
REDO_PHASES = ("P4", "P5", "P6", "P7")


def flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(tree, Mapping):
        out: dict[str, Any] = {}
        for key, value in tree.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: tree}


def changed_keys(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    a, b = flatten(old), flatten(new)
    return sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))


def is_allowed(key: str) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in ALLOWED)


def units_to_remove(units: Mapping[str, Any]) -> list[str]:
    """Phase 4+ completions, per-task Phase 5 decision payloads and stream units; never measurements."""
    out = []
    for key in units:
        head, _, rest = key.partition(":")
        if key in {f"phase:{p}" for p in REDO_PHASES}:
            out.append(key)
        elif head == "P5" and rest and ":" not in rest:            # P5:<task> (decisions + tables)
            out.append(key)
        elif head in ("P6", "P7"):
            out.append(key)
    return sorted(out)


def study_running() -> list[int]:
    try:
        import psutil
    except ImportError:                                            # pragma: no cover
        return []
    me = os.getpid()
    running = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = proc.info["cmdline"] or []
        # A python process whose script is run_full_study.py -- not a shell whose command text
        # merely mentions it (which would match every ssh / tmux wrapper).
        if (proc.info["pid"] != me and cmd and "python" in Path(cmd[0]).name
                and any(Path(part).name == "run_full_study.py" for part in cmd[1:3])):
            running.append(proc.info["pid"])
    return running


def plan(run_dir: Path, adapters_dir: Path, cfg: Any) -> dict[str, Any]:
    state_path = run_dir / "decision_study" / "checkpoint" / "state.json"
    if not state_path.exists():
        raise SystemExit(f"no checkpoint at {state_path}")
    state = json.loads(state_path.read_text())
    if any(s.get("decision") == DECISION for s in state.get("surgeries", [])):
        raise SystemExit(f"{DECISION} surgery was already applied to {state_path}")
    used = json.loads((run_dir / "config_used.json").read_text())
    if used.get("config_hash") != state.get("config_hash"):
        raise SystemExit("config_used.json does not belong to this checkpoint (hash mismatch); refusing")
    if (used.get("run_mode"), used.get("experiment_id")) != (cfg.run_mode.value, cfg.experiment_id):
        raise SystemExit(f"run mode / experiment id differ from the run's "
                         f"({used.get('run_mode')}, {used.get('experiment_id')}); refusing")
    changed = changed_keys(used["config"], cfg.raw)
    outside = [k for k in changed if not is_allowed(k)]
    if outside:
        raise SystemExit(f"config changes outside {ALLOWED}: {outside}; refusing")
    if not changed:
        raise SystemExit("the config is unchanged; nothing to do")
    archive = [path for path in (
        run_dir / "utility_network",
        run_dir / "decisions.jsonl",
        run_dir / "decision_study" / "utility_supervision.json",
        run_dir / "decision_study" / "policy_comparison.json",
        adapters_dir / "_stream",
        *sorted((run_dir / "decision_study").glob("evaluation_*.json")),
        *sorted((run_dir / "decision_study").glob("stream_*.json")),
    ) if path.exists()]
    clashes = [str(p) for p in archive if Path(str(p) + SUFFIX).exists()]
    if clashes or Path(str(state_path) + SUFFIX).exists():
        raise SystemExit(f"backup targets already exist: {clashes or [str(state_path) + SUFFIX]}; refusing")
    return {"state_path": state_path, "state": state, "old_hash": state["config_hash"], "new_hash": cfg.config_hash,
            "changed_keys": changed, "remove_units": units_to_remove(state["units"]),
            "remove_phases": [p for p in REDO_PHASES if p in state.get("phases", {})], "archive": archive}


def apply(p: Mapping[str, Any]) -> None:
    state, state_path = dict(p["state"]), p["state_path"]
    (Path(str(state_path) + SUFFIX)).write_text(state_path.read_text())
    for path in p["archive"]:
        path.rename(Path(str(path) + SUFFIX))
    for key in p["remove_units"]:
        state["units"].pop(key, None)
    for phase in p["remove_phases"]:
        state["phases"].pop(phase, None)
    state["config_hash"] = p["new_hash"]
    state.setdefault("surgeries", []).append({
        "decision": DECISION, "at_utc": datetime.now(timezone.utc).isoformat(),
        "old_config_hash": p["old_hash"], "new_config_hash": p["new_hash"], "changed_keys": p["changed_keys"],
        "removed_units": p["remove_units"], "removed_phases": p["remove_phases"],
        "archived": [str(x) + SUFFIX for x in p["archive"]],
        "kept": "P5:train:* adapters, P5:<task>:option:* measurements, evaluation and loss caches"})
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str))
    os.replace(tmp, state_path)


def main() -> int:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    args = parser.parse_args()
    cfg = resolve_from_args(args)
    running = study_running()
    if running:
        raise SystemExit(f"a study process is running (pids {running}); stop it first")
    run_dir = Path(args.results_root) / cfg.run_mode.value / cfg.experiment_id
    adapters_dir = Path("adapters") / cfg.run_mode.value / cfg.experiment_id
    p = plan(run_dir, adapters_dir, cfg)
    print(json.dumps({k: (v if k != "state" else "...") for k, v in p.items()}, indent=2, default=str))
    if not args.apply:
        print("\ndry run: nothing changed (pass --apply)")
        return 0
    apply(p)
    print(f"\n{DECISION} surgery applied; resume with run_full_study.py ... --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

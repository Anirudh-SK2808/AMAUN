"""Proof that a run reproduces another run's splits for the tasks whose adapters it reuses.

l4_full_study loads seed-sum / seed-qa / seed-sentiment from run 1 instead of retraining them, so
their VALIDATION (labels), SUPPORT (probes, features) and TRAIN (KD retention) rows must be run
1's rows exactly.  Run 1 recorded no per-row hashes (D-022), so the check uses every piece of
evidence it did record, and fails on the first mismatch:

1. the data config of each task equals the reference run's ``config_used.json``;
2. every provider load's snapshot key equals the reference's (``decision_study/data.json``);
3. split sizes are equal;
4. per-split row digests are equal, when the reference has them (runs from this change on);
5. the TRAIN token count (prompt + target after the task's truncation, i.e. the attention-mask sum
   training logged as ``tokens_seen`` over exactly one epoch) equals the reference adapter's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from amaun.data.preprocess import tokenize_example
from amaun.data.registry import TaskSpec
from amaun.data.splits import SplitName, TaskSplits

#: Data-config keys that determine a task's rows; compared verbatim against the reference run.
_GLOBAL_DATA_KEYS = ("provider", "seed", "split_ratios", "split_caps", "split_caps_strict", "sampling",
                     "shuffle_buffer_size", "smoke_max_examples_per_split", "synthetic_pool_size",
                     "synthetic_test_size")
_PER_TASK_DATA_KEYS = ("volumes", "task_split_ratios", "task_split_caps", "task_split_counts")


def load_reference(run_dir: str | Path) -> dict[str, Any]:
    """Read a finished run's ``config_used.json``, ``decision_study/data.json`` and adapters."""
    root = Path(run_dir)
    out: dict[str, Any] = {"root": str(root)}
    for key, relative in (("config_used", "config_used.json"), ("data", "decision_study/data.json"),
                          ("initial_adapters", "decision_study/initial_adapters.json")):
        path = root / relative
        out[key] = json.loads(path.read_text()) if path.exists() else None
    return out


def task_data_config(raw: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    """Everything in a resolved config that determines ``task_id``'s rows."""
    data = raw.get("data") or {}
    out: dict[str, Any] = {key: data.get(key) for key in _GLOBAL_DATA_KEYS}
    for key in _PER_TASK_DATA_KEYS:
        out[key] = (data.get(key) or {}).get(task_id)
    out["task"] = ((raw.get("datasets") or {}).get("tasks") or {}).get(task_id)
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def train_token_count(tokenizer: Any, task: TaskSpec, splits: TaskSplits) -> int:
    total = 0
    for example in splits[SplitName.TRAIN]:
        input_ids, _ = tokenize_example(
            tokenizer, example, max_input_tokens=task.max_input_tokens,
            max_target_tokens=task.max_target_tokens, prompt_truncation=task.prompt_truncation,
        )
        total += len(input_ids)
    return total


def verify_reference_splits(
    *,
    raw: Mapping[str, Any],
    tasks: Mapping[str, TaskSpec],
    splits: Mapping[str, TaskSplits],
    reference: Mapping[str, Any],
    task_ids: Sequence[str],
    tokenizer: Any | None,
) -> dict[str, Any]:
    """Run every check for every task; ``ok`` only if all pass."""
    ref_config = ((reference.get("config_used") or {}).get("config")) or {}
    ref_data = reference.get("data") or {}
    ref_adapters = {a["task_id"]: a for a in (reference.get("initial_adapters") or [])}
    checks: list[dict[str, Any]] = []

    def check(task_id: str, name: str, ok: bool, **detail: Any) -> None:
        checks.append({"task_id": task_id, "check": name, "ok": bool(ok), **detail})

    if not ref_config or not ref_data:
        return {"ok": False, "checks": [], "error": f"reference run incomplete at {reference.get('root')}"}
    for task_id in task_ids:
        if task_id not in splits:
            check(task_id, "present", False, reason="task has no splits in this run")
            continue
        mine, theirs = task_data_config(raw, task_id), task_data_config(ref_config, task_id)
        differing = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        check(task_id, "data_config_identical", not differing, differing_keys=differing)

        provenance = splits[task_id].provenance
        ref_prov = (ref_data.get("provenance") or {}).get(task_id) or {}
        my_keys = [load.get("snapshot_key") for load in provenance.get("loads", [])]
        ref_keys = [load.get("snapshot_key") for load in ref_prov.get("loads", [])]
        check(task_id, "snapshot_keys_identical", my_keys == ref_keys, this_run=my_keys, reference=ref_keys)

        sizes = splits[task_id].counts()
        ref_sizes = (ref_data.get("split_sizes") or {}).get(task_id)
        check(task_id, "split_sizes_identical", sizes == ref_sizes, this_run=sizes, reference=ref_sizes)

        if ref_prov.get("split_digests"):
            digests = provenance.get("split_digests") or splits[task_id].digests()
            check(task_id, "split_row_digests_identical", digests == ref_prov["split_digests"])
        else:
            check(task_id, "split_row_digests_identical", True,
                  note="reference recorded no row digests (D-022); not comparable")

        adapter = ref_adapters.get(task_id)
        if adapter is not None and tokenizer is not None:
            tokens = train_token_count(tokenizer, tasks[task_id], splits[task_id])
            check(task_id, "train_tokens_equal_reference_tokens_seen", tokens == adapter.get("tokens_seen"),
                  this_run=tokens, reference=adapter.get("tokens_seen"), adapter_id=adapter.get("adapter_id"))
        else:
            check(task_id, "train_tokens_equal_reference_tokens_seen", False,
                  reason="no reference adapter record for this task" if adapter is None else "no tokenizer")
    return {"ok": bool(checks) and all(c["ok"] for c in checks), "checks": checks,
            "reference": reference.get("root")}

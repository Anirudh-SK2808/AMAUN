"""M2 -- cross-task, cross-split leakage auditing.

``TaskSplits.assert_disjoint`` covers one task.  This module covers the whole corpus: it
checks every (task, split) pair against every other, which catches the cases a per-task
check cannot -- for example the same Wikipedia paragraph appearing in one task's TRAIN and
another task's TEST, which would inflate forward transfer for reasons that have nothing to
do with adapter management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from amaun.data.splits import SplitName, TaskSplits
from amaun.errors import LeakageError


@dataclass
class LeakageReport:
    pairs_checked: int = 0
    overlaps: dict[str, int] = field(default_factory=dict)
    critical: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.critical

    def as_dict(self) -> dict[str, object]:
        return {
            "pairs_checked": self.pairs_checked,
            "overlaps": self.overlaps,
            "critical": self.critical,
            "clean": self.clean,
        }


# Any split appearing in a TEST-vs-other overlap is critical: it means a decision or a
# training step could have seen evaluation data.
_CRITICAL_WITH_TEST = {SplitName.TRAIN, SplitName.VALIDATION, SplitName.SUPPORT}


def audit(splits_by_task: Mapping[str, TaskSplits], *, raise_on_critical: bool = True) -> LeakageReport:
    """Audit every (task, split) pair for content overlap."""
    report = LeakageReport()
    hashes: dict[tuple[str, SplitName], set[str]] = {}
    for task_id, task_splits in splits_by_task.items():
        task_splits.assert_disjoint()
        for name in SplitName:
            hashes[(task_id, name)] = task_splits.hashes(name)

    keys = sorted(hashes, key=lambda k: (k[0], k[1].value))
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            report.pairs_checked += 1
            overlap = hashes[left] & hashes[right]
            if not overlap:
                continue
            label = f"{left[0]}:{left[1].value} <-> {right[0]}:{right[1].value}"
            report.overlaps[label] = len(overlap)
            involves_test = SplitName.TEST in (left[1], right[1])
            other = left[1] if right[1] is SplitName.TEST else right[1]
            if involves_test and other in _CRITICAL_WITH_TEST:
                report.critical.append(label)

    if report.critical and raise_on_critical:
        raise LeakageError(
            "TEST data overlaps a split that influences training or decisions: "
            + "; ".join(report.critical)
        )
    return report


# --------------------------------------------------------------------------------------
# Content-level audit across tasks (l4_full_study item 23)
# --------------------------------------------------------------------------------------

_POOL_SPLITS = (SplitName.TRAIN, SplitName.VALIDATION, SplitName.SUPPORT)


def content_audit(
    splits_by_task: Mapping[str, TaskSplits],
    *,
    duplicate_groups: Mapping[str, str] | None = None,
    raise_on_critical: bool = True,
) -> LeakageReport:
    """Audit every task pair by *content* hash (prompt + target, no task id).

    :func:`audit` compares ``example_id``, which includes the task id, so it cannot see a row
    shared by two tasks built from the same corpus.  Critical here:

    - any TRAIN row of one task in another task's VALIDATION, SUPPORT or TEST;
    - for a declared duplicate (``duplicate_groups[dup] = original``), any row shared between the
      two training pools (TRAIN / VALIDATION / SUPPORT): the duplicate must come from rows outside
      the original's pool.

    TEST <-> TEST overlap between a duplicate and its original is expected (same TEST set) and is
    reported, not flagged.
    """
    duplicates = dict(duplicate_groups or {})
    report = LeakageReport()
    hashes = {(task, name): splits.content_hashes(name)
              for task, splits in splits_by_task.items() for name in SplitName}
    tasks = sorted(splits_by_task)
    for i, left in enumerate(tasks):
        for right in tasks[i + 1:]:
            related = duplicates.get(left) == right or duplicates.get(right) == left
            for a in SplitName:
                for b in SplitName:
                    report.pairs_checked += 1
                    overlap = hashes[(left, a)] & hashes[(right, b)]
                    if not overlap:
                        continue
                    label = f"{left}:{a.value} <-> {right}:{b.value} (content)"
                    report.overlaps[label] = len(overlap)
                    train_into_other = ((a is SplitName.TRAIN and b is not SplitName.TRAIN)
                                        or (b is SplitName.TRAIN and a is not SplitName.TRAIN))
                    pools_shared = related and a in _POOL_SPLITS and b in _POOL_SPLITS
                    if train_into_other or pools_shared:
                        report.critical.append(label)
    if report.critical and raise_on_critical:
        raise LeakageError("content shared across tasks where it must not be: " + "; ".join(report.critical))
    return report

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

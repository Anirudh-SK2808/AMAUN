"""M2 -- deterministic four-way splits with structural leakage prevention.

Master-prompt S6 and brief S5 require four disjoint splits per task, with SUPPORT being the
*only* thing the manager may see and TEST touched once, by the evaluator.

The split function has two properties that matter more than they look:

1. **Assignment is a pure function of content**, not of row index:
   ``bucket = sha256(task_id | example_hash | seed) mod N``.  Row-index splitting breaks
   silently if the upstream dataset is ever reshuffled or re-versioned, which would
   reassign examples between SUPPORT and TEST across runs -- a leak that no test on a single
   run could detect.

2. **SUPPORT is carved from the training pool.**  It is never drawn from whatever upstream
   split becomes our TEST, so "the manager saw SUPPORT" can never mean "the manager saw
   TEST" (assumption A-05).
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from amaun.errors import DataError, LeakageError, SplitAccessViolation


class SplitName(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    SUPPORT = "support"
    TEST = "test"


# Resolution of the hash bucketing.  Higher = finer-grained ratio control.
_BUCKETS = 10_000


@dataclass
class Example:
    """One preprocessed example.  ``example_id`` is a content hash, hence split-stable."""

    prompt: str
    target: str
    task_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def example_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.task_id.encode())
        digest.update(b"\x00")
        digest.update(self.prompt.encode())
        digest.update(b"\x00")
        digest.update(self.target.encode())
        return digest.hexdigest()

    @property
    def content_hash(self) -> str:
        """Hash of the rendered prompt and target only -- no task id.

        ``example_id`` includes the task id, so two tasks built from the same rows (``qa`` and its
        duplicate ``qa-B``) never collide on it.  Cross-task leakage and row-sharing checks use
        this hash instead (l4_full_study item 23).
        """
        digest = hashlib.sha256()
        digest.update(self.prompt.encode())
        digest.update(b"\x00")
        digest.update(self.target.encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class SplitRatios:
    """Fractions of the *training pool* assigned to train/validation/support.

    TEST is not a fraction of the training pool: it comes from a separate upstream source
    (documented per task in ``TaskSpec.upstream_test_source``), which is what keeps it
    genuinely held out rather than merely partitioned.
    """

    train: float
    validation: float
    support: float

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "SplitRatios":
        missing = [k for k in ("train", "validation", "support") if k not in section]
        if missing:
            raise DataError(f"data.split_ratios missing keys: {missing}")
        ratios = cls(
            train=float(section["train"]),
            validation=float(section["validation"]),
            support=float(section["support"]),
        )
        total = ratios.train + ratios.validation + ratios.support
        if abs(total - 1.0) > 1e-6:
            raise DataError(
                f"split_ratios over the training pool must sum to 1.0, got {total:.6f} "
                f"(TEST comes from a separate upstream source, not from this pool)"
            )
        return ratios


def _bucket(task_id: str, example_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{task_id}|{example_id}|{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _BUCKETS


def assign_split(example: Example, *, seed: int, ratios: SplitRatios) -> SplitName:
    """Assign one training-pool example to TRAIN, VALIDATION, or SUPPORT."""
    bucket = _bucket(example.task_id, example.example_id, seed)
    train_cut = int(round(ratios.train * _BUCKETS))
    val_cut = train_cut + int(round(ratios.validation * _BUCKETS))
    if bucket < train_cut:
        return SplitName.TRAIN
    if bucket < val_cut:
        return SplitName.VALIDATION
    return SplitName.SUPPORT


@dataclass
class TaskSplits:
    """The four splits for one task, plus the provenance of how they were made."""

    task_id: str
    splits: dict[SplitName, list[Example]]
    seed: int
    ratios: SplitRatios
    provenance: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: SplitName | str) -> list[Example]:
        return self.splits[SplitName(name)]

    def counts(self) -> dict[str, int]:
        return {name.value: len(examples) for name, examples in self.splits.items()}

    def hashes(self, name: SplitName) -> set[str]:
        return {example.example_id for example in self.splits[SplitName(name)]}

    def content_hashes(self, name: SplitName) -> set[str]:
        return {example.content_hash for example in self.splits[SplitName(name)]}

    def digests(self) -> dict[str, str]:
        """One sha256 per split over its sorted content hashes: equal digests <=> equal row sets."""
        out = {}
        for name, examples in self.splits.items():
            digest = hashlib.sha256()
            for item in sorted(example.content_hash for example in examples):
                digest.update(item.encode())
            out[name.value] = digest.hexdigest()
        return out

    def assert_disjoint(self) -> None:
        """Invariant I2.1: verify pairwise disjointness by content hash, not by construction.

        Verifying the *result* rather than trusting the split function is the point: a future
        change to preprocessing, deduplication, or upstream source could introduce overlap
        that the partition logic itself would never reveal.
        """
        names = list(self.splits)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = self.hashes(left) & self.hashes(right)
                if overlap:
                    raise LeakageError(
                        f"task '{self.task_id}': splits {left.value} and {right.value} share "
                        f"{len(overlap)} example(s). Example of a leaked id: {sorted(overlap)[0][:16]}"
                    )

    def subsample(self, max_per_split: int, *, seed: int) -> "TaskSplits":
        """idea.md S0.2.1: uniform subsampling across *all four* splits, one config key.

        Deterministic (sorted by example_id then sliced) so a smoke run is reproducible.
        """
        reduced = {}
        for name, examples in self.splits.items():
            ordered = sorted(examples, key=lambda e: _bucket(e.task_id, e.example_id, seed))
            reduced[name] = ordered[:max_per_split]
        return TaskSplits(
            task_id=self.task_id,
            splits=reduced,
            seed=self.seed,
            ratios=self.ratios,
            provenance={**self.provenance, "subsampled_to": max_per_split},
        )

    def cap(self, caps: Mapping[str, int], *, seed: int, strict: bool) -> "TaskSplits":
        """Cap individual splits to exact sizes (``data.split_caps``), deterministically.

        Selection uses the same hash ordering as :meth:`subsample`, so it is reproducible and
        independent of upstream row order.  With ``strict`` a split that is *smaller* than its
        cap raises: a run configured for 6 000 TRAIN rows must not silently train on fewer.
        """
        reduced = dict(self.splits)
        shortfalls = []
        for name, limit in caps.items():
            if limit is None:
                continue
            split = SplitName(name)
            examples = self.splits[split]
            if len(examples) < int(limit):
                shortfalls.append(f"{split.value}: have {len(examples)}, cap {int(limit)}")
            ordered = sorted(examples, key=lambda e: _bucket(e.task_id, e.example_id, seed))
            reduced[split] = ordered[: int(limit)]
        if shortfalls and strict:
            raise DataError(
                f"task '{self.task_id}': split(s) smaller than data.split_caps -- "
                + "; ".join(shortfalls)
                + ". Raise data.volumes.<task>.train_pool / test, or the task's split ratios, "
                "rather than training or evaluating on fewer rows than configured."
            )
        return TaskSplits(
            task_id=self.task_id,
            splits=reduced,
            seed=self.seed,
            ratios=self.ratios,
            provenance={**self.provenance, "split_caps": {str(k): v for k, v in caps.items()},
                        "split_cap_shortfalls": shortfalls},
        )


def build_splits(
    task_id: str,
    *,
    training_pool: Sequence[Example],
    test_pool: Sequence[Example],
    seed: int,
    ratios: SplitRatios,
    provenance: dict[str, Any] | None = None,
    exact_counts: Mapping[str, int] | None = None,
) -> TaskSplits:
    """Partition a training pool three ways and attach a separately-sourced test pool.

    ``exact_counts`` (``data.task_split_counts``) replaces the ratio assignment for a corpus with
    no slack (FinQA: 5 700 + 250 + 250 of 6 203 usable rows): the pool is ordered by the same
    content-keyed bucket hash and cut into consecutive TRAIN / VALIDATION / SUPPORT blocks, so the
    assignment is still a pure function of content.  Too few rows raises.
    """
    splits: dict[SplitName, list[Example]] = {name: [] for name in SplitName}
    if exact_counts:
        wanted = [(SplitName(name), int(exact_counts[name])) for name in ("train", "validation", "support")]
        total = sum(count for _, count in wanted)
        # Identical rows share a bucket, so ratio assignment keeps them together; consecutive cuts
        # could split them across a boundary, so exact mode keeps one copy of each.
        training_pool = list({e.example_id: e for e in training_pool}.values())
        if len(training_pool) < total:
            raise DataError(f"task '{task_id}': exact split counts need {total} rows, the pool has "
                            f"{len(training_pool)}")
        ordered = sorted(training_pool, key=lambda e: (_bucket(e.task_id, e.example_id, seed), e.example_id))
        start = 0
        for name, count in wanted:
            splits[name] = ordered[start : start + count]
            start += count
    else:
        for example in training_pool:
            splits[assign_split(example, seed=seed, ratios=ratios)].append(example)
    splits[SplitName.TEST] = list(test_pool)

    result = TaskSplits(
        task_id=task_id,
        splits=splits,
        seed=seed,
        ratios=ratios,
        provenance={
            "training_pool_size": len(training_pool),
            "test_pool_size": len(test_pool),
            "split_method": ("exact counts over sha256(task_id|content_hash|seed) order"
                             if exact_counts else "sha256(task_id|content_hash|seed) mod 10000"),
            **({"exact_counts": dict(exact_counts)} if exact_counts else {}),
            **(provenance or {}),
        },
    )
    result.assert_disjoint()
    return result


class SplitAccessGuard:
    """A typed capability: holds splits but yields only the permitted ones.

    This is the mechanism behind invariant I2.4.  The manager is handed a guard permitting
    ``{SUPPORT}``; the evaluator one permitting ``{TEST}``.  There is no code path by which
    the manager holds a TEST iterator, so creating a leak requires editing the guard
    construction -- a visible, reviewable act rather than an accident.
    """

    def __init__(
        self,
        splits: TaskSplits,
        permitted: Iterable[SplitName],
        *,
        owner: str,
    ) -> None:
        self._splits = splits
        self.permitted = frozenset(SplitName(p) for p in permitted)
        self.owner = owner
        self.access_log: list[str] = []

    @property
    def task_id(self) -> str:
        return self._splits.task_id

    def get(self, name: SplitName | str) -> list[Example]:
        requested = SplitName(name)
        if requested not in self.permitted:
            raise SplitAccessViolation(
                f"'{self.owner}' may read {sorted(p.value for p in self.permitted)} but "
                f"requested '{requested.value}' for task '{self._splits.task_id}'. "
                f"This would violate the split-isolation contract "
                f"(docs/master_prompt.md S6)."
            )
        self.access_log.append(requested.value)
        return self._splits[requested]

    def counts(self) -> dict[str, int]:
        """Sizes of permitted splits only (a size leak is still a leak)."""
        return {n.value: len(self._splits[n]) for n in self.permitted}


def manager_guard(splits: TaskSplits) -> SplitAccessGuard:
    """The manager sees SUPPORT and nothing else (brief S5)."""
    return SplitAccessGuard(splits, {SplitName.SUPPORT}, owner="manager")


def trainer_guard(splits: TaskSplits) -> SplitAccessGuard:
    """Adapter training sees TRAIN (and VALIDATION for early stopping)."""
    return SplitAccessGuard(splits, {SplitName.TRAIN, SplitName.VALIDATION}, owner="trainer")


def evaluator_guard(splits: TaskSplits) -> SplitAccessGuard:
    """Only the evaluator sees TEST (invariant I12.1)."""
    return SplitAccessGuard(splits, {SplitName.TEST}, owner="evaluator")


def utility_guard(splits: TaskSplits) -> SplitAccessGuard:
    """Utility-network supervision: everything except TEST (idea.md S9)."""
    return SplitAccessGuard(
        splits, {SplitName.TRAIN, SplitName.VALIDATION, SplitName.SUPPORT}, owner="utility"
    )
